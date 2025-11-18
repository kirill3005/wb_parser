from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
import pandas as pd

from common.cache import load_feature_pkg, make_key, save_feature_pkg
from common.features.types import FeaturePackage

if importlib.util.find_spec("torch") is not None:  # pragma: no cover - optional dependency
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms
else:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    DataLoader = None  # type: ignore
    Dataset = object  # type: ignore
    transforms = None  # type: ignore
    models = None  # type: ignore
    _IMPORT_ERROR = ImportError("torch is not installed")

if importlib.util.find_spec("open_clip") is not None:  # pragma: no cover - optional dependency
    import open_clip  # type: ignore
else:  # pragma: no cover - optional dependency
    open_clip = None

Backbone = Literal["resnet50", "mobilenet_v3_small", "vit_b_16", "clip_vit_b32"]


class _ImageDataset(Dataset):
    def __init__(self, items: List[str], transform):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path = self.items[idx]
        from PIL import Image

        with Image.open(path) as img:
            img = img.convert("RGB")
            return self.transform(img)


def _fingerprint() -> str:
    text = Path(__file__).read_text(encoding="utf-8")
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def _device(auto: Literal["auto", "cpu", "cuda"]):
    if auto == "cpu":
        return "cpu"
    if auto == "cuda":
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"


def _precision(auto: Literal["auto", "fp32", "fp16", "bf16"], device: str) -> str:
    if auto != "auto":
        return auto
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return "bf16"
        return "fp16"
    return "fp32"


def _gem(x, p: float = 3.0, eps: float = 1e-6):
    return torch.mean((x.clamp(min=eps)) ** p, dim=(-2, -1)) ** (1.0 / p)


def _build_transform(image_size: int, normalize: Literal["imagenet", "clip"]):
    if transforms is None:
        raise RuntimeError("torchvision is required for transforms")
    if normalize == "imagenet":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
    return transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _build_model(backbone: Backbone, pool: Literal["avg", "gem", "token"]):
    if torch is None or models is None:
        raise RuntimeError(
            "torch/torchvision not available. Install torch or use img_stats.build as a fallback."
        )

    if backbone == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if hasattr(models, "ResNet50_Weights") else None
        base = models.resnet50(weights=weights)
        feature_dim = base.fc.in_features
        encoder = nn.Sequential(*(list(base.children())[:-2]))

        def forward(x):
            feat = encoder(x)
            if pool == "gem":
                pooled = _gem(feat)
            else:
                pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            return pooled

        return forward, feature_dim

    if backbone == "mobilenet_v3_small":
        weights = (
            models.MobileNet_V3_Small_Weights.DEFAULT
            if hasattr(models, "MobileNet_V3_Small_Weights")
            else None
        )
        base = models.mobilenet_v3_small(weights=weights)
        feature_dim = (
            base.classifier[0].in_features if hasattr(base.classifier[0], "in_features") else 576
        )
        encoder = base.features

        def forward(x):
            feat = encoder(x)
            if pool == "gem":
                pooled = _gem(feat)
            else:
                pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            return pooled

        return forward, feature_dim

    if backbone == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT if hasattr(models, "ViT_B_16_Weights") else None
        base = models.vit_b_16(weights=weights)
        feature_dim = base.hidden_dim

        def forward(x):
            feats = base.forward_features(x)
            if pool == "token":
                pooled = feats[:, 0]
            else:
                pooled = feats[:, 1:].mean(dim=1)
            return pooled

        return forward, feature_dim

    if backbone == "clip_vit_b32":
        if open_clip is None:
            raise RuntimeError("open_clip is required for clip_vit_b32 backbone")
        model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        feature_dim = model.visual.output_dim

        def forward(x):
            return model.encode_image(x)

        return forward, feature_dim

    raise ValueError(f"Unsupported backbone: {backbone}")


def _aggregate_embeddings(
    embeds: List[np.ndarray], agg: Literal["mean", "max", "gem", "first"], p: float = 3.0
) -> np.ndarray:
    arr = np.stack(embeds, axis=0)
    if agg == "mean":
        return arr.mean(axis=0)
    if agg == "max":
        return arr.max(axis=0)
    if agg == "first":
        return arr[0]
    if agg == "gem":
        return np.power(np.mean(np.power(np.clip(arr, 1e-6, None), p), axis=0), 1.0 / p)
    raise ValueError(f"Unsupported agg: {agg}")


def _read_partial(base: Path, split: str) -> Dict[str, np.ndarray]:
    data: Dict[str, np.ndarray] = {}
    for part in sorted(base.glob(f"{split}.part_*.parquet")):
        df = pd.read_parquet(part)
        for idx, row in df.iterrows():
            data[str(idx)] = row.to_numpy()
    return data


def _save_part(base: Path, split: str, buffer: Dict[str, np.ndarray], part_idx: int):
    if not buffer:
        return
    df = pd.DataFrame(buffer).T
    df.to_parquet(base / f"{split}.part_{part_idx:03d}.parquet")
    buffer.clear()


def _finalize_split(base: Path, split: str, all_embeddings: Dict[str, np.ndarray], prefix: str, dtype: str):
    df = pd.DataFrame(all_embeddings).T
    df.columns = [f"{prefix}__{i:04d}" for i in range(df.shape[1])]
    df = df.astype(dtype)
    df.to_parquet(base / f"{split}.parquet")
    for part in base.glob(f"{split}.part_*.parquet"):
        part.unlink()
    return df


def _data_stamp(train_df: pd.DataFrame, test_df: pd.DataFrame, id_to_images: Dict[str, List[str]]):
    first_items = sorted(id_to_images.items())[:200]
    payload = "|".join(f"{k}:{v[0] if v else ''}" for k, v in first_items)
    return {
        "train_ids": len(train_df),
        "test_ids": len(test_df),
        "total_paths": sum(len(v) for v in id_to_images.values()),
        "first_hash": hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8],
    }


def _run_inference(
    model_fn,
    feature_dim: int,
    device: str,
    precision: str,
    images: List[str],
    transform,
    batch_size: int,
    num_workers: int,
):
    if torch is None:
        raise RuntimeError("torch not available")

    ds = _ImageDataset(images, transform)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    model = model_fn
    torch_device = torch.device(device)

    if not isinstance(model, torch.nn.Module):
        class _Wrapper(nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            def forward(self, x):
                return self.fn(x)

        model = _Wrapper(model_fn)

    model.eval()
    model.to(torch_device)

    autocast_dtype = None
    if precision == "fp16":
        autocast_dtype = torch.float16
    elif precision == "bf16":
        autocast_dtype = torch.bfloat16

    outputs: List[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(torch_device)
            ctx = (
                torch.autocast(device_type=torch_device.type, dtype=autocast_dtype)
                if autocast_dtype
                else torch.no_grad()
            )
            with ctx:
                feats = model(batch)
            feats = feats.detach().cpu().numpy()
            outputs.extend(feats)
    return outputs


def _prepare_cached_pkg(cache_key: str):
    return load_feature_pkg("img_embed", cache_key)


def build(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_col: str,
    id_to_images: Dict[str, List[str]],
    *,
    backbone: Backbone = "resnet50",
    image_size: int = 224,
    agg: Literal["mean", "max", "gem", "first"] = "mean",
    pool: Literal["avg", "gem", "token"] = "avg",
    batch_size: int = 64,
    num_workers: int = 2,
    device: Literal["auto", "cpu", "cuda"] = "auto",
    precision: Literal["auto", "fp32", "fp16", "bf16"] = "auto",
    normalize: Literal["imagenet", "clip"] = "imagenet",
    prefix: str = "img",
    dtype: Literal["float32", "float16"] = "float16",
    checkpoint_every: int = 2000,
    use_cache: bool = True,
    cache_key_extra: dict | None = None,
) -> FeaturePackage:
    """CNN/ViT/CLIP embeddings builder.

    Пример::
        from common.features import store
        from common.features.img_index import build_from_dir
        from common.features.img_embed import build as img_build

        ids_train = train["id"].astype(str)
        ids_test = test["id"].astype(str)
        id2imgs = build_from_dir("data/images", pd.concat([ids_train, ids_test]).unique(), pattern="{id}/*.jpg", max_per_id=4)

        FS = store.FeatureStore()
        FS.add(img_build(train, test, id_col="id", id_to_images=id2imgs,
                         backbone="resnet50", image_size=224, agg="mean", pool="avg",
                         batch_size=64, device="auto", precision="auto", dtype="float16"))
    """

    if torch is None:
        raise RuntimeError(
            "torch/torchvision not available. Install torch to use img_embed or rely on img_stats.build as fallback."
        ) from _IMPORT_ERROR

    params = {
        "backbone": backbone,
        "image_size": image_size,
        "agg": agg,
        "pool": pool,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "device": device,
        "precision": precision,
        "normalize": normalize,
        "dtype": dtype,
    }
    data_stamp = _data_stamp(train_df, test_df, id_to_images)
    if cache_key_extra:
        data_stamp.update(cache_key_extra)

    cache_key = make_key(params, _fingerprint(), data_stamp)
    if use_cache:
        cached = _prepare_cached_pkg(cache_key)
        if cached is not None:
            return cached

    base = Path("artifacts/features/img_embed") / cache_key
    base.mkdir(parents=True, exist_ok=True)

    device_resolved = _device(device)
    precision_resolved = _precision(precision, device_resolved)
    if device_resolved == "cuda":
        torch.backends.cudnn.benchmark = True

    transform = _build_transform(image_size, normalize)
    model_fn, feature_dim = _build_model(backbone, pool)

    train_ids_order = train_df[id_col].astype(str).tolist()
    test_ids_order = test_df[id_col].astype(str).tolist()
    unique_train_ids = list(dict.fromkeys(train_ids_order))
    unique_test_ids = list(dict.fromkeys(test_ids_order))

    processed_train = _read_partial(base, "train") if base.exists() else {}
    processed_test = _read_partial(base, "test") if base.exists() else {}
    buffer_train: Dict[str, np.ndarray] = {}
    buffer_test: Dict[str, np.ndarray] = {}
    part_idx = 0

    def process_split(ids: List[str], processed: Dict[str, np.ndarray], buffer: Dict[str, np.ndarray], split: str):
        nonlocal part_idx
        for sid in ids:
            if sid in processed:
                continue
            imgs = id_to_images.get(str(sid), [])
            valid_imgs = [p for p in imgs if Path(p).exists()]
            if not valid_imgs:
                processed[sid] = np.zeros(feature_dim, dtype=dtype)
            else:
                try:
                    embed_list = _run_inference(
                        model_fn,
                        feature_dim,
                        device_resolved,
                        precision_resolved,
                        valid_imgs,
                        transform,
                        batch_size,
                        num_workers,
                    )
                    if not embed_list:
                        processed[sid] = np.zeros(feature_dim, dtype=dtype)
                    else:
                        agg_vec = _aggregate_embeddings([np.asarray(e) for e in embed_list], agg)
                        processed[sid] = agg_vec.astype(dtype)
                except Exception:
                    processed[sid] = np.zeros(feature_dim, dtype=dtype)
            buffer[sid] = processed[sid]
            if len(buffer) >= checkpoint_every:
                _save_part(base, split, buffer, part_idx)
                part_idx += 1
        _save_part(base, split, buffer, part_idx)
        part_idx += 1

    t0 = time.time()
    process_split(unique_train_ids, processed_train, buffer_train, "train")
    process_split(unique_test_ids, processed_test, buffer_test, "test")

    train_df_embeddings = _finalize_split(base, "train", processed_train, prefix, dtype)
    test_df_embeddings = _finalize_split(base, "test", processed_test, prefix, dtype)

    train = train_df_embeddings.loc[train_ids_order].reset_index(drop=True)
    test = test_df_embeddings.loc[test_ids_order].reset_index(drop=True)

    cols = list(train.columns)
    meta = {
        "name": "img_embed",
        "params": params,
        "time_sec": round(time.time() - t0, 3),
        "cache_key": cache_key,
        "deps": ["img_index"],
        "backbone": backbone,
        "device": device_resolved,
        "oof": False,
    }

    pkg = FeaturePackage(
        name="img_embed",
        train=train,
        test=test,
        kind="dense",
        cols=cols,
        meta=meta,
    )

    if use_cache:
        save_feature_pkg("img_embed", cache_key, pkg)
    return pkg
