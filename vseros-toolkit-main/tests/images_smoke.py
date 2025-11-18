from __future__ import annotations

import tempfile
from pathlib import Path

import importlib.util
import numpy as np
import pandas as pd
from PIL import Image

from common.features.img_index import build_from_dir
from common.features import img_stats

img_embed = None
if importlib.util.find_spec("common.features.img_embed") is not None:
    from common.features import img_embed  # type: ignore


def _make_image(path: Path, color: tuple[int, int, int]):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (32, 24), color=color)
    img.save(path)


def test_img_stats_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ids = ["a", "b"]
        for idx, color in zip(ids, [(255, 0, 0), (0, 255, 0)]):
            _make_image(root / idx / "img1.jpg", color)
            _make_image(root / idx / "nested" / "img2.png", color)

        train_df = pd.DataFrame({"id": ["a", "b"]})
        test_df = pd.DataFrame({"id": ["b", "a"]})

        mapping = build_from_dir(root, ids=train_df["id"].tolist() + test_df["id"].tolist())
        pkg = img_stats.build(train_df, test_df, id_col="id", id_to_images=mapping, use_cache=False)
        assert pkg.train.shape[0] == len(train_df)
        assert pkg.test.shape[0] == len(test_df)
        assert pkg.kind == "dense"

        if img_embed is not None:
            try:
                img_embed.build(
                    train_df,
                    test_df,
                    id_col="id",
                    id_to_images=mapping,
                    backbone="resnet50",
                    image_size=64,
                    batch_size=1,
                    num_workers=0,
                    use_cache=False,
                )
            except Exception:
                # environments without torch/weights can skip
                pass


if __name__ == "__main__":
    test_img_stats_smoke()
