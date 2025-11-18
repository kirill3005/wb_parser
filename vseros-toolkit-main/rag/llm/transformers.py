from __future__ import annotations

from typing import Any, Dict, List, Optional

import random

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .base import LLMClient
from .config import ChatMessage, LLMConfig


class TransformersLLMClient(LLMClient):
    """
    Реализация LLMClient на базе transformers (AutoModelForCausalLM + AutoTokenizer).

    Идея:
      - Внутри умеет грузить модель и токенайзер по model_name,
      - Поддерживает generate / generate_chat,
      - Учитывает базовые параметры из LLMConfig (temperature, top_p, топ_k и т.д.),
      - Позволяет слегка управлять dtype / device через config.

    Внешний код (RAG-пайплайны) с этим классом не завязан на детали transformers.
    """

    def __init__(
        self,
        config: LLMConfig,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> None:
        super().__init__(config)

        # --- Seed для воспроизводимости (насколько это возможно) ---
        self._ensure_seed(config.seed)

        # --- Токенайзер ---
        backend_params: Dict[str, Any] = dict(config.backend_params or {})
        trust_remote_code: bool = bool(backend_params.pop("trust_remote_code", False))

        if tokenizer is None:
            self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
                config.model_name,
                use_fast=True,
                trust_remote_code=trust_remote_code,
            )
        else:
            self.tokenizer = tokenizer

        # Гарантируем наличие pad_token
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # Самый крайний случай — используем unk как pad
                self.tokenizer.pad_token = self.tokenizer.unk_token

        # --- Модель ---
        dtype = self._resolve_dtype(config.dtype)

        # Если device_map не задан в backend_params — выставим базовый по device
        device_map = backend_params.pop("device_map", None)
        if device_map is None:
            if config.device in ("cuda", "auto"):
                device_map = "auto"
            else:
                # Для CPU просто грузим без device_map, дальше .to("cpu") при желании
                device_map = None

        if model is None:
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
                **backend_params,
            )
        else:
            self.model = model

        # Если явно хотим CPU — перенесём модель
        if config.device == "cpu":
            self.model.to("cpu")

        self.model.eval()

    # -------------------------------------------------------------------------
    # Обязательные методы интерфейса
    # -------------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        config: Optional[LLMConfig] = None,
    ) -> str:
        """
        Обычный completion-режим: один prompt → один ответ.
        """
        cfg = config or self.config

        # Токенизация с учётом максимальной длины модели
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_input_length(),
        )

        device = self._get_model_device()
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs = self._build_generate_kwargs(cfg, inputs)

        with torch.no_grad():
            output_ids = self.model.generate(**gen_kwargs)

        # Обрезаем префикс prompt'а
        generated = output_ids[0, inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        text = text.strip()

        # Применяем stop-строки (если заданы)
        text = self._apply_stop_strings(text, cfg.stop)

        return text

    def generate_chat(
        self,
        messages: List[ChatMessage],
        config: Optional[LLMConfig] = None,
    ) -> str:
        """
        Chat-режим: список сообщений (system/user/assistant) → ответ assistant.
        """
        cfg = config or self.config

        # Если токенайзер умеет chat_template — используем его
        if hasattr(self.tokenizer, "apply_chat_template"):
            # transformers >= 4.36: apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            chat_dicts = [{"role": m.role, "content": m.content} for m in messages]
            prompt = self.tokenizer.apply_chat_template(
                chat_dicts,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Простейший fallback: сами собираем "чат"
            parts: List[str] = []
            for m in messages:
                role = m.role.upper()
                parts.append(f"[{role}]\n{m.content}\n")
            parts.append("[ASSISTANT]\n")
            prompt = "\n".join(parts)

        return self.generate(prompt, cfg)

    # -------------------------------------------------------------------------
    # Внутренние хелперы
    # -------------------------------------------------------------------------

    def _resolve_dtype(self, dtype_str: str) -> torch.dtype:
        """
        Маппинг строки из LLMConfig.dtype → torch.dtype.
        """
        dtype_str = (dtype_str or "auto").lower()

        if dtype_str in ("auto", ""):
            # Простая эвристика: если есть CUDA — берём float16, иначе float32
            return torch.float16 if torch.cuda.is_available() else torch.float32
        if dtype_str in ("float16", "fp16", "half"):
            return torch.float16
        if dtype_str in ("bfloat16", "bf16"):
            return torch.bfloat16
        if dtype_str in ("float32", "fp32"):
            return torch.float32

        # По умолчанию — float16 (часто разумно для GPU)
        return torch.float16

    def _ensure_seed(self, seed: Optional[int]) -> None:
        """
        Фиксируем seed для random / numpy / torch (по максимуму).
        """
        if seed is None:
            return

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _get_model_device(self) -> torch.device:
        """
        Возвращает устройство, на котором живёт модель.
        """
        # В большинстве случаев параметр device у первого параметра модели — это нужное нам устройство
        try:
            return next(self.model.parameters()).device  # type: ignore[arg-type]
        except StopIteration:
            # Теоретически модель без параметров — крайне странный кейс, но на всякий случай
            return torch.device("cpu")

    def _max_input_length(self) -> int:
        """
        Определяет максимальную длину входа для токенайзера.
        """
        # У разных токенайзеров model_max_length может быть "огромным" (int(1e30)),
        # поэтому стоит ограничить сверху.
        max_len = getattr(self.tokenizer, "model_max_length", 2048)
        if max_len is None or max_len > 10_000_000:
            # Если значение "мнимое" — выберем что-то разумное
            return 4096
        return int(max_len)

    def _build_generate_kwargs(
        self,
        cfg: LLMConfig,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Собирает kwargs для model.generate() на основе LLMConfig и входных тензоров.
        """
        gen_kwargs: Dict[str, Any] = dict(inputs)

        # Параметры семплинга
        gen_kwargs["max_new_tokens"] = cfg.max_new_tokens
        gen_kwargs["do_sample"] = bool(cfg.do_sample)

        if cfg.do_sample:
            gen_kwargs["temperature"] = cfg.temperature
            gen_kwargs["top_p"] = cfg.top_p
            gen_kwargs["top_k"] = cfg.top_k
        else:
            # В детерминированном режиме temperature / top_p / top_k не критичны,
            # но некоторые модели требуют валидных значений.
            gen_kwargs["temperature"] = 0.0
            gen_kwargs["top_p"] = 1.0
            gen_kwargs["top_k"] = 0

        gen_kwargs["repetition_penalty"] = cfg.repetition_penalty

        # EOS / PAD токены
        gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        eos_ids = self._get_eos_token_ids()
        if eos_ids is not None:
            gen_kwargs["eos_token_id"] = eos_ids

        return gen_kwargs

    def _get_eos_token_ids(self) -> Optional[List[int]]:
        """
        Возвращает список eos_token_id (если есть), иначе None.
        """
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            return None
        return [int(eos_id)]

    def _apply_stop_strings(
        self,
        text: str,
        stop: Optional[List[str]],
    ) -> str:
        """
        После генерации обрезает текст по первой встреченной stop-строке.

        Это не идеально (лучше реализовывать stopping_criteria в generate),
        но для олимпиадного кода так проще и быстрее.
        """
        if not stop:
            return text

        cut_positions: List[int] = []
        for s in stop:
            if not s:
                continue
            idx = text.find(s)
            if idx != -1:
                cut_positions.append(idx)

        if not cut_positions:
            return text

        cut = min(cut_positions)
        return text[:cut].rstrip()


__all__ = [
    "TransformersLLMClient",
]
