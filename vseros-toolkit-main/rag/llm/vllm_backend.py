from __future__ import annotations

from typing import Any, Dict, List, Optional

import random

import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .base import LLMClient
from .config import ChatMessage, LLMConfig


class VLLMLLMClient(LLMClient):
    """
    Реализация LLMClient на базе vLLM (локальный GPU-оптимизированный backend).

    Идея:
      - vLLM берёт на себя всё тяжёлое по KV-кэшу, батчингу и скорости;
      - мы даём единый интерфейс generate / generate_chat;
      - конфиг LLMConfig задаёт семплинг, модель, dtype и т.п.
    """

    def __init__(
        self,
        config: LLMConfig,
        llm: Optional[LLM] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> None:
        super().__init__(config)

        # Фиксируем seed (насколько это возможно) для random / numpy.
        self._ensure_seed(config.seed)

        backend_params: Dict[str, Any] = dict(config.backend_params or {})
        trust_remote_code: bool = bool(backend_params.pop("trust_remote_code", False))

        # --- Токенайзер: нужен для chat_template и, при желании, для system/user-разметки ---
        if tokenizer is None:
            self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
                config.model_name,
                use_fast=True,
                trust_remote_code=trust_remote_code,
            )
        else:
            self.tokenizer = tokenizer

        # --- vLLM LLM ---
        if llm is None:
            llm_kwargs: Dict[str, Any] = dict(backend_params)
            llm_kwargs.setdefault("model", config.model_name)

            # dtype в vLLM — строка ("float16", "bfloat16", "float32" и т.п.)
            dtype_str = self._resolve_dtype_str(config.dtype)
            if dtype_str is not None:
                llm_kwargs.setdefault("dtype", dtype_str)

            # Здесь можно передать tensor_parallel_size, gpu_memory_utilization и т.п. через backend_params
            self.llm: LLM = LLM(**llm_kwargs)
        else:
            self.llm = llm

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
        sampling_params = self._make_sampling_params(cfg)

        # vLLM умеет batched-генерацию, но здесь мы даём один prompt
        outputs = self.llm.generate(
            [prompt],
            sampling_params=sampling_params,
        )

        # Один запрос → один RequestOutput → один candidate
        text = outputs[0].outputs[0].text
        text = text.strip()

        # На всякий случай ещё раз применим stop-строки (если заданы),
        # хотя vLLM сам должен их учесть.
        text = self._apply_stop_strings(text, cfg.stop)

        return text

    def generate_chat(
        self,
        messages: List[ChatMessage],
        config: Optional[LLMConfig] = None,
    ) -> str:
        """
        Chat-режим: список сообщений (system/user/assistant) → ответ assistant.

        Мы здесь только собираем правильный prompt, а дальше зовём generate().
        """
        cfg = config or self.config

        # Если токенайзер умеет chat_template — используем его.
        if hasattr(self.tokenizer, "apply_chat_template"):
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

    def _make_sampling_params(self, cfg: LLMConfig) -> SamplingParams:
        """
        Собираем SamplingParams для vLLM из LLMConfig.

        Здесь мы не пытаемся учесть абсолютно все параметры vLLM, только те,
        которые реально нужны в олимпиадном пайплайне.
        """
        if cfg.do_sample:
            temperature = cfg.temperature
            top_p = cfg.top_p
            top_k = cfg.top_k
        else:
            # Детерминированный режим: температура 0, top_p=1
            temperature = 0.0
            top_p = 1.0
            # top_k=-1 в vLLM обычно означает "не ограничивать по top-k"
            top_k = -1

        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=cfg.max_new_tokens,
            stop=cfg.stop or None,
            repetition_penalty=cfg.repetition_penalty,
            n=1,
            seed=cfg.seed,
        )
        return params

    def _resolve_dtype_str(self, dtype_str: str) -> Optional[str]:
        """
        Преобразуем LLMConfig.dtype в строку, которую ожидает vLLM.

        vLLM обычно понимает:
          - "float16"
          - "bfloat16"
          - "float32"
        """
        if not dtype_str:
            return None

        s = dtype_str.lower()
        if s in ("auto",):
            return None
        if s in ("float16", "fp16", "half"):
            return "float16"
        if s in ("bfloat16", "bf16"):
            return "bfloat16"
        if s in ("float32", "fp32"):
            return "float32"

        # Если что-то экзотическое — не навязываем, пусть vLLM сам решает
        return None

    def _ensure_seed(self, seed: Optional[int]) -> None:
        """
        Фиксируем seed для random / numpy.
        (У vLLM есть свой seed в SamplingParams, его мы тоже задаём.)
        """
        if seed is None:
            return

        random.seed(seed)
        np.random.seed(seed)

    def _apply_stop_strings(
        self,
        text: str,
        stop: Optional[List[str]],
    ) -> str:
        """
        После генерации обрезает текст по первой встреченной stop-строке.

        Это запасной вариант на случай, если backend не полностью учёл stop.
        Для vLLM это, по идее, лишнее, но дёшево и безопасно.
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
    "VLLMLLMClient",
]
