from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal

# Роль сообщения в чат-формате
Role = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    """
    Простое представление одного сообщения в чат-диалоге.

    Пример:
        ChatMessage(role="user", content="Объясни, что такое градиентный бустинг")
    """
    role: Role
    content: str


@dataclass
class LLMConfig:
    """
    Конфиг для LLM-клиента.

    Это то, что ты будешь передавать в разные backend'ы (transformers / vLLM и т.п.),
    чтобы не таскать миллион аргументов по функциям.

    Основная идея:
      - model_name: строка с именем модели (HF repo id, локальный путь и т.п.)
      - max_new_tokens, temperature, top_p, top_k, repetition_penalty, do_sample:
        стандартные параметры семплинга
      - json_mode: если True, мы ожидаем строго JSON-ответ и можем строить соответствующий промпт
      - stop: список стоп-строк (backend сам решает, как их применять)
      - device: "cuda" / "cpu" / "auto"
      - dtype: "auto" / "float16" / "bfloat16" / "int8" и т.п.
      - seed: фиксируем seed для воспроизводимости (если backend это использует)
      - backend_params: любые специфичные штуки под конкретный backend
    """

    # --- Основные параметры модели и семплинга ---

    model_name: str

    max_new_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.0
    do_sample: bool = True

    # --- Формат вывода / управление JSON-режимом ---

    json_mode: bool = False
    stop: Optional[List[str]] = None

    # --- Инфраструктурные настройки ---

    # "cuda", "cpu" или что-то типа "auto" — дальше backend сам решит, как это интерпретировать
    device: str = "cuda"

    # "auto" / "float16" / "bfloat16" / "int8" и т.п. — конкретную интерпретацию делает backend
    dtype: str = "auto"

    # seed может использоваться backend'ом для фиксации генерации
    seed: Optional[int] = 42

    # Дополнительные параметры, специфичные для конкретного backend'а
    backend_params: Dict[str, Any] = field(default_factory=dict)

    def with_overrides(self, **overrides: Any) -> "LLMConfig":
        """
        Удобный метод: сделать копию конфига с переопределёнными полями.

        Пример:
            cfg_strict = base_cfg.with_overrides(
                temperature=0.0,
                do_sample=False,
            )
        """
        data = self.to_dict()
        data.update(overrides)
        return LLMConfig(**data)

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразовать конфиг в обычный словарь (например, чтобы логировать или сохранять).
        """
        return {
            "model_name": self.model_name,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.do_sample,
            "json_mode": self.json_mode,
            "stop": list(self.stop) if self.stop is not None else None,
            "device": self.device,
            "dtype": self.dtype,
            "seed": self.seed,
            "backend_params": dict(self.backend_params),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMConfig":
        """
        Обратная операция к to_dict: удобно, если читаешь конфиг из json/yaml.
        Лишние ключи можно сначала отфильтровать снаружи.
        """
        return cls(**data)


__all__ = [
    "Role",
    "ChatMessage",
    "LLMConfig",
]
