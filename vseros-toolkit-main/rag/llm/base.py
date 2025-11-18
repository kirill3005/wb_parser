from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from .config import LLMConfig, ChatMessage


# Удобный алиас для значений, которые мы ожидаем от JSON-ответов
JsonValue = Union[Dict[str, Any], List[Any]]


class LLMClient(ABC):
    """
    Базовый абстрактный класс для всех LLM-бэкендов (transformers, vLLM, llama.cpp и т.п.).

    Основная идея:
      - снаружи весь код (RAG-пайплайны) работают только с этим интерфейсом;
      - конкретные реализации прячут внутри себя детали загрузки модели,
        устройство батчинга, quant, LoRA и прочую магию.

    Минимальный контракт:
      - generate(...)      — обычный completion: prompt -> текст
      - generate_chat(...) — chat-режим: список сообщений -> ответ assistant
      - generate_json(...) — high-level helper для "верни JSON" (поверх generate)
    """

    def __init__(self, config: LLMConfig):
        """
        :param config: базовая конфигурация модели (имя, параметры семплинга и т.д.)
        """
        self.config = config

    # -------------------------------------------------------------------------
    # Абстрактные методы: реализация обязательна в каждом backend'е
    # -------------------------------------------------------------------------

    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: Optional[LLMConfig] = None,
    ) -> str:
        """
        Сгенерировать ответ по одному prompt'у (completion-режим).

        :param prompt: строка с полным prompt'ом
        :param config: при необходимости можно передать временный override конфига
        :return: сгенерированный текст (без исходного prompt'а)
        """
        raise NotImplementedError

    @abstractmethod
    def generate_chat(
        self,
        messages: List[ChatMessage],
        config: Optional[LLMConfig] = None,
    ) -> str:
        """
        Сгенерировать ответ в chat-режиме.

        :param messages: список сообщений (system/user/assistant)
        :param config: временный override конфига (если нужен)
        :return: текст ответа от "assistant"
        """
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Дополнительные удобные методы (не обязаны переопределяться)
    # -------------------------------------------------------------------------

    def generate_json(
        self,
        prompt: str,
        schema_hint: Optional[str] = None,
        config: Optional[LLMConfig] = None,
    ) -> JsonValue:
        """
        High-level helper: заставить модель вернуть JSON и распарсить его.

        Как это работает:
          1. Оборачивает user_prompt в промпт вида
             "ответь строго в формате JSON, вот схема: ..., вот задание: ..."
             (см. json_utils.build_json_prompt).
          2. Вызывает self.generate(...) с этим промптом.
          3. Пытается распарсить ответ как JSON, с небольшой "починкой", если нужно.

        :param prompt: текст задания (user_prompt), без обёртки под JSON
        :param schema_hint: текстовое описание ожидаемой схемы JSON
                            (например, пример структуры или подсказка полей)
        :param config: опциональный override LLMConfig
        :return: dict или list — разобранный JSON-ответ
        """
        from .json_utils import generate_and_parse_json

        effective_config = config or self.config
        return generate_and_parse_json(
            client=self,
            user_prompt=prompt,
            schema_hint=schema_hint,
            config=effective_config,
        )


__all__ = [
    "LLMClient",
    "JsonValue",
]
