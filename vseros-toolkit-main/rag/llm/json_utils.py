from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .base import LLMClient, JsonValue
from .config import LLMConfig


def build_json_prompt(user_prompt: str, schema_hint: Optional[str] = None) -> str:
    """
    Собирает промпт, который просит модель вернуть строго JSON.

    Здесь мы намеренно:
      - просим не добавлять ничего до/после JSON;
      - при наличии schema_hint даём модели пример/описание структуры;
      - в конце явно формулируем задание.
    """
    parts: List[str] = [
        "Ты — помощник, который ВСЕГДА отвечает строго в формате JSON.",
        "Не добавляй никакого текста до или после JSON.",
        "Не используй комментарии, пояснения, markdown-разметку или тройные кавычки.",
    ]

    if schema_hint:
        parts.append("")
        parts.append("Вот описание желаемой схемы JSON (пример структуры или подсказка полей):")
        parts.append(schema_hint.strip())

    parts.append("")
    parts.append("Теперь задание (сформируй ответ строго в формате JSON):")
    parts.append(user_prompt.strip())

    return "\n".join(parts)


def _strip_code_fence(text: str) -> str:
    """
    Убирает обёртку ```json ... ``` / ``` ... ``` вокруг ответа, если модель так ответила.

    Это частый паттерн: модель игнорирует инструкцию
    «не добавляй ничего кроме JSON» и всё равно возвращает markdown-кодовый блок.
    """
    t = text.strip()

    # Варианты начала: ```json, ```JSON, ``` (без указания языка)
    if t.startswith("```"):
        # Уберём первую строку с ```
        lines = t.splitlines()
        if not lines:
            return t

        # Отбрасываем первую строку (``` или ```json)
        lines = lines[1:]

        # Если последний элемент — тоже ```, убираем его
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]

        t = "\n".join(lines).strip()

    return t


def _extract_json_substring(text: str) -> str:
    """
    Пытается вытащить "ядро" JSON из ответа:
      - ищет первое вхождение '{' или '[',
      - затем пытается найти "соответствующую" закрывающую скобку,
      - если аккуратный перебор не получается, обрезает по последней '}' или ']'.

    Это не идеальный JSON-парсер, но для LLM-ответов обычно достаточно.
    """
    s = text.strip()

    # Ищем первое '{' или '['
    first_obj = s.find("{")
    first_arr = s.find("[")
    candidates = [i for i in (first_obj, first_arr) if i != -1]

    if not candidates:
        # Вообще не нашли JSON-подобных скобок — просто вернём оригинал,
        # пусть json.loads уже упадёт ясной ошибкой.
        return s

    start = min(candidates)
    open_char = s[start]
    close_char = "}" if open_char == "{" else "]"

    # Попробуем пройтись по строке и отбалансировать скобки
    depth = 0
    end = -1
    for i in range(start, len(s)):
        ch = s[i]
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                end = i
                break

    if end != -1:
        return s[start : end + 1]

    # Фоллбек: если аккуратно не получилось, обрежем по последней закрывающей
    last_brace = s.rfind("}")
    last_bracket = s.rfind("]")
    last = max(last_brace, last_bracket)
    if last != -1 and last > start:
        return s[start : last + 1]

    # Совсем крайний случай
    return s[start:]


def try_parse_json(text: str) -> JsonValue:
    """
    Пытается распарсить JSON из текста модели.

    Стратегия:
      1. Убрать возможный markdown-кодовый блок (```json ... ```).
      2. Попробовать json.loads на всём тексте.
      3. Если не получилось — попытаться вытащить подстроку с JSON и снова json.loads.
      4. Если и это не удалось — пробрасываем исходную ошибку дальше.
    """
    # 1. Убираем обёртку ```json ... ```
    cleaned = _strip_code_fence(text)

    # 2. Проба №1 — целиком
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 3. Проба №2 — вытащить "ядро" JSON
    candidate = _extract_json_substring(cleaned)
    try:
        return json.loads(candidate)
    except Exception as e:
        # Здесь можно добавить логирование, если нужно.
        # Пока просто выбрасываем понятную ошибку.
        raise ValueError(
            f"Не удалось распарсить JSON из текста модели. "
            f"Обрезанный фрагмент:\n{candidate[:500]}"
        ) from e


def generate_and_parse_json(
    client: LLMClient,
    user_prompt: str,
    schema_hint: Optional[str],
    config: LLMConfig,
) -> JsonValue:
    """
    Высокоуровневая функция: строит JSON-ориентированный промпт,
    вызывает client.generate и парсит результат в Python-структуру.

    Логика:
      - на основе user_prompt и schema_hint собираем промпт через build_json_prompt;
      - при необходимости усиливаем "строгость" (temperature=0, do_sample=False, json_mode=True);
      - вызываем client.generate(...);
      - парсим ответ как JSON через try_parse_json.
    """
    prompt = build_json_prompt(user_prompt, schema_hint)

    # Делаем более "строгий" конфиг для JSON-ответа.
    # Если json_mode уже включён — оставим, но всё равно имеет смысл
    # выключить случайный семплинг.
    effective_cfg = config.with_overrides(
        json_mode=True,
        temperature=0.0,
        do_sample=False,
        top_p=1.0,
        top_k=0,
    )

    raw_text = client.generate(prompt, effective_cfg)
    return try_parse_json(raw_text)


__all__ = [
    "build_json_prompt",
    "try_parse_json",
    "generate_and_parse_json",
]
