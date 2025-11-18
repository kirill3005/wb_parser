def count_tokens_simple(text: str) -> int:
    """
    Очень простой подсчёт "токенов": количество слов по split().
    На практике лучше передать свою функцию count_tokens с использованием
    реального токенайзера (например, tiktoken/HF).
    """
    return len(text.split())