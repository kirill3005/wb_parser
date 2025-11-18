from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Callable, Any

from .corpus import Document, LocalChunk

class BaseChunker(Protocol):
    """
    Базовый интерфейс чанкера:
    получает Document, возвращает список LocalChunk.
    """

    def __call__(self, doc: Document) -> List[LocalChunk]:
        ...


# На всякий случай алиас, если где-то удобнее использовать type-хинт:
Chunker = Callable[[Document], List[LocalChunk]]


@dataclass
class CharChunker:
    """
    Чанкер по символам с overlap.

    Логика:
      [0 : chunk_size]
      [chunk_size - overlap : 2*chunk_size - overlap]
      ...

    Параметры
    ---------
    chunk_size_chars:
        Максимальная длина чанка в символах.
    chunk_overlap_chars:
        Перекрытие между чанками в символах.
    min_chunk_chars:
        Минимальная длина чанка, чтобы добавить его.
    """

    chunk_size_chars: int = 800
    chunk_overlap_chars: int = 100
    min_chunk_chars: int = 1

    def __call__(self, doc: Document) -> List[LocalChunk]:
        text = doc.text or ""
        n = len(text)
        if n == 0:
            return []

        if self.chunk_size_chars <= 0:
            raise ValueError("CharChunker: chunk_size_chars должен быть > 0")
        if self.chunk_overlap_chars < 0:
            raise ValueError("CharChunker: chunk_overlap_chars должен быть >= 0")
        if self.min_chunk_chars <= 0:
            raise ValueError("CharChunker: min_chunk_chars должен быть > 0")

        chunks: List[LocalChunk] = []
        pos = 0

        while pos < n:
            end = min(n, pos + self.chunk_size_chars)
            chunk_text = text[pos:end]

            if len(chunk_text) < self.min_chunk_chars:
                # слишком короткий хвост — игнорируем
                break

            chunks.append(
                LocalChunk(
                    text=chunk_text,
                    start_char=pos,
                    end_char=end,
                    meta=None,
                )
            )

            if end >= n:
                break

            # следующий старт с overlap
            pos = end - self.chunk_overlap_chars
            if pos < 0:
                pos = 0

        return chunks


@dataclass
class ParagraphChunker:
    """
    Чанкер по абзацам.

    По умолчанию:
      - делит текст по двум переводам строки ("\n\n"),
      - пропускает пустые или слишком короткие блоки.

    min_paragraph_chars:
        Минимальная длина абзаца, чтобы добавить его в чанки.
    """

    min_paragraph_chars: int = 1

    def __call__(self, doc: Document) -> List[LocalChunk]:
        text = doc.text or ""
        if not text:
            return []

        if self.min_paragraph_chars <= 0:
            raise ValueError("ParagraphChunker: min_paragraph_chars должен быть > 0")

        chunks: List[LocalChunk] = []
        offset = 0

        # Простейшее разбиение по пустой строке как разделителю абзацев.
        # Можно потом заменить на более умную логику под конкретный формат.
        parts = text.split("\n\n")

        for block in parts:
            raw_block = block
            block_stripped = raw_block.strip()
            if not block_stripped:
                # пустой абзац: примерно сдвигаем offset
                offset += len(raw_block) + 2  # учитываем "\n\n"
                continue

            if len(block_stripped) < self.min_paragraph_chars:
                offset += len(raw_block) + 2
                continue

            # Ищем block_stripped в исходном тексте, начиная с offset.
            # Это не идеально, но для большинства случаев ок.
            start = text.find(block_stripped, offset)
            if start == -1:
                # если не нашли — грубо сдвинемся дальше и пропустим блок
                offset += len(raw_block) + 2
                continue

            end = start + len(block_stripped)

            chunks.append(
                LocalChunk(
                    text=block_stripped,
                    start_char=start,
                    end_char=end,
                    meta=None,
                )
            )

            offset = end

        return chunks

@dataclass
class TokenChunker:
    """
    Чанкер по токенам под HF-токенайзер.

    Ожидается токенайзер с интерфейсом типа:
        tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )

    Логика:
      - считаем токены и offsets (char-диапазоны для каждого токена),
      - идём окнами по токенам: [0 : max_tokens],
        [max_tokens - overlap_tokens : 2*max_tokens - overlap_tokens], ...
      - для каждого окна берём char-диапазон по offsets и делаем LocalChunk.
    """

    tokenizer: Any
    max_tokens: int = 256
    overlap_tokens: int = 32
    min_chunk_tokens: int = 1

    def __call__(self, doc: Document) -> List[LocalChunk]:
        text = doc.text or ""
        if not text:
            return []

        if self.max_tokens <= 0:
            raise ValueError("TokenChunker: max_tokens должен быть > 0")
        if self.overlap_tokens < 0:
            raise ValueError("TokenChunker: overlap_tokens должен быть >= 0")
        if self.min_chunk_tokens <= 0:
            raise ValueError("TokenChunker: min_chunk_tokens должен быть > 0")

        # Кодируем текст в токены + получаем карту смещений
        encoding = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )

        input_ids = encoding["input_ids"]
        offsets = encoding["offset_mapping"]
        n_tokens = len(input_ids)

        if n_tokens == 0:
            return []

        if len(offsets) != n_tokens:
            raise ValueError(
                f"TokenChunker: len(offset_mapping)={len(offsets)} "
                f"не совпадает с len(input_ids)={n_tokens}"
            )

        chunks: List[LocalChunk] = []
        pos = 0

        while pos < n_tokens:
            end = min(n_tokens, pos + self.max_tokens)
            window_len = end - pos

            if window_len < self.min_chunk_tokens:
                break

            # Берём char-диапазон по offsets
            start_char = offsets[pos][0]
            end_char = offsets[end - 1][1] if end > pos else offsets[pos][1]

            # На всякий случай немного проверим
            if start_char is None or end_char is None:
                # если токенайзер вернул "дырявые" offsets — пропускаем такой блок
                pos = end
                continue

            if start_char >= end_char:
                pos = end
                continue

            chunk_text = text[start_char:end_char]

            chunks.append(
                LocalChunk(
                    text=chunk_text,
                    start_char=start_char,
                    end_char=end_char,
                    meta={
                        "token_start": pos,
                        "token_end": end,
                    },
                )
            )

            if end >= n_tokens:
                break

            # следующий старт c overlap
            pos = end - self.overlap_tokens
            if pos < 0:
                pos = 0

        return chunks

# Готовые дефолтные инстансы, чтобы было удобно писать:
#   add_chunks_to_corpus(corpus, chunker=chunkers.char_chunker)
#   add_chunks_to_corpus(corpus, chunker=chunkers.paragraph_chunker)

char_chunker = CharChunker()
paragraph_chunker = ParagraphChunker()