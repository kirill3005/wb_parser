from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass
class Indexers:
    """Bidirectional id<->index mappings saved as json.

    The mappings are intentionally kept as plain dicts for portability.
    """

    user2idx: Dict[str, int] = field(default_factory=dict)
    item2idx: Dict[str, int] = field(default_factory=dict)
    session2idx: Dict[str, int] = field(default_factory=dict)

    def save(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "user2idx.json").write_text(json.dumps(self.user2idx, ensure_ascii=False), encoding="utf-8")
        (path / "item2idx.json").write_text(json.dumps(self.item2idx, ensure_ascii=False), encoding="utf-8")
        (path / "session2idx.json").write_text(json.dumps(self.session2idx, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, directory: str) -> "Indexers":
        path = Path(directory)
        def _maybe(file: str) -> Dict[str, int]:
            fp = path / file
            if fp.exists():
                return json.loads(fp.read_text(encoding="utf-8"))
            return {}

        return cls(
            user2idx=_maybe("user2idx.json"),
            item2idx=_maybe("item2idx.json"),
            session2idx=_maybe("session2idx.json"),
        )
