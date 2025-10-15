import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any


@dataclass
class Record:
    text: str
    meta: Dict[str, Any]


class MetaStore:
    """
    Stores/loads records in a jsonl file with the SAME order as vectors in FAISS.
    The nth line corresponds to the nth vector.
    """

    def __init__(self, jsonl_path: Path):
        self.path = jsonl_path
        self._records: List[Record] = []

    def __len__(self) -> int:
        return len(self._records)

    def load(self) -> None:
        self._records.clear()
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    self._records.append(Record(text=obj["text"], meta=obj.get("meta", {})))

    def append_many(self, records: Iterable[Record]) -> None:
        self._records.extend(records)
        # append to disk to avoid rewriting the whole file
        with self.path.open("a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps({"text": r.text, "meta": r.meta}, ensure_ascii=False) + "\n")

    def get(self, idx: int) -> Record:
        return self._records[idx]


