import csv
from pathlib import Path
from typing import Dict, List

class CSVService:
    """Reads CSV rows and extracts fields via a key map."""
    def __init__(self, csv_path: Path, keys_map: Dict[str, List[str]]):
        self.csv_path = csv_path
        self.keys_map = keys_map

    def read_rows(self):
        with open(self.csv_path, "r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def _first_non_empty(row: dict, keys: List[str]) -> str:
        for k in keys:
            v = (row.get(k) or "").strip()
            if v:
                return v
        return ""

    def extract_fields(self, row: dict):
        return {
            "image": self._first_non_empty(row, self.keys_map["image"]),
            "title": self._first_non_empty(row, self.keys_map["title"]),
            "date": self._first_non_empty(row, self.keys_map["date"]),
            "source": self._first_non_empty(row, self.keys_map["source"]),
            "category": self._first_non_empty(row, self.keys_map["category"]),
        }
