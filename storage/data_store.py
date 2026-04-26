import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


class DataStore:
    def __init__(self, runs_dir: str, cache_index_path: str) -> None:
        self.runs_dir = Path(runs_dir)
        self.cache_index_path = Path(cache_index_path)
        self.stage_cache_path = self.cache_index_path.parent / "stage_cache_index.json"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.cache_index_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.cache_index_path.exists():
            self._write_json(self.cache_index_path, {})
        if not self.stage_cache_path.exists():
            self._write_json(self.stage_cache_path, {})

    def create_run(self) -> str:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        (self.runs_dir / run_id).mkdir(parents=True, exist_ok=True)
        return run_id

    def run_dir(self, run_id: str) -> Path:
        path = self.runs_dir / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_json(self, run_id: str, filename: str, payload: Any) -> Path:
        target = self.run_dir(run_id) / filename
        self._write_json(target, payload)
        return target

    def save_text(self, run_id: str, filename: str, content: str) -> Path:
        target = self.run_dir(run_id) / filename
        target.write_text(content, encoding="utf-8")
        return target

    def load_json(self, run_id: str, filename: str) -> Any:
        target = self.run_dir(run_id) / filename
        return json.loads(target.read_text(encoding="utf-8"))

    def load_text(self, run_id: str, filename: str) -> str:
        target = self.run_dir(run_id) / filename
        return target.read_text(encoding="utf-8")

    def append_run_index(self, row: dict[str, Any]) -> None:
        index_path = self.runs_dir / "run_index.jsonl"
        with index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def audio_fingerprint(self, audio_path: str) -> str:
        path = Path(audio_path)
        sha256 = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                sha256.update(chunk)
        stat = path.stat()
        return f"{sha256.hexdigest()}:{stat.st_size}"

    def file_fingerprint(self, file_path: str) -> str:
        path = Path(file_path)
        sha256 = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                sha256.update(chunk)
        stat = path.stat()
        return f"{sha256.hexdigest()}:{stat.st_size}"

    def dir_fingerprint(self, dir_path: str) -> str:
        path = Path(dir_path)
        if not path.exists():
            return "missing"
        files = sorted([p for p in path.rglob("*") if p.is_file()], key=lambda p: str(p))
        sha256 = hashlib.sha256()
        for file in files:
            rel = str(file.relative_to(path))
            sha256.update(rel.encode("utf-8"))
            with file.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    sha256.update(chunk)
        return sha256.hexdigest()

    def payload_fingerprint(self, payload: Any) -> str:
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def settings_fingerprint(self, settings_subset: dict[str, Any], prompts: dict[str, Any]) -> str:
        text = json.dumps({"settings": settings_subset, "prompts": prompts}, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_cached_run(self, cache_key: str) -> str | None:
        index = self._read_json(self.cache_index_path, {})
        item = index.get(cache_key)
        return item.get("run_id") if isinstance(item, dict) else None

    def set_cached_run(self, cache_key: str, run_id: str) -> None:
        index = self._read_json(self.cache_index_path, {})
        index[cache_key] = {"run_id": run_id, "updated_at": datetime.now().isoformat()}
        self._write_json(self.cache_index_path, index)

    def clear_cache(self) -> None:
        self._write_json(self.cache_index_path, {})
        self._write_json(self.stage_cache_path, {})

    def get_stage_cached_run(self, stage: str, cache_key: str) -> str | None:
        index = self._read_json(self.stage_cache_path, {})
        stage_map = index.get(stage, {})
        item = stage_map.get(cache_key) if isinstance(stage_map, dict) else None
        return item.get("run_id") if isinstance(item, dict) else None

    def set_stage_cached_run(self, stage: str, cache_key: str, run_id: str) -> None:
        index = self._read_json(self.stage_cache_path, {})
        if stage not in index or not isinstance(index.get(stage), dict):
            index[stage] = {}
        index[stage][cache_key] = {"run_id": run_id, "updated_at": datetime.now().isoformat()}
        self._write_json(self.stage_cache_path, index)

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _read_json(path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
