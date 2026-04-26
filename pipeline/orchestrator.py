import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

from modules.asr_module import asr_transcribe_full, diarize_transcript, normalize_audio
from modules.kb_module import extract_reference_docs, extract_resume_info
from modules.qa_module import extract_qa, optimize_qa, parse_transcript_to_rounds
from pipeline.config_loader import load_json, validate_prompts
from storage.data_store import DataStore

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    def __init__(self, settings_path: str = "config/settings.json") -> None:
        self.settings_path = Path(settings_path)
        self.settings = load_json(self.settings_path)
        self.prompts = load_json(self.settings.get("prompt_config_path", "config/prompts.json"))
        validate_prompts(self.prompts)
        self.data_store = DataStore(
            runs_dir=self.settings["storage"]["runs_dir"],
            cache_index_path=self.settings["storage"]["cache_index_path"],
        )
        self.client = OpenAI(
            api_key=self.settings["api"]["dashscope_api_key"],
            base_url=self.settings["api"]["dashscope_base_url"],
        )

    def reload_configs(self) -> None:
        self.settings = load_json(self.settings_path)
        self.prompts = load_json(self.settings.get("prompt_config_path", "config/prompts.json"))
        validate_prompts(self.prompts)

    def run(self, audio_path: str, resume_path: str, docs_dir: str) -> dict[str, Any]:
        if not self.settings["api"]["dashscope_api_key"]:
            raise RuntimeError("请先在设置中填写 dashscope_api_key")

        fingerprint = self.data_store.audio_fingerprint(audio_path)
        setting_fp = self.data_store.settings_fingerprint(
            {
                "models": self.settings["models"],
                "pipeline": self.settings["pipeline"],
            },
            self.prompts,
        )
        cache_key = f"{fingerprint}:{setting_fp}"
        cached_run = self.data_store.get_cached_run(cache_key)
        if cached_run:
            logger.info("cache_hit run_id=%s", cached_run)
            return {
                "run_id": cached_run,
                "cache_hit": True,
                "ui_view_model": self.data_store.load_json(cached_run, "ui_view_model.json"),
            }

        logger.info("cache_miss audio=%s", audio_path)
        run_id = self.data_store.create_run()
        trace_path = self.data_store.run_dir(run_id) / "trace.jsonl"

        def trace(stage: str, status: str, extra: dict[str, Any] | None = None) -> None:
            row = {
                "ts": datetime.now().isoformat(),
                "run_id": run_id,
                "stage": stage,
                "status": status,
            }
            if extra:
                row.update(extra)
            with trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        started = time.time()
        self.data_store.save_json(
            run_id,
            "input_manifest.json",
            {
                "audio_path": str(Path(audio_path).resolve()),
                "resume_path": str(Path(resume_path).resolve()),
                "docs_dir": str(Path(docs_dir).resolve()),
                "settings_snapshot": self.settings,
                "prompt_path": self.settings["prompt_config_path"],
            },
        )
        try:
            trace("asr", "start")
            asr_setting_fp = self.data_store.settings_fingerprint(
                {
                    "asr_model": self.settings["models"]["asr_model"],
                    "llm_model": self.settings["models"]["llm_model"],
                    "max_tokens": self.settings["models"]["max_tokens"],
                    "asr_segment_seconds": self.settings["pipeline"]["asr_segment_seconds"],
                    "asr_min_segment_seconds": self.settings["pipeline"]["asr_min_segment_seconds"],
                    "dashscope_base_url": self.settings["api"]["dashscope_base_url"],
                },
                {"asr": self.prompts["asr"]},
            )
            asr_cache_key = f"{fingerprint}:{asr_setting_fp}"
            cached_asr_run = self.data_store.get_stage_cached_run("asr", asr_cache_key)
            transcript_raw = ""
            transcript = ""

            if cached_asr_run:
                logger.info("asr_cache_hit run_id=%s", cached_asr_run)
                trace("asr", "cache_hit", {"cached_run_id": cached_asr_run})
                transcript = self.data_store.load_text(cached_asr_run, "asr_transcript.txt")
                try:
                    transcript_raw = self.data_store.load_text(cached_asr_run, "asr_transcript_raw.txt")
                except FileNotFoundError:
                    transcript_raw = transcript
            else:
                normalized_audio = normalize_audio(audio_path)
                transcript_raw = asr_transcribe_full(
                    api_key=self.settings["api"]["dashscope_api_key"],
                    audio_path=normalized_audio,
                    model=self.settings["models"]["asr_model"],
                    segment_seconds=self.settings["pipeline"]["asr_segment_seconds"],
                    min_segment_seconds=self.settings["pipeline"]["asr_min_segment_seconds"],
                )
                transcript = diarize_transcript(
                    api_key=self.settings["api"]["dashscope_api_key"],
                    base_url=self.settings["api"]["dashscope_base_url"],
                    model=self.settings["models"]["llm_model"],
                    max_tokens=self.settings["models"]["max_tokens"],
                    transcript=transcript_raw,
                    prompt_template=self.prompts["asr"]["diarization_prompt"],
                )
                self.data_store.set_stage_cached_run("asr", asr_cache_key, run_id)
                trace("asr", "cache_miss")

            self.data_store.save_text(run_id, "asr_transcript_raw.txt", transcript_raw)
            self.data_store.save_text(run_id, "asr_transcript.txt", transcript)
            trace("asr", "done", {"chars": len(transcript)})

            trace("kb_resume", "start")
            resume_file_fp = self.data_store.file_fingerprint(resume_path)
            kb_resume_setting_fp = self.data_store.settings_fingerprint(
                {
                    "kb_model": self.settings["models"]["kb_model"],
                    "dashscope_base_url": self.settings["api"]["dashscope_base_url"],
                },
                {"kb_resume": self.prompts["kb"]["resume_extract_prompt"]},
            )
            kb_resume_cache_key = f"{resume_file_fp}:{kb_resume_setting_fp}"
            cached_kb_resume_run = self.data_store.get_stage_cached_run("kb_resume", kb_resume_cache_key)
            if cached_kb_resume_run:
                logger.info("kb_resume_cache_hit run_id=%s", cached_kb_resume_run)
                trace("kb_resume", "cache_hit", {"cached_run_id": cached_kb_resume_run})
                resume_info = self.data_store.load_json(cached_kb_resume_run, "resume_info.json")
            else:
                resume_info = extract_resume_info(
                    client=self.client,
                    model=self.settings["models"]["kb_model"],
                    resume_path=resume_path,
                    prompt_template=self.prompts["kb"]["resume_extract_prompt"],
                )
                self.data_store.set_stage_cached_run("kb_resume", kb_resume_cache_key, run_id)
                trace("kb_resume", "cache_miss")
            self.data_store.save_json(run_id, "resume_info.json", resume_info)
            trace("kb_resume", "done")

            trace("kb_docs", "start")
            docs_dir_fp = self.data_store.dir_fingerprint(docs_dir)
            resume_payload_fp = self.data_store.payload_fingerprint(resume_info)
            kb_docs_setting_fp = self.data_store.settings_fingerprint(
                {
                    "kb_model": self.settings["models"]["kb_model"],
                    "dashscope_base_url": self.settings["api"]["dashscope_base_url"],
                },
                {"kb_docs": self.prompts["kb"]["reference_summary_prompt"]},
            )
            kb_docs_cache_key = f"{docs_dir_fp}:{resume_payload_fp}:{kb_docs_setting_fp}"
            cached_kb_docs_run = self.data_store.get_stage_cached_run("kb_docs", kb_docs_cache_key)
            if cached_kb_docs_run:
                logger.info("kb_docs_cache_hit run_id=%s", cached_kb_docs_run)
                trace("kb_docs", "cache_hit", {"cached_run_id": cached_kb_docs_run})
                docs_map = self.data_store.load_json(cached_kb_docs_run, "reference_docs.json")
            else:
                docs_map = extract_reference_docs(
                    client=self.client,
                    model=self.settings["models"]["kb_model"],
                    docs_dir=docs_dir,
                    resume_info=resume_info,
                    prompt_template=self.prompts["kb"]["reference_summary_prompt"],
                )
                self.data_store.set_stage_cached_run("kb_docs", kb_docs_cache_key, run_id)
                trace("kb_docs", "cache_miss")
            self.data_store.save_json(run_id, "reference_docs.json", docs_map)
            trace("kb_docs", "done", {"doc_groups": len(docs_map)})

            trace("qa_extract", "start")
            transcript_fp = self.data_store.payload_fingerprint({"transcript": transcript})
            qa_extract_setting_fp = self.data_store.settings_fingerprint(
                {
                    "llm_model": self.settings["models"]["llm_model"],
                    "history_rounds": self.settings["pipeline"]["history_rounds"],
                    "dashscope_base_url": self.settings["api"]["dashscope_base_url"],
                },
                {"qa_extract": self.prompts["qa"]["extraction_prompt"]},
            )
            qa_extract_cache_key = f"{transcript_fp}:{qa_extract_setting_fp}"
            cached_qa_extract_run = self.data_store.get_stage_cached_run("qa_extract", qa_extract_cache_key)
            if cached_qa_extract_run:
                logger.info("qa_extract_cache_hit run_id=%s", cached_qa_extract_run)
                trace("qa_extract", "cache_hit", {"cached_run_id": cached_qa_extract_run})
                qa_extracted = self.data_store.load_json(cached_qa_extract_run, "qa_extraction.json")
            else:
                rounds = parse_transcript_to_rounds(transcript)
                qa_extracted = extract_qa(
                    client=self.client,
                    model=self.settings["models"]["llm_model"],
                    rounds=rounds,
                    prompt_template=self.prompts["qa"]["extraction_prompt"],
                    history_rounds=self.settings["pipeline"]["history_rounds"],
                )
                self.data_store.set_stage_cached_run("qa_extract", qa_extract_cache_key, run_id)
                trace("qa_extract", "cache_miss")
            self.data_store.save_json(run_id, "qa_extraction.json", qa_extracted)
            trace("qa_extract", "done", {"rounds": len(qa_extracted)})

            trace("qa_optimize", "start")
            qa_optimize_input_fp = self.data_store.payload_fingerprint(
                {
                    "qa_extracted": qa_extracted,
                    "docs_map": docs_map,
                    "resume_info": resume_info,
                }
            )
            qa_optimize_setting_fp = self.data_store.settings_fingerprint(
                {
                    "llm_model": self.settings["models"]["llm_model"],
                    "history_rounds": self.settings["pipeline"]["history_rounds"],
                    "dashscope_base_url": self.settings["api"]["dashscope_base_url"],
                },
                {
                    "qa_optimize": self.prompts["qa"]["optimization_prompt"],
                    "qa_followup": self.prompts["qa"]["followup_prompt"],
                },
            )
            qa_optimize_cache_key = f"{qa_optimize_input_fp}:{qa_optimize_setting_fp}"
            cached_qa_optimize_run = self.data_store.get_stage_cached_run("qa_optimize", qa_optimize_cache_key)
            if cached_qa_optimize_run:
                logger.info("qa_optimize_cache_hit run_id=%s", cached_qa_optimize_run)
                trace("qa_optimize", "cache_hit", {"cached_run_id": cached_qa_optimize_run})
                qa_optimized = self.data_store.load_json(cached_qa_optimize_run, "qa_optimization.json")
            else:
                qa_optimized = optimize_qa(
                    client=self.client,
                    model=self.settings["models"]["llm_model"],
                    qa_items=qa_extracted,
                    docs_map=docs_map,
                    resume_info=resume_info,
                    optimize_prompt_template=self.prompts["qa"]["optimization_prompt"],
                    followup_prompt_template=self.prompts["qa"]["followup_prompt"],
                    history_rounds=self.settings["pipeline"]["history_rounds"],
                )
                self.data_store.set_stage_cached_run("qa_optimize", qa_optimize_cache_key, run_id)
                trace("qa_optimize", "cache_miss")
            self.data_store.save_json(run_id, "qa_optimization.json", qa_optimized)
            trace("qa_optimize", "done")

            ui_view_model = {
                "run_id": run_id,
                "audio_file": Path(audio_path).name,
                "transcript": transcript,
                "knowledge_files_count": len([p for p in Path(docs_dir).rglob("*") if p.is_file()]),
                "qa_items": qa_optimized,
            }
            self.data_store.save_json(run_id, "ui_view_model.json", ui_view_model)
            self.data_store.append_run_index(
                {
                    "run_id": run_id,
                    "audio_file": Path(audio_path).name,
                    "created_at": datetime.now().isoformat(),
                    "cache_key": cache_key,
                    "elapsed_seconds": round(time.time() - started, 2),
                }
            )
            self.data_store.set_cached_run(cache_key, run_id)
            return {"run_id": run_id, "cache_hit": False, "ui_view_model": ui_view_model}
        except Exception as exc:
            trace("pipeline", "error", {"error": str(exc)})
            logger.exception("pipeline failed run_id=%s", run_id)
            raise

    def get_recent_logs(self, limit: int = 80) -> str:
        app_log = Path(self.settings["logging"]["app_log_path"])
        if not app_log.exists():
            return "暂无日志"
        lines = app_log.read_text(encoding="utf-8").splitlines()
        return "\n".join(lines[-limit:])

    def clear_cache(self) -> str:
        self.data_store.clear_cache()
        return "缓存已清理"
