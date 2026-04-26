import json
import logging
from pathlib import Path
from typing import Any

import gradio as gr

from logging_config import setup_logging
from pipeline.config_loader import load_json, validate_prompts
from pipeline.orchestrator import PipelineOrchestrator

SETTINGS_PATH = Path("config/settings.json")


def _load_settings() -> dict[str, Any]:
    return load_json(SETTINGS_PATH)


def _save_settings(payload: dict[str, Any]) -> None:
    SETTINGS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_orchestrator() -> PipelineOrchestrator:
    settings = _load_settings()
    setup_logging(
        level=settings["logging"]["level"],
        app_log_path=settings["logging"]["app_log_path"],
        error_log_path=settings["logging"]["error_log_path"],
    )
    return PipelineOrchestrator(str(SETTINGS_PATH))


orchestrator = _build_orchestrator()
logger = logging.getLogger(__name__)


def run_analysis(audio_file, resume_file, docs_dir: str):
    if not audio_file:
        raise gr.Error("请先上传录音文件")
    if not resume_file:
        raise gr.Error("请先上传简历文件")
    if not docs_dir or not Path(docs_dir).exists():
        raise gr.Error("请填写有效的知识库目录路径")

    result = orchestrator.run(
        audio_path=audio_file.name,
        resume_path=resume_file.name,
        docs_dir=docs_dir,
    )
    ui_model = result["ui_view_model"]
    qa_items = ui_model.get("qa_items", [])
    qa_choices = [str(item["round_id"]) for item in qa_items]
    first = qa_items[0] if qa_items else {}

    status = f"运行完成，run_id: `{result['run_id']}`，缓存命中: `{result['cache_hit']}`"
    kb_files = [[str(p.name)] for p in Path(docs_dir).rglob("*") if p.is_file()]
    process_md = (
        "1. 上传音频 -> 2. 语音识别 -> 3. 说话人分离 -> 4. 问答抽取 -> "
        "5. 知识检索 -> 6. 回答优化 -> 7. 结果展示"
    )
    return (
        status,
        ui_model.get("audio_file", ""),
        ui_model.get("transcript", ""),
        kb_files,
        process_md,
        gr.update(choices=qa_choices, value=qa_choices[0] if qa_choices else None),
        first.get("interviewer", ""),
        first.get("original_answer", ""),
        first.get("optimized_answer", ""),
        "、".join(first.get("tech_points", [])),
        "\n".join(f"- {x}" for x in first.get("followups", [])),
        ui_model,
    )


def on_round_change(selected_round: str, ui_model: dict[str, Any]):
    if not ui_model or not selected_round:
        return "", "", "", "", ""
    qa_items = ui_model.get("qa_items", [])
    target = next((item for item in qa_items if str(item.get("round_id")) == str(selected_round)), None)
    if not target:
        return "", "", "", "", ""
    return (
        target.get("interviewer", ""),
        target.get("original_answer", ""),
        target.get("optimized_answer", ""),
        "、".join(target.get("tech_points", [])),
        "\n".join(f"- {x}" for x in target.get("followups", [])),
    )


def load_logs():
    return orchestrator.get_recent_logs(120)


def clear_cache():
    return orchestrator.clear_cache()


def save_settings(
    api_key: str,
    base_url: str,
    asr_model: str,
    llm_model: str,
    kb_model: str,
    max_tokens: int,
    segment_seconds: int,
    min_segment_seconds: int,
    history_rounds: int,
    log_level: str,
    prompt_config_path: str,
):
    settings = _load_settings()
    settings["api"]["dashscope_api_key"] = api_key.strip()
    settings["api"]["dashscope_base_url"] = base_url.strip()
    settings["models"]["asr_model"] = asr_model.strip()
    settings["models"]["llm_model"] = llm_model.strip()
    settings["models"]["kb_model"] = kb_model.strip()
    settings["models"]["max_tokens"] = int(max_tokens)
    settings["pipeline"]["asr_segment_seconds"] = int(segment_seconds)
    settings["pipeline"]["asr_min_segment_seconds"] = int(min_segment_seconds)
    settings["pipeline"]["history_rounds"] = int(history_rounds)
    settings["logging"]["level"] = log_level
    settings["prompt_config_path"] = prompt_config_path.strip()

    prompts = load_json(settings["prompt_config_path"])
    validate_prompts(prompts)
    _save_settings(settings)

    global orchestrator
    orchestrator = _build_orchestrator()
    return "设置已保存并生效。"


def reload_prompts():
    orchestrator.reload_configs()
    return "Prompt 配置已重新加载。"


def build_ui():
    settings = _load_settings()
    with gr.Blocks(title="Interview Review AI") as demo:
        ui_state = gr.State({})
        gr.Markdown("## Interview Review AI")
        with gr.Tabs():
            with gr.Tab("面试复盘"):
                with gr.Row():
                    with gr.Column(scale=2):
                        audio_file = gr.File(label="上传录音", file_types=["audio"])
                        resume_file = gr.File(label="上传简历", file_types=[".pdf", ".doc", ".docx", ".txt"])
                        docs_dir = gr.Textbox(label="知识库目录路径", placeholder="例如: /Users/xxx/knowledge_docs")
                        run_btn = gr.Button("开始复盘", variant="primary")
                        status = gr.Markdown("等待执行")
                        process_view = gr.Markdown("流程展示区")
                    with gr.Column(scale=3):
                        audio_name = gr.Textbox(label="音频文件")
                        transcript = gr.Textbox(label="转写结果", lines=8)
                        kb_files = gr.Dataframe(label="关联知识库文件", headers=["文件名"], datatype=["str"])

                with gr.Row():
                    round_selector = gr.Dropdown(label="问答轮次", choices=[])
                    refresh_log_btn = gr.Button("刷新日志")
                    clear_cache_btn = gr.Button("清理缓存")

                with gr.Row():
                    question = gr.Textbox(label="面试官问题", lines=2)
                    tech_points = gr.Textbox(label="涉及技术点")
                with gr.Row():
                    original = gr.Textbox(label="原始回答", lines=6)
                    optimized = gr.Textbox(label="优化后回答", lines=6)
                followups = gr.Textbox(label="追问建议", lines=4)
                logs_view = gr.Textbox(label="执行日志", lines=10)

            with gr.Tab("知识库"):
                gr.Markdown("### 知识库区域\n展示已上传资料并用于问答优化。")
                gr.Markdown("建议将项目资料统一放在一个目录，并在面试复盘页填写该目录路径。")

            with gr.Tab("设置"):
                api_key = gr.Textbox(label="DashScope API Key", value=settings["api"]["dashscope_api_key"], type="password")
                base_url = gr.Textbox(label="API Base URL", value=settings["api"]["dashscope_base_url"])
                asr_model = gr.Textbox(label="ASR 模型", value=settings["models"]["asr_model"])
                llm_model = gr.Textbox(label="LLM 模型", value=settings["models"]["llm_model"])
                kb_model = gr.Textbox(label="知识库模型", value=settings["models"]["kb_model"])
                max_tokens = gr.Number(label="max_tokens", value=settings["models"]["max_tokens"], precision=0)
                segment_seconds = gr.Number(label="ASR 分段秒数", value=settings["pipeline"]["asr_segment_seconds"], precision=0)
                min_segment_seconds = gr.Number(label="ASR 最小分段秒数", value=settings["pipeline"]["asr_min_segment_seconds"], precision=0)
                history_rounds = gr.Number(label="历史轮次窗口", value=settings["pipeline"]["history_rounds"], precision=0)
                log_level = gr.Dropdown(label="日志级别", choices=["DEBUG", "INFO", "WARNING", "ERROR"], value=settings["logging"]["level"])
                prompt_path = gr.Textbox(label="Prompt 配置路径", value=settings["prompt_config_path"])
                with gr.Row():
                    save_btn = gr.Button("保存设置", variant="primary")
                    reload_prompt_btn = gr.Button("重新加载 Prompt")
                settings_status = gr.Markdown("")

        run_btn.click(
            fn=run_analysis,
            inputs=[audio_file, resume_file, docs_dir],
            outputs=[
                status,
                audio_name,
                transcript,
                kb_files,
                process_view,
                round_selector,
                question,
                original,
                optimized,
                tech_points,
                followups,
                ui_state,
            ],
        )
        round_selector.change(
            fn=on_round_change,
            inputs=[round_selector, ui_state],
            outputs=[question, original, optimized, tech_points, followups],
        )
        refresh_log_btn.click(fn=load_logs, outputs=[logs_view])
        clear_cache_btn.click(fn=clear_cache, outputs=[status])

        save_btn.click(
            fn=save_settings,
            inputs=[
                api_key, base_url, asr_model, llm_model, kb_model, max_tokens,
                segment_seconds, min_segment_seconds, history_rounds, log_level, prompt_path,
            ],
            outputs=[settings_status],
        )
        reload_prompt_btn.click(fn=reload_prompts, outputs=[settings_status])
    return demo


if __name__ == "__main__":
    try:
        build_ui().launch()
    except Exception as exc:
        logger.exception("启动失败: %s", exc)
        raise
