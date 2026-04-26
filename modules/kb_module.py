import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from openai import OpenAI

from pipeline.config_loader import render_prompt

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^```\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def _chat_with_file(client: OpenAI, model: str, file_path: Path, prompt: str) -> dict[str, Any]:
    file_obj = client.files.create(file=file_path, purpose="file-extract")
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"fileid://{file_obj.id}"},
            {"role": "user", "content": prompt},
        ],
    )
    content = completion.choices[0].message.content or "{}"
    return _extract_json(content)


def extract_resume_info(
    client: OpenAI,
    model: str,
    resume_path: str,
    prompt_template: str,
) -> dict[str, Any]:
    path = Path(resume_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"简历文件不存在: {path}")
    result = _chat_with_file(client, model, path, prompt_template)
    if "projects" not in result:
        logger.warning("简历抽取结果缺少 projects 字段。")
    return result


def extract_reference_docs(
    client: OpenAI,
    model: str,
    docs_dir: str,
    resume_info: dict[str, Any],
    prompt_template: str,
) -> dict[str, list[str]]:
    docs_path = Path(docs_dir).expanduser().resolve()
    if not docs_path.exists():
        raise FileNotFoundError(f"知识库目录不存在: {docs_path}")
    files = [p for p in docs_path.rglob("*") if p.is_file()]
    if not files:
        return {}
    project_names = [item.get("project_name", "") for item in resume_info.get("projects", []) if isinstance(item, dict)]
    grouped: dict[str, list[str]] = defaultdict(list)
    for file_path in files:
        prompt = render_prompt(prompt_template, project_names=project_names)
        parsed = _chat_with_file(client, model, file_path, prompt)
        project_name = parsed.get("project_name", "未分类项目")
        summary = parsed.get("summary", "")
        if summary:
            grouped[project_name].append(summary)
    return dict(grouped)
