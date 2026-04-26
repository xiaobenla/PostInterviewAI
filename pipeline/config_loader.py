import json
import re
from pathlib import Path
from typing import Any


REQUIRED_PROMPT_KEYS = [
    "asr.diarization_prompt",
    "kb.resume_extract_prompt",
    "kb.reference_summary_prompt",
    "qa.extraction_prompt",
    "qa.optimization_prompt",
    "qa.followup_prompt",
]


def load_json(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {file_path}")
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"配置文件顶层必须是对象: {file_path}")
    return payload


def validate_prompts(prompts: dict[str, Any]) -> None:
    for key in REQUIRED_PROMPT_KEYS:
        cur: Any = prompts
        for part in key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                raise ValueError(f"缺少 prompt 配置: {key}")
            cur = cur[part]
        if not isinstance(cur, str) or not cur.strip():
            raise ValueError(f"prompt 内容为空: {key}")


def render_prompt(template: str, **kwargs: Any) -> str:
    # 仅替换形如 {var_name} 的变量占位符，保留 JSON 示例中的其他大括号
    # 例如 {"project_name":""} 不会被误解析为 format 占位符
    pattern = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")

    def replacer(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in kwargs:
            return match.group(0)
        value = kwargs[key]
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    rendered = pattern.sub(replacer, template)

    # 兼容历史 prompt 占位符风格：{{var_name}} 与 #var_name#
    for key, value in kwargs.items():
        if isinstance(value, (dict, list)):
            text = json.dumps(value, ensure_ascii=False)
        else:
            text = str(value)
        rendered = rendered.replace(f"{{{{{key}}}}}", text)
        rendered = rendered.replace(f"#{key}#", text)
    return rendered
