import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import dashscope
from openai import OpenAI

from pipeline.config_loader import render_prompt


logger = logging.getLogger(__name__)


def normalize_audio(local_audio_path: str) -> str:
    src = Path(local_audio_path).expanduser().resolve()
    if src.suffix.lower() not in {".m4a", ".aac", ".amr", ".caf"}:
        return str(src)
    with tempfile.NamedTemporaryFile(prefix="asr_audio_", suffix=".wav", delete=False) as tmp:
        dst = Path(tmp.name)
    cmd = ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return str(dst)


def split_audio_with_ffmpeg(audio_path: Path, segment_seconds: int) -> list[Path]:
    tmp_dir = Path(tempfile.mkdtemp(prefix="asr_segments_"))
    out_pattern = str(tmp_dir / "part_%03d.wav")
    cmd = [
        "ffmpeg", "-y", "-i", str(audio_path), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        "-f", "segment", "-segment_time", str(segment_seconds), out_pattern,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    parts = sorted(tmp_dir.glob("part_*.wav"))
    if not parts:
        raise RuntimeError("音频切分失败，未生成分段文件")
    return parts


def _extract_text_from_response(response) -> str:
    output = getattr(response, "output", None)
    if output is None and isinstance(response, dict):
        output = response.get("output")
    if output is None:
        raise RuntimeError("ASR 响应缺少 output")
    text = getattr(output, "text", None) or (output.get("text") if hasattr(output, "get") else None)
    if text:
        return str(text).strip()
    choices = getattr(output, "choices", None) or (output.get("choices") if hasattr(output, "get") else None)
    if not choices:
        raise RuntimeError("ASR 响应缺少文本内容")
    message = choices[0].message if hasattr(choices[0], "message") else choices[0].get("message")
    content = getattr(message, "content", None) if hasattr(message, "content") else message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content).strip()
    raise RuntimeError("无法解析 ASR 响应文本")


def _asr_call(api_key: str, model: str, audio_file: Path):
    return dashscope.MultiModalConversation.call(
        model=model,
        messages=[{"role": "user", "content": [{"audio": audio_file.as_uri()}]}],
        api_key=api_key,
        result_format="message",
        asr_options={"enable_itn": False},
    )


def _is_limit_error(code: str, message: str) -> bool:
    text = f"{code} {message}".lower()
    return "file size is too large" in text or "audio is too long" in text


def asr_transcribe_full(api_key: str, audio_path: str, model: str, segment_seconds: int, min_segment_seconds: int) -> str:
    path = Path(audio_path).expanduser().resolve()
    rsp = _asr_call(api_key, model, path)
    if rsp.status_code == 200:
        return _extract_text_from_response(rsp)
    if not _is_limit_error(rsp.code, rsp.message):
        raise RuntimeError(f"ASR 失败: {rsp.code} {rsp.message}")
    parts = split_audio_with_ffmpeg(path, segment_seconds)
    transcripts: list[str] = []
    for part in parts:
        part_rsp = _asr_call(api_key, model, part)
        if part_rsp.status_code == 200:
            transcripts.append(_extract_text_from_response(part_rsp))
            continue
        if segment_seconds <= min_segment_seconds:
            raise RuntimeError(f"分段后 ASR 仍失败: {part_rsp.code} {part_rsp.message}")
        sub_parts = split_audio_with_ffmpeg(part, max(min_segment_seconds, segment_seconds // 2))
        for sub in sub_parts:
            sub_rsp = _asr_call(api_key, model, sub)
            if sub_rsp.status_code != 200:
                raise RuntimeError(f"细分 ASR 失败: {sub_rsp.code} {sub_rsp.message}")
            transcripts.append(_extract_text_from_response(sub_rsp))
    return "\n".join(transcripts).strip()


def diarize_transcript(
    api_key: str,
    base_url: str,
    model: str,
    max_tokens: int,
    transcript: str,
    prompt_template: str,
) -> str:
    prompt = render_prompt(prompt_template, transcript=transcript)
    client = OpenAI(api_key=api_key, base_url=base_url)
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.1,
    )
    result = completion.choices[0].message.content or ""
    if not result.strip():
        raise RuntimeError("角色识别失败: 模型返回为空")
    pattern = re.compile(r"^\[(面试官|求职者)\]:", re.MULTILINE)
    if not pattern.search(result):
        logger.warning("角色识别结果格式可能异常，未检测到标准角色前缀。")
    return result
