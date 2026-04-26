import json
import re
from difflib import get_close_matches
from typing import Any

from openai import OpenAI

from pipeline.config_loader import render_prompt


def parse_transcript_to_rounds(transcript_text: str) -> list[dict[str, str]]:
    pattern = re.compile(r"^\[(面试官|求职者)\]:\s*(.*)$")
    utterances: list[dict[str, str]] = []
    for line in transcript_text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = pattern.match(line)
        if match:
            utterances.append({"role": match.group(1), "text": match.group(2)})

    rounds: list[dict[str, str]] = []
    idx = 0
    while idx < len(utterances):
        if utterances[idx]["role"] != "面试官":
            idx += 1
            continue
        interviewer = utterances[idx]["text"]
        idx += 1
        answers: list[str] = []
        while idx < len(utterances) and utterances[idx]["role"] == "求职者":
            answers.append(utterances[idx]["text"])
            idx += 1
        if answers:
            rounds.append({"interviewer": interviewer, "candidate": " ".join(answers).strip()})
    return rounds


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^```\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def _format_history(rounds: list[dict[str, Any]]) -> str:
    if not rounds:
        return "无"
    lines = []
    for item in rounds:
        lines.append(f"[面试官]: {item['interviewer']}")
        lines.append(f"[求职者]: {item['candidate']}")
    return "\n".join(lines)


def _match_project_docs(project_name: Any, docs_map: dict[str, list[str]]) -> list[str]:
    if not isinstance(project_name, str) or not project_name.strip():
        return []
    name = project_name.strip()
    if name in docs_map:
        return docs_map[name]
    for key in docs_map:
        if name in key or key in name:
            return docs_map[key]
    best = get_close_matches(name, list(docs_map.keys()), n=1, cutoff=0.35)
    return docs_map.get(best[0], []) if best else []


def _chat_json(client: OpenAI, model: str, prompt: str) -> dict[str, Any]:
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    content = completion.choices[0].message.content or "{}"
    return _extract_json(content)


def extract_qa(
    client: OpenAI,
    model: str,
    rounds: list[dict[str, str]],
    prompt_template: str,
    history_rounds: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for idx, current in enumerate(rounds):
        history = rounds[max(0, idx - history_rounds):idx]
        prompt = render_prompt(
            prompt_template,
            history_qa=_format_history(history),
            current_qa=f"[面试官]: {current['interviewer']}\n[求职者]: {current['candidate']}",
        )
        parsed = _chat_json(client, model, prompt)
        result.append(
            {
                "round_id": idx + 1,
                "interviewer": current["interviewer"],
                "candidate": current["candidate"],
                "project_name": parsed.get("project_name"),
                "tech_points": parsed.get("tech_points", []),
                "confidence": parsed.get("confidence"),
            }
        )
    return result


def optimize_qa(
    client: OpenAI,
    model: str,
    qa_items: list[dict[str, Any]],
    docs_map: dict[str, list[str]],
    resume_info: dict[str, Any],
    optimize_prompt_template: str,
    followup_prompt_template: str,
    history_rounds: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    resume_text = json.dumps(resume_info, ensure_ascii=False, indent=2)
    for idx, item in enumerate(qa_items):
        history = qa_items[max(0, idx - history_rounds):idx]
        docs = _match_project_docs(item.get("project_name"), docs_map)
        optimize_prompt = render_prompt(
            optimize_prompt_template,
            history_qa=_format_history(history),
            current_qa=f"[面试官]: {item['interviewer']}\n[求职者]: {item['candidate']}",
            project_docs="\n".join(docs) if docs else "无相关项目补充资料",
            resume_info=resume_text,
        )
        optimized = _chat_json(client, model, optimize_prompt)
        followup_prompt = render_prompt(
            followup_prompt_template,
            question=item["interviewer"],
            answer=optimized.get("optimized_answer", item["candidate"]),
            tech_points=item.get("tech_points", []),
        )
        followup = _chat_json(client, model, followup_prompt)
        output.append(
            {
                "round_id": item["round_id"],
                "project_name": item.get("project_name"),
                "tech_points": item.get("tech_points", []),
                "interviewer": item["interviewer"],
                "original_answer": item["candidate"],
                "optimized_answer": optimized.get("optimized_answer", ""),
                "optimization_points": optimized.get("optimization_points", []),
                "followups": followup.get("followups", []),
            }
        )
    return output
