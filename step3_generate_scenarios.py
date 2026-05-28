from __future__ import annotations

import argparse
import json
from pathlib import Path


SYSTEM_PROMPT = """You are a creative writer specializing in culturally grounded stories.
Given a cultural group, context, and a culturally expected behavior,
write a short realistic story (3-5 sentences) where a character either
follows or violates this cultural norm.

Guidelines:
- Write in third-person perspective
- Make the story concrete and specific
- Include at least one other character who reacts
- Do NOT explicitly state whether the action is acceptable or not
- Do NOT use words like 'rude', 'inappropriate', or 'acceptable'
- Do NOT mention the cultural group name directly in the story"""


def make_yes_prompt(row: dict) -> str:
    return (
        f"Cultural group: {row.get('cultural_group', '')}\n"
        f"Context: {row.get('context', 'social interaction')}\n"
        f"Character: {row.get('actor', 'a person')}\n"
        f"Culturally expected behavior: {row['actor_behavior']}\n\n"
        "Write a short story (3-5 sentences) where the character "
        "FOLLOWS this culturally expected behavior in a realistic everyday situation. "
        "Include at least one other character who reacts positively or naturally."
    )


def make_no_prompt(row: dict) -> str:
    return (
        f"Cultural group: {row.get('cultural_group', '')}\n"
        f"Context: {row.get('context', 'social interaction')}\n"
        f"Character: {row.get('actor', 'a person')}\n"
        f"Culturally expected behavior in this group: {row['actor_behavior']}\n\n"
        "Write a short story (3-5 sentences) where the character "
        "VIOLATES this cultural norm — doing the opposite of what is expected. "
        "Include at least one other character who reacts. "
        "Do NOT use words like 'rude', 'inappropriate', or 'unacceptable'."
    )

def load_jsonl(filepath: str) -> list[dict]:
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def apply_chat_template(tokenizer, messages: list[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def build_prompts(
    rows: list[dict],
    tokenizer,
    mode: str,
) -> list[str]:
    prompts = []
    for row in rows:
        user_content = make_yes_prompt(row) if mode == "yes" else make_no_prompt(row)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        prompts.append(apply_chat_template(tokenizer, messages))
    return prompts


def run_generation(
    prompts: list[str],
    model_id: str,
    max_model_len: int,
    temperature: float = 0.8,
    max_tokens: int = 512,
) -> list[str]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_id,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=max_model_len,
    )
    params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
    )
    outputs = llm.generate(prompts, params)
    return [out.outputs[0].text.strip() for out in outputs]

def main():
    parser = argparse.ArgumentParser(description="Yes/No 시나리오 생성")
    parser.add_argument("--input", required=True, help="split_a_prm.jsonl 경로")
    parser.add_argument("--output", default="data/scenarios_a.jsonl")
    parser.add_argument("--model_id", required=True, help="모델 경로")
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    print(f"데이터 로드: {args.input}")
    data = load_jsonl(args.input)
    if args.limit:
        data = data[:args.limit]
    print(f"{len(data)}개 로드 완료")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for mode in ["yes", "no"]:
        print(f"\n[{mode.upper()} 시나리오 생성 중...]")
        prompts = build_prompts(data, tokenizer, mode)
        outputs = run_generation(prompts, args.model_id, args.max_model_len)

        with out_path.open("a", encoding="utf-8") as f:
            for row, story in zip(data, outputs):
                result = {
                    "id": f"{row.get('source_id', row.get('id', ''))}_{mode}",
                    "source_id": row.get("source_id", row.get("id", "")),
                    "cultural_group": row.get("cultural_group", ""),
                    "context": row.get("context", ""),
                    "actor": row.get("actor", ""),
                    "actor_behavior": row.get("actor_behavior", ""),
                    "agreement": row.get("agreement", 0),
                    "eval_whole_desc": row.get("eval_whole_desc", ""),
                    "gold_label": mode,
                    "story": story,
                    "generation_model": args.model_id,
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"  {mode} 시나리오 {len(outputs)}개 저장 완료")

    print(f"\n완료! 저장: {out_path}")


if __name__ == "__main__":
    main()
