from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import load_jsonl, get_source_id, parse_json_output
from generation import build_prompts_gptoss, load_llm, generate, _free_gpu


SYSTEM_PROMPT = """You are judging whether the following description expresses a cultural norm.

A cultural norm is a socially expected or appropriate behavior in a cultural group.
It reflects what people are expected to do, not just what they happen to do.

Do NOT confuse a cultural norm with:
- a common habit (frequently observed but not expected)
- an observed tendency (a statistical pattern, not a rule)
- an individual preference (personal choice)
- a neutral fact (descriptive information with no normative expectation)

A behavior should only be classified as a norm if it reflects a social expectation.
A strong indicator of a norm is that violating it would lead to social disapproval,
criticism, or perceived inappropriateness.

norm_strength scale:
  0 = not a norm at all
  1 = weak norm (optional, rarely enforced)
  2 = moderate norm (generally expected)
  3 = strong norm (strongly expected, violation causes social disapproval)

category options:
  "norm" / "habit" / "preference" / "fact"

Output JSON only:
{
  "is_norm": true or false,
  "norm_strength": 0 or 1 or 2 or 3,
  "category": "norm" or "habit" or "preference" or "fact",
  "reason": "one short sentence explaining your judgment"
}"""


class NormPromptConfig:
    def format_system(self) -> str:
        return SYSTEM_PROMPT

    def format_user(self, row: dict) -> str:
        return (
            f"Cultural group: {row.get('cultural_group', '')}\n"
            f"Context: {row.get('context', '')}\n"
            f"Description: {row.get('eval_whole_desc', row.get('actor_behavior', ''))}\n\n"
            "Is this a cultural norm?"
        )


def score_norm(
    data:                  list[dict],
    model_path:            str,
    output_path:           str,
    reasoning_effort:      str   = "low",
    batch_size:            int   = 32,
    tensor_parallel:       int   = 2,
    max_model_len:         int   = 8192,
    gpu_memory_utilization: float = 0.85,
    max_tokens:            int   = 1024,
) -> None:
    from transformers import AutoTokenizer

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if out_path.exists():
        for row in load_jsonl(str(out_path)):
            done_ids.add(row.get("source_id", ""))
        print(f"Resume: {len(done_ids)}개 이미 완료")

    remaining = [d for d in data if get_source_id(d) not in done_ids]
    if not remaining:
        print("모두 완료됨.")
        return

    print(f"처리: {len(remaining)}개")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    cfg       = NormPromptConfig()

    llm = load_llm(
        model_id               = model_path,
        tensor_parallel        = tensor_parallel,
        max_model_len          = max_model_len,
        gpu_memory_utilization = gpu_memory_utilization,
    )

    try:
        with out_path.open("a", encoding="utf-8") as f:
            for i in range(0, len(remaining), batch_size):
                batch   = remaining[i:i + batch_size]
                prompts = build_prompts_gptoss(batch, cfg, tokenizer, reasoning_effort)
                outputs = generate(llm, prompts, max_tokens=max_tokens)

                for row, raw in zip(batch, outputs):
                    parsed = parse_json_output(raw)
                    result = {
                        "source_id":    get_source_id(row),
                        "model_name":   "gpt_oss_20b",
                        "is_norm":      bool(parsed.get("is_norm", False)),
                        "norm_strength":int(parsed.get("norm_strength", 0)),
                        "category":     parsed.get("category", "habit"),
                        "reason":       parsed.get("reason", ""),
                        "raw_output":   raw,
                    }
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

                print(f"진행: {min(i+batch_size, len(remaining))}/{len(remaining)}", end="\r")
    finally:
        _free_gpu(llm)

    print(f"\n완료: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="norm-likeness scoring (GPT-OSS-20B)")
    parser.add_argument("--input",                  default="data/culturebank_reddit_filtered.jsonl")
    parser.add_argument("--output",                 default="data/norm_scores/norm_gpt_oss_20b.jsonl")
    parser.add_argument("--model_path",             default="/home/nlpgpu10/hdd2/jieun/gpt-oss-20B")
    parser.add_argument("--reasoning_effort",       default="high", choices=["low", "medium", "high"])
    parser.add_argument("--batch_size",             type=int,   default=16)
    parser.add_argument("--tensor_parallel",        type=int,   default=2)
    parser.add_argument("--max_model_len",          type=int,   default=8192)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_tokens",             type=int,   default=4096)
    parser.add_argument("--limit",                  type=int,   default=None)
    args = parser.parse_args()

    data = load_jsonl(args.input)
    if args.limit:
        data = data[:args.limit]

    print(f"데이터: {len(data):,}개")
    print(f"reasoning_effort:       {args.reasoning_effort}")
    print(f"tensor_parallel:        {args.tensor_parallel}")
    print(f"max_model_len:          {args.max_model_len}")
    print(f"gpu_memory_utilization: {args.gpu_memory_utilization}")
    print(f"max_tokens:             {args.max_tokens}")

    score_norm(
        data                   = data,
        model_path             = args.model_path,
        output_path            = args.output,
        reasoning_effort       = args.reasoning_effort,
        batch_size             = args.batch_size,
        tensor_parallel        = args.tensor_parallel,
        max_model_len          = args.max_model_len,
        gpu_memory_utilization = args.gpu_memory_utilization,
        max_tokens             = args.max_tokens,
    )

    print(f"\n다음 단계:")
    print(f"  (다른 모델 scoring 후)")
    print(f"  python step0c_select.py --input {args.input} --score_dir data/norm_scores")


if __name__ == "__main__":
    main()
