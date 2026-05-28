from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from utils import load_jsonl, get_source_id
from generation import build_prompts_gptoss, build_prompts_default, load_llm, generate, _free_gpu
from prompt import get_prompt_config


def extract_thinking(raw: str) -> str:
    think_match = re.search(r"<think>(.*?)</think>", raw, flags=re.DOTALL)
    if think_match:
        return think_match.group(1).strip()
    analysis_match = re.search(r"^analysis(.*?)assistantfinal", raw, flags=re.DOTALL)
    if analysis_match:
        return analysis_match.group(1).strip()
    return ""


def parse_final_answer(raw: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    text = re.sub(r"^analysis.*?assistantfinal\s*", "", text, flags=re.DOTALL)

    for line in reversed(text.strip().split("\n")):
        match = re.search(r"\b(Yes|No|Neither)\b", line, re.IGNORECASE)
        if match:
            return match.group(1).lower()

    match = re.search(r"\b(Yes|No|Neither)\b", text, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    return "unknown"

def generate_traces(
    data:                  list[dict],
    model_path:            str,
    output_path:           str,
    prompt_type:           str   = "A",
    reasoning_effort:      str   = "high",
    batch_size:            int   = 16,
    tensor_parallel:       int   = 1,
    max_model_len:         int   = 16384,
    gpu_memory_utilization: float = 0.85,
    max_tokens:            int   = 16000,
    gold_label_filter:     str | None = None,
    model_type:            str   = "gptoss",
    start_source_id:       str | None = None,
) -> None:
    from transformers import AutoTokenizer

    cfg      = get_prompt_config(prompt_type)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if out_path.exists():
        for row in load_jsonl(str(out_path)):
            done_ids.add(row.get("id", ""))
        print(f"Resume: {len(done_ids)}개 이미 완료")

    if start_source_id is not None:
        source_ids = [str(d.get("source_id", "")) for d in data]
        try:
            start_idx = source_ids.index(str(start_source_id))
            data = data[start_idx:]
            print(f"start_source_id={start_source_id} → {len(data)}개부터 시작")
        except ValueError:
            print(f"⚠️  source_id '{start_source_id}' 없음 → 전체 처리")

    remaining = [
        d for d in data
        if d.get("id", "") not in done_ids
        and d.get("story", "").strip()
        and (gold_label_filter is None or d.get("gold_label") == gold_label_filter)
    ]
    if gold_label_filter:
        print(f"gold_label 필터: '{gold_label_filter}'만 처리")
    if not remaining:
        print("모두 완료됨.")
        return

    print(f"처리: {len(remaining)}개 | prompt_type={prompt_type}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    llm = load_llm(
        model_id               = model_path,
        tensor_parallel        = tensor_parallel,
        max_model_len          = max_model_len,
        gpu_memory_utilization = gpu_memory_utilization,
    )

    n_total    = len(remaining)
    n_done     = 0
    n_override = 0
    n_unknown  = 0
    start_time = time.time()

    try:
        with out_path.open("a", encoding="utf-8") as f:
            pbar = tqdm(total=n_total, unit="story", desc=f"trace 생성 [Prompt {prompt_type}]")
            for i in range(0, n_total, batch_size):
                batch   = remaining[i:i + batch_size]
                if model_type == "gptoss":
                    prompts = build_prompts_gptoss(batch, cfg, tokenizer, reasoning_effort)
                else:
                    prompts = build_prompts_default(batch, cfg, tokenizer)
                outputs = generate(llm, prompts, max_tokens=max_tokens)

                for row, raw in zip(batch, outputs):
                    thinking     = extract_thinking(raw)
                    final_answer = parse_final_answer(raw)

                    n_done += 1
                    if final_answer == "unknown":
                        n_unknown += 1
                    if row.get("gold_label") == "no" and final_answer == "yes":
                        n_override += 1

                    result = {
                        "id":             row.get("id", ""),
                        "source_id":      get_source_id(row),
                        "cultural_group": row.get("cultural_group", ""),
                        "gold_label":     row.get("gold_label", ""),
                        "story":          row.get("story", ""),
                        "rule_of_thumb":  row.get("rule_of_thumb", ""),
                        "actor_behavior": row.get("actor_behavior", ""),
                        "prompt_type":    prompt_type,
                        "reasoning_effort": reasoning_effort,
                        "thinking":       thinking,
                        "final_answer":   final_answer,
                        "raw_output":     raw,
                    }
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

                pbar.update(len(batch))

                elapsed       = time.time() - start_time
                speed         = n_done / elapsed if elapsed > 0 else 0
                remaining_sec = (n_total - n_done) / speed if speed > 0 else 0
                pbar.set_postfix({
                    "done":     f"{n_done}/{n_total}",
                    "override": f"{n_override}({n_override/max(n_done,1):.0%})",
                    "unknown":  n_unknown,
                    "speed":    f"{speed:.1f}it/s",
                    "ETA":      f"{remaining_sec/3600:.1f}h",
                })
            pbar.close()

        elapsed = time.time() - start_time
        print(f"\n{'='*50}")
        print(f"완료: {n_done}개 | 소요: {elapsed/3600:.1f}h ({elapsed/60:.0f}min)")
        print(f"override (no→yes): {n_override}개 ({n_override/max(n_done,1):.1%})")
        print(f"unknown:           {n_unknown}개 ({n_unknown/max(n_done,1):.1%})")
        print(f"speed:             {n_done/elapsed:.1f} stories/sec")
        print(f"{'='*50}")
    finally:
        _free_gpu(llm)

    print(f"\n완료: {out_path}")

    traces  = load_jsonl(str(out_path))
    unknown = sum(1 for d in traces if d["final_answer"] == "unknown")
    by_label: dict[str, dict] = {}
    for d in traces:
        gl = d["gold_label"]
        fa = d["final_answer"]
        by_label.setdefault(gl, {})
        by_label[gl][fa] = by_label[gl].get(fa, 0) + 1

    print(f"\n=== 추론 결과 [Prompt {prompt_type}] ===")
    print(f"총: {len(traces)}개 | 파싱 실패(unknown): {unknown}개 ({unknown/max(len(traces),1):.1%})")
    for gl, dist in sorted(by_label.items()):
        print(f"  gold={gl}: {dist}")

def main():
    parser = argparse.ArgumentParser(description="Reasoning trace 생성")
    parser.add_argument("--input",                  required=True,
                        help="scenarios.jsonl 경로 (step0e 출력)")
    parser.add_argument("--output",                 default="data/traces_A.jsonl")
    parser.add_argument("--model_path",             required=True)
    parser.add_argument("--prompt_type",            default="A", choices=["A", "B", "C"],
                        help="A: country+story / B: +rule_of_thumb / C: +background")
    parser.add_argument("--reasoning_effort",       default="high",
                        choices=["low", "medium", "high"])
    parser.add_argument("--batch_size",             type=int,   default=16)
    parser.add_argument("--tensor_parallel",        type=int,   default=2)
    parser.add_argument("--max_model_len",          type=int,   default=16384)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_tokens",             type=int,   default=16000)
    parser.add_argument("--limit",                  type=int,   default=None)
    parser.add_argument("--gold_label_filter",      default=None,
                        choices=["yes", "no", "neutral"],
                        help="특정 gold_label만 처리 (예: 'no'만 처리해서 시간 절약)")
    parser.add_argument("--model_type",              default="gptoss",
                        choices=["gptoss", "default"],
                        help="gptoss: GPT-OSS 계열 / default: DeepSeek, Qwen, Llama 등")
    parser.add_argument("--start_source_id",         default=None,
                        help="이 source_id부터 처리 (예: '2925')")
    args = parser.parse_args()

    data = load_jsonl(args.input)
    if args.limit:
        data = data[:args.limit]

    total_filtered = sum(1 for d in data
                         if args.gold_label_filter is None
                         or d.get("gold_label") == args.gold_label_filter)
    print(f"시나리오:         {len(data):,}개 → 처리 대상: {total_filtered:,}개")
    print(f"prompt_type:      {args.prompt_type}")
    print(f"reasoning_effort: {args.reasoning_effort}")
    print(f"tensor_parallel:  {args.tensor_parallel}")
    print(f"max_model_len:    {args.max_model_len}")
    print(f"max_tokens:       {args.max_tokens}")
    if args.gold_label_filter:
        print(f"gold_label_filter: {args.gold_label_filter}")

    generate_traces(
        data                   = data,
        model_path             = args.model_path,
        output_path            = args.output,
        prompt_type            = args.prompt_type,
        reasoning_effort       = args.reasoning_effort,
        batch_size             = args.batch_size,
        tensor_parallel        = args.tensor_parallel,
        max_model_len          = args.max_model_len,
        gpu_memory_utilization = args.gpu_memory_utilization,
        max_tokens             = args.max_tokens,
        gold_label_filter      = args.gold_label_filter,
        model_type             = args.model_type,
        start_source_id        = args.start_source_id,
    )

    print(f"\n다음 단계:")
    print(f"  python step0g_empirical_screen.py \\\n"
          f"    --input {args.output} \\\n"
          f"    --scenarios {args.input}")


if __name__ == "__main__":
    main()
