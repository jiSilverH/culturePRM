from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer

from setproctitle import setproctitle


def parse_answer_rao(text) -> str:
    if not isinstance(text, str) or not text.strip():
        return "neutral"

    for close_tag in ["</think>", "[/THINK]", "<|/thinking|>"]:
        if close_tag in text:
            text = text.split(close_tag)[-1]
            break

    if not text.strip():
        return "neutral"

    cleaned = text

    cleaned = re.sub(
        r"\b[1-3]\s*\)\s*(?:yes|no|neither|neutral)\b",
        "", cleaned, flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\boptions?\s*(?:are\s*|:\s*)?[\(\[]?"
        r"\s*(?:yes|no|neither|neutral|or|,|/|\s)+",
        "", cleaned, flags=re.IGNORECASE,
    )

    last_match = None
    for m in re.finditer(r"\b(neither|neutral|yes|no)\b", cleaned, re.IGNORECASE):
        last_match = m

    if last_match is None:
        low = text.lower()
        if "option 1" in low: return "yes"
        if "option 2" in low: return "no"
        if "option 3" in low: return "neutral"
        return "neutral"

    ans = last_match.group(1).lower()
    return "neutral" if ans == "neither" else ans

def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def build_prm_input(country: str, rot: str, story: str, sentence: str) -> str:
    return (
        f"Country: {country}\n"
        f"Rule-of-Thumb: {rot}\n\n"
        f"Story: {story}\n\n"
        f"Reasoning step: {sentence}"
    )


def norm(x):
    x = str(x).lower().strip()
    if x == "neither":
        return "neutral"
    return x


def compute_all_metrics(preds: list[dict]) -> dict:
    total = len(preds)
    correct = sum(1 for p in preds if norm(p["pred"]) == norm(p["gold"]))

    by_gold = defaultdict(list)
    for p in preds:
        g = norm(p["gold"])
        pr = norm(p["pred"])
        by_gold[g].append(pr == g)
    per_class = {k: sum(v) / max(len(v), 1) for k, v in by_gold.items()}

    bs = [p for p in preds
          if norm(p["gold"]) in ("yes", "no") and norm(p["pred"]) in ("yes", "no")]
    bs_correct = sum(1 for p in bs if norm(p["pred"]) == norm(p["gold"]))

    bp = [p for p in preds if norm(p["gold"]) in ("yes", "no")]
    bp_correct = sum(1 for p in bp if norm(p["pred"]) == norm(p["gold"]))

    return {
        "3class_acc":      correct / max(total, 1),
        "yes_acc":         per_class.get("yes", 0.0),
        "no_acc":          per_class.get("no", 0.0),
        "neutral_acc":     per_class.get("neutral", 0.0),
        "binary_strict":   bs_correct / max(len(bs), 1),
        "binary_strict_n": len(bs),
        "binary_permissive":  bp_correct / max(len(bp), 1),
        "binary_permissive_n": len(bp),
        "total":           total,
        "correct":         correct,
        "pred_dist":       dict(Counter(norm(p["pred"]) for p in preds)),
        "gold_dist":       dict(Counter(norm(p["gold"]) for p in preds)),
    }

def safe_extract_thinking(completion) -> str:
    if not isinstance(completion, str):
        return ""
    if "</think>" in completion:
        return completion.split("</think>")[0].replace("<think>", "").strip()
    return completion.strip()

def load_traces_cache(path: str) -> list[dict]:
    rows = []
    n_empty = 0
    n_bad = 0
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                n_empty += 1
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                n_bad += 1
                print(f"  [warn] line {line_no}: JSON 파싱 실패, 건너뜀 ({e})")
                continue
            if not isinstance(d, dict):
                n_bad += 1
                print(f"  [warn] line {line_no}: dict 아님, 건너뜀")
                continue
            if "completions" not in d or not d["completions"]:
                n_bad += 1
                print(f"  [warn] line {line_no}: completions 비어있음, 건너뜀")
                continue
            rows.append(d)
    return rows, n_empty, n_bad

def worker_process(
    rank: int,
    gpu_id: int,
    inputs_shard: list[tuple],
    reward_model_path: str,
    reward_type: str,
    batch_size: int,
    max_length: int,
    return_dict,
):
    setproctitle(f"python3")

    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    import sys
    sys.path.append(str(Path(__file__).parent))

    tokenizer = AutoTokenizer.from_pretrained(reward_model_path)

    if reward_type == "prm":
        from train_prm import CulturePRM
        model = CulturePRM(reward_model_path).to(device).to(torch.bfloat16)
        model.regressor.load_state_dict(
            torch.load(f"{reward_model_path}/regressor.pt", map_location=device)
        )
    else:
        from train_orm import CultureORM
        model = CultureORM(reward_model_path).to(device).to(torch.bfloat16)
        model.classifier.load_state_dict(
            torch.load(f"{reward_model_path}/classifier.pt", map_location=device)
        )
    model.eval()

    indexed = list(enumerate(inputs_shard))
    indexed.sort(key=lambda x: len(x[1][1]))

    results = {}

    if rank == 0:
        from tqdm import tqdm
        iterator = tqdm(range(0, len(indexed), batch_size),
                        desc=f"GPU{gpu_id}", position=rank)
    else:
        iterator = range(0, len(indexed), batch_size)

    for i in iterator:
        batch_indexed = indexed[i:i + batch_size]
        global_indices = [x[1][0] for x in batch_indexed]
        texts = [x[1][1] for x in batch_indexed]

        enc = tokenizer(
            texts, return_tensors="pt",
            max_length=max_length, truncation=True, padding=True,
        ).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(**enc)
            if reward_type == "prm":
                if hasattr(out, "logits"):
                    scores = torch.tanh(out.logits.squeeze(-1))
                else:
                    scores = out.squeeze(-1)
                if scores.dim() == 0:
                    scores = scores.unsqueeze(0)
                score_list = scores.float().cpu().tolist()
            else:
                logits = out.logits if hasattr(out, "logits") else out
                probs = torch.softmax(logits.float(), dim=-1)
                score_list = (probs[:, 1] * 2 - 1).cpu().tolist()

        for gidx, s in zip(global_indices, score_list):
            results[gidx] = s

    return_dict[rank] = results


def parallel_score(
    all_inputs: list[str],
    n_gpus: int,
    reward_model_path: str,
    reward_type: str,
    batch_size: int,
    max_length: int,
) -> list[float]:
    indexed_inputs = list(enumerate(all_inputs))

    shards = [[] for _ in range(n_gpus)]
    for i, item in enumerate(indexed_inputs):
        shards[i % n_gpus].append(item)

    print(f"  Sharding: {[len(s) for s in shards]}")

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        gpu_ids = [int(x) for x in cuda_visible.split(",")]
        gpu_ids_for_workers = list(range(n_gpus))
    else:
        gpu_ids_for_workers = list(range(n_gpus))

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    return_dict = manager.dict()

    processes = []
    for rank in range(n_gpus):
        p = mp.Process(
            target=worker_process,
            args=(rank, gpu_ids_for_workers[rank], shards[rank],
                  reward_model_path, reward_type,
                  batch_size, max_length, return_dict),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    all_scores = [0.0] * len(all_inputs)
    for rank in range(n_gpus):
        for gidx, score in return_dict[rank].items():
            all_scores[gidx] = score

    return all_scores


def write_empty_metrics(args, reason: str):
    out_path  = Path(args.output)
    meta_path = out_path.with_suffix(".metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.touch()
    with open(meta_path, "w") as f:
        json.dump({
            "traces_cache":    args.traces_cache,
            "reward_model":    args.reward_model_path,
            "reward_type":     args.reward_type,
            "prm_aggregation": args.prm_aggregation if args.reward_type == "prm" else None,
            "n_samples":       0,
            "n_rows":          0,
            "error":           reason,
        }, f, indent=2)
    print(f"[error meta] {meta_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces_cache",      required=True)
    parser.add_argument("--reward_model_path", required=True)
    parser.add_argument("--output",            required=True)
    parser.add_argument("--reward_type",       default="prm",
                        choices=["prm", "orm"])
    parser.add_argument("--prm_aggregation",   default="last",
                        choices=["min", "mean", "last"])
    parser.add_argument("--seed",              type=int, default=42)
    parser.add_argument("--batch_size",        type=int, default=1024,
                        help="멀티 GPU 시 GPU 당 batch. 1B 모델이면 2048-4096 가능")
    parser.add_argument("--n_use",             type=int, default=None)
    parser.add_argument("--skip_on_error",     action="store_true", default=True,
                        help="유효 row 가 0 이거나 cache 가 망가졌으면 exit 0 으로 종료")
    args = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError("No GPU available")
    print(f"\nDetected {n_gpus} GPUs")

    print(f"\nCache 로드: {args.traces_cache}")
    if not Path(args.traces_cache).exists():
        print(f"  [error] cache 파일 없음")
        write_empty_metrics(args, "traces_cache file not found")
        return

    try:
        rows, n_empty, n_bad = load_traces_cache(args.traces_cache)
    except Exception as e:
        print(f"  [error] cache 로드 실패: {e}")
        write_empty_metrics(args, f"cache load error: {e}")
        return

    print(f"  {len(rows)} valid rows ({n_empty} empty, {n_bad} bad lines skipped)")

    if not rows:
        print(f"  [error] 유효한 row 가 0 개. 평가 건너뜀.")
        write_empty_metrics(args, "no valid rows in cache")
        return

    cache_n = len(rows[0]["completions"])
    n_samples = args.n_use if args.n_use else cache_n
    if args.n_use:
        for row in rows:
            row["completions"] = row["completions"][:n_samples]
    print(f"  N samples: {n_samples}")

    if args.reward_type == "prm":
        all_inputs, meta = [], []
        for row_idx, row in enumerate(rows):
            country = row.get("country", "")
            story   = row.get("story", "")
            rot     = row.get("rule_of_thumb", "")

            for comp_idx, completion in enumerate(row["completions"]):
                thinking = safe_extract_thinking(completion)
                sentences = split_sentences(thinking) if thinking else []

                if not sentences:
                    all_inputs.append(build_prm_input(country, rot, story, "."))
                    meta.append((row_idx, comp_idx, 0, True))
                else:
                    for sent_idx, sentence in enumerate(sentences):
                        all_inputs.append(
                            build_prm_input(country, rot, story, sentence)
                        )
                        meta.append((row_idx, comp_idx, sent_idx, False))
        max_length = 512
        print(f"\nTotal PRM steps: {len(all_inputs):,}")
    else:
        all_inputs, meta = [], []
        for row_idx, row in enumerate(rows):
            country = row.get("country", "")
            story   = row.get("story", "")
            for comp_idx, completion in enumerate(row["completions"]):
                thinking = safe_extract_thinking(completion)
                input_text = (
                    f"Country: {country}\n\n"
                    f"Story: {story}\n\n"
                    f"Reasoning: {thinking[:1000]}"
                )
                all_inputs.append(input_text)
                meta.append((row_idx, comp_idx, 0, False))
        max_length = 4096
        print(f"\nTotal ORM traces: {len(all_inputs):,}")

    if not all_inputs:
        print(f"  [error] inputs 가 0 개. 평가 건너뜀.")
        write_empty_metrics(args, "no valid inputs after preprocessing")
        return

    print(f"\nScoring on {n_gpus} GPU(s) | batch_size={args.batch_size} | bf16")

    if n_gpus > 1:
        all_scores = parallel_score(
            all_inputs, n_gpus,
            args.reward_model_path, args.reward_type,
            args.batch_size, max_length,
        )
    else:
        from tqdm import tqdm
        device = torch.device("cuda:0")
        tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path)

        import sys
        sys.path.append(str(Path(__file__).parent))
        if args.reward_type == "prm":
            from train_prm import CulturePRM
            model = CulturePRM(args.reward_model_path).to(device).to(torch.bfloat16)
            model.regressor.load_state_dict(
                torch.load(f"{args.reward_model_path}/regressor.pt", map_location=device)
            )
        else:
            from train_orm import CultureORM
            model = CultureORM(args.reward_model_path).to(device).to(torch.bfloat16)
            model.classifier.load_state_dict(
                torch.load(f"{args.reward_model_path}/classifier.pt", map_location=device)
            )
        model.eval()

        indexed = list(enumerate(all_inputs))
        indexed.sort(key=lambda x: len(x[1]))

        all_scores = [0.0] * len(all_inputs)
        for i in tqdm(range(0, len(indexed), args.batch_size), desc="Scoring"):
            batch = indexed[i:i + args.batch_size]
            indices = [x[0] for x in batch]
            texts = [x[1] for x in batch]

            enc = tokenizer(
                texts, return_tensors="pt",
                max_length=max_length, truncation=True, padding=True,
            ).to(device)

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(**enc)
                if args.reward_type == "prm":
                    if hasattr(out, "logits"):
                        scores = torch.tanh(out.logits.squeeze(-1))
                    else:
                        scores = out.squeeze(-1)
                    if scores.dim() == 0:
                        scores = scores.unsqueeze(0)
                    score_list = scores.float().cpu().tolist()
                else:
                    logits = out.logits if hasattr(out, "logits") else out
                    probs = torch.softmax(logits.float(), dim=-1)
                    score_list = (probs[:, 1] * 2 - 1).cpu().tolist()

            for idx, s in zip(indices, score_list):
                all_scores[idx] = s

    if args.reward_type == "prm":
        step_rewards = defaultdict(lambda: defaultdict(list))
        for score, (row_idx, comp_idx, _, is_dummy) in zip(all_scores, meta):
            if not is_dummy:
                step_rewards[row_idx][comp_idx].append(score)

        def aggregate(rewards, method):
            if not rewards: return -1.0
            if method == "min":  return min(rewards)
            if method == "mean": return sum(rewards) / len(rewards)
            if method == "last": return rewards[-1]
            return min(rewards)
    else:
        orm_score_map = {}
        for score, (row_idx, comp_idx, _, _) in zip(all_scores, meta):
            orm_score_map[(row_idx, comp_idx)] = score

    random.seed(args.seed)
    preds_bon, preds_rand, preds_base = [], [], []
    results = []

    for row_idx, row in enumerate(rows):
        completions = row["completions"]
        gold = row.get("gold", "")

        if args.reward_type == "prm":
            scores = [
                aggregate(step_rewards[row_idx][comp_idx], args.prm_aggregation)
                for comp_idx in range(len(completions))
            ]
        else:
            scores = [
                orm_score_map.get((row_idx, comp_idx), -1.0)
                for comp_idx in range(len(completions))
            ]

        try:
            best_idx = scores.index(max(scores))
        except (ValueError, TypeError):
            best_idx = 0

        pred_bon  = parse_answer_rao(completions[best_idx])
        pred_rand = parse_answer_rao(random.choice(completions))
        pred_base = parse_answer_rao(completions[0])

        common = {"gold": gold, "country": row.get("country", "")}
        preds_bon.append( {**common, "pred": pred_bon})
        preds_rand.append({**common, "pred": pred_rand})
        preds_base.append({**common, "pred": pred_base})

        results.append({
            "id":         row.get("id", row_idx),
            "country":    row.get("country", ""),
            "story":      row.get("story", ""),
            "gold":       gold,
            "pred_bon":   pred_bon,
            "pred_rand":  pred_rand,
            "pred_base":  pred_base,
            "best_score": max(scores) if scores else 0.0,
            "scores":     scores,
        })

    bon_m  = compute_all_metrics(preds_bon)
    rand_m = compute_all_metrics(preds_rand)
    base_m = compute_all_metrics(preds_base)

    label = args.reward_type.upper()
    if args.reward_type == "prm":
        label = f"PRM-{args.prm_aggregation}"

    print(f"\n{'='*70}")
    print(f"Best-of-{n_samples} (Rao prompt + Rao parsing) | reward: {label}")
    print(f"GPUs: {n_gpus} | Cache: {args.traces_cache}")
    print(f"{'='*70}")

    print(f"\n{'Method':30s} {'3-class':>9s} {'Binary-S':>9s} {'Binary-P':>9s} "
          f"{'yes':>7s} {'no':>7s} {'neutral':>9s}")
    for name, m in [
        ("Base (1st sample)",                base_m),
        (f"Random-of-{n_samples}",            rand_m),
        (f"Best-of-{n_samples} ({label})",    bon_m),
    ]:
        print(f"{name:30s} {m['3class_acc']:>8.1%} "
              f"{m['binary_strict']:>8.1%} {m['binary_permissive']:>8.1%} "
              f"{m['yes_acc']:>6.1%} {m['no_acc']:>6.1%} {m['neutral_acc']:>8.1%}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    metrics_path = out_path.with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "traces_cache":    args.traces_cache,
            "reward_model":    args.reward_model_path,
            "reward_type":     args.reward_type,
            "prm_aggregation": args.prm_aggregation if args.reward_type=="prm" else None,
            "n_samples":       n_samples,
            "n_rows":          len(rows),
            "n_gpus":          n_gpus,
            "batch_size":      args.batch_size,
            "prompt_style":    "rao",
            "parser":          "rao_fixed",
            "metrics": {
                "bon":    bon_m,
                "random": rand_m,
                "base":   base_m,
            },
        }, f, indent=2, ensure_ascii=False)

    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
