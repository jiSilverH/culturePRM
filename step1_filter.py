from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import load_jsonl, save_jsonl


VAGUE_KEYWORDS = [
    "various", "different ways", "many forms",
    "multiple", "wide range", "diverse ways",
]


def filter_data(data: list[dict], min_agreement: float = 0.7) -> list[dict]:
    filtered = []
    for row in data:
        if row.get("agreement", 0) < min_agreement:
            continue
        behavior = row.get("actor_behavior", "")
        if len(behavior.split()) < 8:
            continue
        if any(v in behavior.lower() for v in VAGUE_KEYWORDS):
            continue
        filtered.append(row)
    return filtered


def main():
    parser = argparse.ArgumentParser(description="CultureBank 기본 필터링")
    parser.add_argument("--input",         required=True)
    parser.add_argument("--output",        default="data/filtered.jsonl")
    parser.add_argument("--min_agreement", type=float, default=0.7)
    parser.add_argument("--limit",         type=int,   default=None)
    args = parser.parse_args()

    data = load_jsonl(args.input)
    if args.limit:
        data = data[:args.limit]

    filtered = filter_data(data, args.min_agreement)
    save_jsonl(filtered, args.output)

    print(f"전체:    {len(data):,}개")
    print(f"필터 후: {len(filtered):,}개 ({len(filtered)/len(data):.1%})")

    topic_dist = Counter(d.get("topic", "N/A") for d in filtered)
    print(f"\nTopic 분포 (상위 5개):")
    for t, cnt in topic_dist.most_common(5):
        print(f"  {t}: {cnt}개")

    print(f"\n저장: {args.output}")
    print(f"다음: python step0b_score_norm.py --input {args.output}")


if __name__ == "__main__":
    main()
