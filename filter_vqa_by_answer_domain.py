"""filter_vqa_by_answer_domain.py

Lọc dataset VQA theo domain (category) và theo số token trong answer.

Usage examples:
  python filter_vqa_by_answer_domain.py \
    --input viet-cultural-vqa/splits/train_data.json \
    --output filtered_train.json \
    --domain kien_truc \
    --max-answer-tokens 10

  python filter_vqa_by_answer_domain.py --input viet-cultural-vqa/splits/train_data.json --output filtered_train.json --domain kien_truc --max-answer-tokens 10 --preserve-records
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ["data", "samples", "items", "examples"]:
            if key in data:
                return data[key]
        values = list(data.values())
        if len(values) == 1 and isinstance(values[0], list):
            return values[0]
    return data


def normalize_category(cat: Optional[str]) -> str:
    return str(cat or "").strip().lower()


def parse_questions(raw: Any) -> List[Dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    if isinstance(raw, list):
        return raw
    return []


def answer_token_count(answer: Any) -> int:
    if answer is None:
        return 0
    return len(str(answer).strip().split())


def filter_record(record: Dict[str, Any], domain: Optional[str], max_tokens: int) -> Optional[Dict[str, Any]]:
    category = normalize_category(record.get("category"))
    if domain and category != normalize_category(domain):
        return None

    questions = parse_questions(record.get("questions"))
    filtered = [qa for qa in questions if answer_token_count(qa.get("answer")) <= max_tokens]
    if not filtered:
        return None

    new_record = dict(record)
    new_record["questions"] = filtered
    return new_record


def flatten_record(record: Dict[str, Any], domain: Optional[str], max_tokens: int) -> List[Dict[str, Any]]:
    category = normalize_category(record.get("category"))
    if domain and category != normalize_category(domain):
        return []

    img_id = str(record.get("image_path", record.get("image_id", "")))
    questions = parse_questions(record.get("questions"))
    pairs: List[Dict[str, Any]] = []
    for qa in questions:
        if answer_token_count(qa.get("answer")) <= max_tokens:
            pairs.append({
                "image_id": img_id,
                "category": record.get("category", ""),
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
            })
    return pairs


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def process_file(input_path: Path, output_path: Path, domain: Optional[str], max_tokens: int, preserve_records: bool) -> None:
    data = load_json(input_path)
    if not isinstance(data, list):
        raise ValueError(f"Không thể đọc dữ liệu record list từ {input_path}")

    if preserve_records:
        filtered_records = [r for r in (filter_record(rec, domain, max_tokens) for rec in data) if r is not None]
        save_json(filtered_records, output_path)
        print(f"Saved {len(filtered_records)} records to {output_path}")
    else:
        filtered_pairs = []
        for rec in data:
            filtered_pairs.extend(flatten_record(rec, domain, max_tokens))
        save_json(filtered_pairs, output_path)
        print(f"Saved {len(filtered_pairs)} QA pairs to {output_path}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lọc dataset VQA theo domain và số token trong answer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", nargs="+", required=True, help="File JSON đầu vào")
    parser.add_argument("--output", "-o", nargs="+", required=True, help="File JSON đầu ra tương ứng")
    parser.add_argument("--domain", "-d", default=None, help="Domain/category lọc (ví dụ: kien_truc)")
    parser.add_argument("--max-answer-tokens", "-m", type=int, default=10, help="Số token tối đa cho answer")
    parser.add_argument("--preserve-records", action="store_true", help="Giữ cấu trúc record gốc và chỉ lọc câu hỏi bên trong")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    input_paths = [Path(p) for p in args.input]
    output_paths = [Path(p) for p in args.output]
    if len(input_paths) != len(output_paths):
        raise SystemExit("Số lượng file input và output phải bằng nhau.")

    for src, dst in zip(input_paths, output_paths):
        process_file(src, dst, args.domain, args.max_answer_tokens, args.preserve_records)


if __name__ == "__main__":
    main()
