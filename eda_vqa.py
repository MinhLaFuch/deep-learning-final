"""
eda_vqa.py — Exploratory Data Analysis cho VietCultural VQA dataset
Chạy: python eda_vqa.py
"""

import json
import os
from pathlib import Path
from collections import Counter, defaultdict
import statistics

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # headless — không cần display

# ── Cấu hình đường dẫn ────────────────────────────────────────────────────────
SPLIT_DIR  = Path("viet-cultural-vqa/splits")
IMAGE_DIR  = Path("viet-cultural-vqa/images")
OUT_DIR    = Path("eda_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CATEGORY = "kien_truc"   # None = tất cả; "Kiến trúc" = chỉ domain đó
MAX_ANSWER_LEN = 10      # chỉ giữ QA có answer <= 10 token

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_split(path: Path, target_category: str = None) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ["data", "samples", "items", "examples"]:
            if key in data:
                data = data[key]
                break
        else:
            data = next(iter(data.values()))

    result = []
    for r in data:
        cat = str(r.get("category", ""))
        if target_category and cat.strip().lower() != target_category.strip().lower():
            continue
        result.append(r)
    return result


def flatten_qa(records: list) -> list:
    """Trả về list of (image_id, category, question, answer)."""
    pairs = []
    for r in records:
        # Use image_path to create unique image_id (not just filename which repeats across categories)
        img_path = str(r.get("image_path", ""))
        category = str(r.get("category", ""))
        qs_raw   = r.get("questions", [])
        if isinstance(qs_raw, str):
            qs_raw = json.loads(qs_raw)
        for qa in qs_raw:
            pairs.append({
                "image_id": img_path,  # Use full path instead of just filename
                "category": category,
                "question": qa.get("question", ""),
                "answer":   qa.get("answer", ""),
            })
    return pairs


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading splits …")
train = load_split(SPLIT_DIR / "train_data.json", TARGET_CATEGORY)
val   = load_split(SPLIT_DIR / "val_data.json",   TARGET_CATEGORY)
test  = load_split(SPLIT_DIR / "test_data.json",  TARGET_CATEGORY)
all_records = train + val + test

train_qa = flatten_qa(train)
val_qa   = flatten_qa(val)
test_qa  = flatten_qa(test)
all_qa   = train_qa + val_qa + test_qa

# Lưu số lượng QA gốc trước khi lọc
total_train_qa = len(train_qa)
total_val_qa   = len(val_qa)
total_test_qa  = len(test_qa)
total_all_qa   = len(all_qa)

# Chỉ giữ QA có độ dài answer nhỏ hơn hoặc bằng MAX_ANSWER_LEN
all_qa = [p for p in all_qa if len(p["answer"].strip().split()) <= MAX_ANSWER_LEN]
train_qa = [p for p in train_qa if len(p["answer"].strip().split()) <= MAX_ANSWER_LEN]
val_qa   = [p for p in val_qa   if len(p["answer"].strip().split()) <= MAX_ANSWER_LEN]
test_qa  = [p for p in test_qa  if len(p["answer"].strip().split()) <= MAX_ANSWER_LEN]

# ── 1. Tổng số ảnh ────────────────────────────────────────────────────────────
unique_images = {p["image_id"] for p in all_qa}
total_images  = len(unique_images)
print(f"\n{'='*50}")
print(f"  Tổng số ảnh duy nhất : {total_images:,}")
print(f"  Tổng cặp QA đã lọc  : {len(all_qa):,} / {total_all_qa:,}")
print(f"    Train đã lọc      : {len(train_qa):,} / {total_train_qa:,}")
print(f"    Val đã lọc        : {len(val_qa):,} / {total_val_qa:,}")
print(f"    Test đã lọc       : {len(test_qa):,} / {total_test_qa:,}")

# ── 2. Số lượng QA / ảnh ──────────────────────────────────────────────────────
qa_per_image: dict[str, int] = Counter(p["image_id"] for p in all_qa)
qa_counts = list(qa_per_image.values())

print(f"\n  QA / ảnh:")
print(f"    Min  : {min(qa_counts)}")
print(f"    Max  : {max(qa_counts)}")
print(f"    Mean : {statistics.mean(qa_counts):.2f}")
print(f"    Mode : {statistics.mode(qa_counts)}")

# ── 3. Thống kê độ dài câu trả lời (theo token, split bằng space) ─────────────
answer_lengths = [len(p["answer"].strip().split()) for p in all_qa]
ans_min = min(answer_lengths)
ans_max = max(answer_lengths)
ans_avg = statistics.mean(answer_lengths)
ans_med = statistics.median(answer_lengths)

print(f"\n  Độ dài câu trả lời (# tokens):")
print(f"    Min    : {ans_min}")
print(f"    Max    : {ans_max}")
print(f"    Mean   : {ans_avg:.2f}")
print(f"    Median : {ans_med:.1f}")
print(f"{'='*50}\n")

# ── 4. Phân phối category ─────────────────────────────────────────────────────
cat_counter = Counter(p.get("category", "unknown") for p in all_qa)
print("  Phân phối category (top 10):")
for cat, cnt in cat_counter.most_common(10):
    print(f"    {cat:30s}: {cnt:,}")

# ── 5. Top 20 câu trả lời phổ biến nhất ──────────────────────────────────────
ans_counter = Counter(p["answer"].strip().lower() for p in all_qa)
print("\n  Top 20 câu trả lời phổ biến:")
for ans, cnt in ans_counter.most_common(20):
    print(f"    {ans:30s}: {cnt:,}")

# Xem image_id của ảnh có nhiều QA nhất
most_qa_img = max(qa_per_image, key=qa_per_image.get)
print(f"Image ID nhiều QA nhất: {most_qa_img} ({qa_per_image[most_qa_img]} QA)")

# Xem thử 5 câu hỏi đầu của ảnh đó
sample = [p for p in all_qa if p["image_id"] == most_qa_img][:5]
for s in sample:
    print(s["question"], "→", s["answer"])

# ── Vẽ biểu đồ ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("VietCultural VQA — EDA Overview", fontsize=14, fontweight="bold")

# 5a. QA / ảnh histogram
axes[0].hist(qa_counts, bins=range(1, max(qa_counts) + 2), color="steelblue", edgecolor="white")
axes[0].set_title("Số QA / ảnh")
axes[0].set_xlabel("Số cặp QA")
axes[0].set_ylabel("Số ảnh")
axes[0].grid(axis="y", alpha=0.3)

# 5b. Độ dài câu trả lời histogram
axes[1].hist(answer_lengths, bins=range(0, ans_max + 2), color="tomato", edgecolor="white")
axes[1].set_title("Độ dài câu trả lời (tokens)")
axes[1].set_xlabel("Số tokens")
axes[1].set_ylabel("Tần suất")
axes[1].axvline(ans_avg, color="black", linestyle="--", label=f"Mean={ans_avg:.1f}")
axes[1].legend()
axes[1].grid(axis="y", alpha=0.3)

# 5c. Category distribution
top_cats  = [c for c, _ in cat_counter.most_common(10)]
top_cnts  = [cat_counter[c] for c in top_cats]
axes[2].barh(top_cats[::-1], top_cnts[::-1], color="seagreen")
axes[2].set_title("Phân phối Category (Top 10)")
axes[2].set_xlabel("Số ảnh")
axes[2].grid(axis="x", alpha=0.3)

plt.tight_layout()
out_fig = OUT_DIR / "eda_overview.png"
fig.savefig(out_fig, dpi=130, bbox_inches="tight")
print(f"\nBiểu đồ EDA lưu tại: {out_fig}")

# ── Summary CSV ───────────────────────────────────────────────────────────────
import csv

summary = [
    ("total_images",      total_images),
    ("total_qa_pairs",    len(all_qa)),
    ("train_qa",          len(train_qa)),
    ("val_qa",            len(val_qa)),
    ("test_qa",           len(test_qa)),
    ("qa_per_img_min",    ans_min),
    ("qa_per_img_max",    max(qa_counts)),
    ("qa_per_img_mean",   round(statistics.mean(qa_counts), 2)),
    ("ans_len_min",       ans_min),
    ("ans_len_max",       ans_max),
    ("ans_len_mean",      round(ans_avg, 2)),
    ("ans_len_median",    ans_med),
]

csv_path = OUT_DIR / "eda_summary.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "value"])
    writer.writerows(summary)

print(f"Summary CSV lưu tại : {csv_path}")
print("\nEDA hoàn thành.")