import json

with open("./viet-cultural-vqa/splits/train_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Xem record đầu tiên
sample = data[0] if isinstance(data, list) else next(iter(data.values()))[0]
print(json.dumps(sample, ensure_ascii=False, indent=2))