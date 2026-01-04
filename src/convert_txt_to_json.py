import json

from common.paths import data_path

input_path = data_path("rewrite", "seed", "article.txt")
output_path = data_path("rewrite", "seed", "law_articles.json")

records = []

with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        obj = json.loads(line)
        input_text = obj.get("input", "").strip()
        answer_text = obj.get("answer", "").strip()

        if answer_text.startswith(input_text):
            content = answer_text[len(input_text):].strip()
        else:
            content = answer_text

        records.append({
            "law_title": input_text,
            "content": content
        })

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"转换完成，共 {len(records)} 条")
print(f"输出路径：{output_path}")
