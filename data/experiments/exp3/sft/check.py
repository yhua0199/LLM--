import json

file_path = "train.json"  # 确保这里是你的文件路径

print(f"正在检查文件: {file_path} ...")

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. 检查是不是列表
    if not isinstance(data, list):
        print(f"❌ 错误: 文件最外层是 {type(data)}，必须是 list (数组)！")
        print("提示: 你的文件可能还是 JSONL 格式，或者缺少最外层的 []。")
        exit()

    print(f"✅ 格式正确: 数据是一个列表，包含 {len(data)} 条样本。")

    # 2. 检查第一条数据的 Key
    if len(data) > 0:
        first_item = data[0]
        print(f"ℹ️ 第一条数据的字段 (Keys): {list(first_item.keys())}")

        # 3. 验证是否与 dataset_info.json 匹配
        required_keys = ["instruction", "output"]  # 根据 Alpaca 格式
        missing = [k for k in required_keys if k not in first_item]

        if missing:
            print(f"❌ 致命错误: 数据中缺少必要字段: {missing}")
            print("dataset_info.json 要求 columns 为: instruction, output")
            print("但你的数据里没有这些 Key。请修改 dataset_info.json 或修改数据 key。")
        else:
            print("✅ 字段检查通过: 包含 instruction 和 output。")

            # 4. 检查是否有空值
            if not first_item.get("instruction") and not first_item.get("input"):
                print("⚠️ 警告: 第一条数据的 instruction 和 input 都为空，这可能导致被过滤。")

except json.JSONDecodeError:
    print("❌ JSON 解析错误: 文件格式不合法。可能有多余的逗号，或者不是标准 JSON。")
except FileNotFoundError:
    print(f"❌ 找不到文件: {file_path}")