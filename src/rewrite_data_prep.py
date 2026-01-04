'''
总共需要采样200条问答对，但是实际中的问答对质量参差不齐，所以采样250条问答对，人工校验至200条
剔除乱码或语病严重的问题：无法理解提问意图的。
剔除答案无效的：如答案是“请拨打我的电话”、“加我微信”或“需要看具体情况”这种没有实质内容的法律建议。
'''


import json
import os
import random

from common.paths import data_path


def sample_rewrite_data(input_file, output_file, sample_count=200):
    """
    从原始法律问答数据集中随机采样并转化为指定格式。
    """
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}")
        return

    data_list = []

    # 1. 读取原始 JSONL 数据
    print(f"正在读取原始数据: {input_file} ...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data_list.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    total_count = len(data_list)
    print(f"读取完成，共加载 {total_count} 条数据。")

    # 2. 随机采样
    if total_count < sample_count:
        sampled_data = data_list
    else:
        sampled_data = random.sample(data_list, sample_count)

    # 3. 转换格式为要求的 {"query": "...", "answer": "..."}
    processed_data = []
    for item in sampled_data:
        # 获取问题
        query = item.get("question", "")
        # 获取答案列表中的第一个元素，如果没有则为空字符串
        answers = item.get("answers", [])
        answer = answers[0] if isinstance(answers, list) and len(answers) > 0 else ""

        processed_data.append({
            "query": query,
            "answer": answer
        })

    # 4. 保存文件
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        # 使用 indent=4 使 JSON 文件易于阅读
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

    print(f"成功采样 {len(processed_data)} 条数据，并保存至: {output_file}")


if __name__ == "__main__":
    # 获取当前脚本文件所在的绝对路径
    raw_path = data_path("rewrite", "raw", "qa_corpus.json")
    output_path = data_path("rewrite", "rewrite_200_base.json")

    print(f"当前工作目录: {os.getcwd()}")  # 打印出来确认一下
    print(f"尝试读取路径: {raw_path}")

    sample_rewrite_data(str(raw_path), str(output_path), sample_count=250)
