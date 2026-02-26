# 实验一：意图识别与 Query 改写

本实验基于 Qwen2.5 系列大语言模型，实现并评估法律问答场景下的 **意图识别** 与 **Query 改写** 模块，并对比了 **双模型方案** 与 **单模型融合方案** 的效果，为后续 RAG 检索与 Agent 系统奠定基础。

---

## 一、实验目标

本实验主要完成以下任务：

1. 对用户输入的 Query 进行意图识别，分类为：
   - 法律类
   - 违规类
   - 闲聊类
2. 对识别为 **法律类** 的 Query 进行规范化改写，生成适合检索的标准法律查询
3. 对比两种系统设计方案：
   - 双模型方案（意图识别 + Query 改写）
   - 单模型融合方案（一次推理同时完成两项任务）

---

## 二、数据集

### 1. 意图识别数据集

- 数据路径：`data/experiments/exp1/intent/raw/intent_2k.json`
- 数据规模：2000 条
- 数据来源：基于人工设计与大模型生成的中文用户问题

数据格式如下：

```json
{
  "问题": "劳动合同到期公司不续签有赔偿吗？",
  "类型": "法律类"
}
```

### 2. Query 改写测试数据集

- 数据路径：`data/experiments/exp1/rewrite/qa_testset_500.json`
- 数据规模：500 条
- 数据构成：
  - 200 条来自已有法律问答数据
  - 300 条基于法条内容由大模型生成的合成数据（以 self-instruct 方式为主）

该数据集用于评估模型在法律场景下的 **Query 改写能力**。  
模型仅接收原始用户问题作为输入，不使用参考答案。

Query 改写的目标是将口语化、冗余的提问转换为**简洁、规范、适合检索的法律查询表达**，例如：

- 原始问题：  
  > 我朋友欠我钱一直不还，我该怎么办？  
- 改写后：  
  > 民间借贷纠纷追讨欠款途径  

---

## 三、实验方法

### 1. 双模型方案

双模型方案将任务拆分为两个独立模块：

1. **意图识别模型**：判断用户问题属于法律类、违规类或闲聊类  
2. **Query 改写模型**：仅对识别为法律类的问题进行规范化改写  

该方案结构清晰，但需要进行两次模型推理，推理成本和系统复杂度较高。

---

### 2. 单模型融合方案

单模型融合方案使用同一个模型，通过统一 Prompt 同时完成：

- 意图识别
- Query 改写

模型输出为结构化 JSON 格式：

```json
{
  "intent": "",
  "rewrite_query": ""
}
```
## 四、模型设置

本实验选用 Qwen2.5 系列模型进行对比实验，具体包括：

- Qwen2.5-0.5B
- Qwen2.5-1.5B
- Qwen2.5-3B
- Qwen2.5-7B

在意图识别与 Query 改写任务中，所有模型均采用相同的数据集、Prompt 设计和评估指标，以保证实验结果的可比性。

---

## 五、评估指标

本实验主要评估 **意图识别任务** 的效果，采用如下指标：

- **Accuracy**：整体分类准确率  
- **Macro Precision**：各类别 Precision 的宏平均  
- **Macro Recall**：各类别 Recall 的宏平均  
- **Macro F1**：各类别 F1 的宏平均  
- **Per-class Precision / Recall / F1**：各类别的详细指标  

Query 改写任务不进行自动化打分，仅对改写结果进行人工观察与定性分析。

---

## 六、实验结果

### 1. 双模型方案结果

双模型方案下，不同模型规模的意图识别评估结果保存在：

```text
output/experiments/exp1/intent/
├── metrics_qwen2.5_0.5b.json
├── metrics_qwen2.5_1.5b.json
├── metrics_qwen2.5_3b.json
└── metrics_qwen2.5_7b.json
```

## 目录结构与扩展方式

为支持后续实验（exp2/exp3 等），目录统一为实验维度管理，默认使用 `exp1`：

```text
data/
└── experiments/
    ├── exp1/
    └── exp2/
prompts/
└── experiments/
    ├── exp1/
    └── exp2/
output/
└── experiments/
    ├── exp1/
    └── exp2/
src/
```

### 运行路径与配置约定

- 默认实验目录（逻辑实验名）：`exp1`
- 指定实验目录：
  - `LLM_EXPERIMENT=exp2 python src/intent/intent_infer.py`
  - 或直接指定路径：`LLM_DATA_DIR=/abs/path/to/data/experiments/exp2 LLM_PROMPTS_DIR=/abs/path/to/prompts/experiments/exp2 LLM_RESULTS_DIR=/abs/path/to/output/experiments/exp2 python src/intent/intent_infer.py`

所有脚本通过统一的路径配置读取 `data / prompts / output`，因此在不同设备或云端运行时只需要切换环境变量即可。


---

## 七、工程化架构迁移（实验架构 -> 工程架构）

当前项目已从按 `expX` 脚本分散组织，迁移为按能力分层组织：

```text
legal_llm_project/
├── configs/
├── data/
├── src/
│   ├── intent/
│   ├── rag/
│   ├── agent/
│   ├── llm/
│   ├── evaluation/
│   └── common/
├── scripts/
├── logs/
├── prompts/
├── output/
└── README.md
```

### 模块说明

- `src/intent/`：意图识别与 Query 改写相关脚本。
- `src/rag/`：语料构建、检索、索引与 RAG 推理/评估。
- `src/agent/`：多模块编排与基线推理。
- `src/llm/`：SFT / DPO / 数据集构建等训练与数据流水线。
- `src/evaluation/`：独立评测脚本。
- `scripts/`：通用工具脚本（如数据格式转换、训练数据预处理）。
- `configs/`：集中管理后续环境、模型、检索等配置（已预留）。
- `data/`、`prompts/`、`output/`、`logs/`：工程运行期输入与产物目录。

### 迁移原则

1. **按类别沉淀到顶层目录**：实验数据、Prompt、结果分别收敛到 `data/experiments`、`prompts/experiments`、`output/experiments`。
2. **代码按能力归类**：避免继续按 `exp1/exp2/exp3` 划分导致的重复与耦合。
3. **渐进式收敛**：先完成目录重构，再逐步统一 CLI 入口、配置文件与公共组件。

### 下一步建议

- 在 `configs/` 增加：`model.yaml`、`rag.yaml`、`runtime.yaml`。
- 为 `src/*` 每个子模块补充统一 `main()` 入口与参数风格。
- 新增 `scripts/run_*.sh` 或 `Makefile` 作为标准运行入口。
