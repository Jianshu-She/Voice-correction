# Voice Correction System / 语音纠错系统

面向儿童英语学习的音素级发音评估与纠正系统。结合微调的 WavLM 语音模型与 LLM（支持 GPT-4o / 通义千问 Qwen），提供智能化、个性化的发音反馈。

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    pronunciation_agent.py                        │
│                        (Agent 编排层)                             │
│                                                                  │
│  输入: audio.mp3 + "Hello, Peter." + user_id                    │
│       │                                                          │
│       ▼                                                          │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐          │
│  │ 步骤 1   │    │   步骤 2     │    │   步骤 3      │          │
│  │ WavLM    │───→│ 用户记忆     │───→│  LLM 反馈     │          │
│  │ 音素评估  │    │ 加载历史     │    │  + 学习计划    │          │
│  └──────────┘    └──────────────┘    └───────────────┘          │
│       │                                      │                   │
│       ▼                                      ▼                   │
│  pipeline_v2.py                    LLM API (GPT-4o / Qwen)     │
│  (WavLM 微调模型)                  中文反馈 + 学习计划            │
│       │                                      │                   │
│       ▼                                      ▼                   │
│  ┌──────────┐                      ┌───────────────┐            │
│  │ 步骤 5   │                      │   步骤 4      │            │
│  │ 返回结果  │◀─────────────────────│ 更新用户记忆   │            │
│  └──────────┘                      └───────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

![architecture_chart](./figures/architecture_chart.png "系统架构")

![flow_chart](./figures/flow_chart.png "数据流")

---

## 快速开始

### 环境要求

- Python 3.10+
- GPU（推荐 ≥22GB 显存）或 Apple Silicon Mac（MPS 后端）
- ffmpeg（音频解码）

### 安装依赖

```bash
pip install torch torchaudio transformers g2p-en huggingface_hub openpyxl openai

# ffmpeg
# macOS
brew install ffmpeg
# Ubuntu/Debian
sudo apt install ffmpeg
# conda
conda install -c conda-forge ffmpeg
```

### 模型下载

WavLM 检查点（~1.2GB）首次使用时自动从 HuggingFace 下载。手动下载：

```bash
huggingface-cli download Jianshu001/wavlm-phoneme-scorer wavlm_finetuned.pt --local-dir .
```

---

## LLM 配置

系统支持多种 LLM 后端，通过 OpenAI 兼容接口接入。

### 使用 OpenAI GPT-4o

```bash
export OPENAI_API_KEY="sk-xxx"

# 默认使用 gpt-4o
python pronunciation_agent.py --audio audio.mp3 --text "Hello, Peter." --user student_001
```

### 使用通义千问 Qwen

支持通过 DashScope API 使用 Qwen 系列模型（qwen-plus、qwen-turbo、qwen-max 等）。

**方式一：环境变量（推荐）**

```bash
export DASHSCOPE_API_KEY="sk-xxx"
# 或者
export QWEN_API_KEY="sk-xxx"

python pronunciation_agent.py --audio audio.mp3 --text "Hello, Peter." \
  --model qwen-plus --user student_001
```

**方式二：命令行参数**

```bash
python pronunciation_agent.py --audio audio.mp3 --text "Hello, Peter." \
  --model qwen-plus --api-key sk-xxx --user student_001
```

**方式三：自定义 API 地址**

```bash
python pronunciation_agent.py --audio audio.mp3 --text "Hello, Peter." \
  --model qwen-turbo \
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --api-key sk-xxx
```

> **自动检测逻辑：** 当 `--model` 以 `qwen` 开头时，系统自动使用 DashScope 的 API 地址和 `DASHSCOPE_API_KEY` / `QWEN_API_KEY` 环境变量。其他模型名默认走 OpenAI。

### 使用其他 OpenAI 兼容 API

任何兼容 OpenAI 接口的服务都可以通过 `--base-url` 接入：

```bash
# 例如本地 Ollama
python pronunciation_agent.py --audio audio.mp3 --text "Hello" \
  --model llama3 --base-url http://localhost:11434/v1 --api-key ollama

# 例如 DeepSeek
export OPENAI_API_KEY="sk-xxx"
python pronunciation_agent.py --audio audio.mp3 --text "Hello" \
  --model deepseek-chat --base-url https://api.deepseek.com/v1
```

---

## 使用方式

### 1. 命令行（CLI）

```bash
# 完整模式：评估 + LLM 反馈 + 用户记忆
python pronunciation_agent.py --audio audio.mp3 --text "Hello, Peter." --user student_001

# 输出 JSON
python pronunciation_agent.py --audio audio.mp3 --text "Hello, Peter." --user student_001 --json

# 仅评估模式（无 LLM，更快）
python pipeline_v2.py --audio audio.mp3 --text "Hello, Peter."

# 指定设备
python pronunciation_agent.py --audio audio.mp3 --text "Hello" --device mps    # Apple Silicon
python pronunciation_agent.py --audio audio.mp3 --text "Hello" --device cuda:0  # NVIDIA GPU
python pronunciation_agent.py --audio audio.mp3 --text "Hello" --device cpu     # CPU（慢）
```

### 2. Python API

```python
from pronunciation_agent import PronunciationAgent

# 使用 GPT-4o
agent = PronunciationAgent()

# 使用 Qwen
agent = PronunciationAgent(model="qwen-plus")

# 自定义 API
agent = PronunciationAgent(
    model="qwen-max",
    api_key="sk-xxx",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

result = agent.run(audio="audio.mp3", text="Hello, Peter.", user_id="student_001")
```

### 3. FastAPI 服务

```bash
# 安装额外依赖
pip install fastapi uvicorn python-multipart

# 启动服务（默认 GPT-4o）
python pronunciation_agent.py --serve --port 8000

# 启动服务（使用 Qwen）
export DASHSCOPE_API_KEY="sk-xxx"
export LLM_MODEL="qwen-plus"
python pronunciation_agent.py --serve --port 8000
```

**API 接口：**

| 接口 | 方法 | 说明 |
|------|------|------|
| `/assess` | POST | 上传音频 + 文本 + user_id，返回完整评估和反馈 |
| `/user/{user_id}` | GET | 查询用户学习历史和统计 |
| `/health` | GET | 健康检查 |

**调用示例：**

```bash
# 评估发音
curl -X POST http://localhost:8000/assess \
  -F "audio=@audio.mp3" \
  -F "text=Hello, Peter." \
  -F "user_id=student_001"

# 查询用户进度
curl http://localhost:8000/user/student_001
```

### 4. 仅评估 API（pipeline_v2.py）

```python
from pipeline_v2 import PronunciationAssessorV2

# 自动从 HuggingFace 下载模型
assessor = PronunciationAssessorV2.from_pretrained()
result = assessor.assess("audio.mp3", "Hello, Peter.")
```

---

## 第一层：发音评估模型（pipeline_v2.py）

核心引擎：将音频 + 参考文本转换为音素级评估结果。

### 处理流程

```
"Hello, Peter."  ──→  G2P (g2p_en)  ──→  /hh ah l ow p iy t er/
                                                    │
audio.mp3  ──→  wav2vec2-xlsr-53 (CTC, 冻结)        │
                    │                                │
                    ▼                                ▼
              帧级音素概率                    预期音素序列
              (T 帧 × 392 IPA)              (ARPAbet → IPA 映射)
                    │                                │
                    └────────── Viterbi 对齐 ─────────┘
                                    │
                                    ▼
                    每个音素的音频帧段
                    /hh/ → 帧 0-15, /ah/ → 帧 16-30, ...
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              WavLM-Large      GOP 分数         帧数
              (微调)           log P(目标)      n_frames
              1024维           - log P(最佳
              隐藏状态          其他)
                    │               │               │
                    ▼               ▼               ▼
              ┌─────────────────────────────────────┐
              │  拼接: hidden(1024) + phone_emb(32)  │
              │        + GOP(1) + n_frames(1)        │
              │              = 1058 维                │
              └─────────────────┬───────────────────┘
                                │
                                ▼
                    MLP (1058 → 512 → 512 → 256)
                    BatchNorm + GELU + Dropout
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
              score_head               pherr_head
              (256 → 64 → 1)          (256 → 64 → 1)
                    │                       │
                    ▼                       ▼
              音素得分 0-100           错误概率 0-1
              (回归)                  (分类, sigmoid)
```

### 使用的模型

| 模型 | 作用 | 状态 |
|------|------|------|
| `wav2vec2-xlsr-53-espeak-cv-ft` | 392 IPA 音素 CTC 模型，提供帧级音素概率用于 Viterbi 对齐和 GOP 计算 | 冻结 |
| `WavLM-Large` (microsoft/wavlm-large) | 语音特征提取骨干网络，输出 1024 维隐藏状态 | **微调（顶部 6 层）** |
| MLP 评分头 | 从拼接特征预测音素得分和错误概率 | 全训练 |

### Viterbi 强制对齐

使用动态规划在 CTC 输出上对齐预期音素序列到音频帧：

```
输入:  CTC 输出 (T 帧 × 392 音素) + 预期序列 [hh, ah, l, ow]
              ↓
扩展: [blank, hh, blank, ah, blank, l, blank, ow, blank]
              ↓
DP:   dp[t][s] = 在帧 t、状态 s 的最佳路径概率
      转移: 停留 / 前进 1 / 跳过 blank 前进 2
              ↓
回溯:  回溯获得每帧的音素分配
              ↓
输出: /hh/ → frames[0..15], /ah/ → frames[16..30], ...
```

### GOP（发音优度）

对每个对齐的音素段：

```
GOP = mean(log P(目标 | 帧)) - mean(max log P(其他 | 帧))

GOP > 0  →  模型确认这是预期音素（正确）
GOP < 0  →  模型认为其他音素更匹配（可能错误）
```

GOP 作为 MLP 的输入特征之一，但最终错误判断由学习到的 pherr_head 做出（而非简单的 GOP 阈值）。

---

## 第二层：用户记忆（UserMemory）

基于 JSON 文件的持久化存储，跟踪每个用户跨会话的学习进度。

```
user_memory/
└── student_001.json
    {
      "user_id": "student_001",
      "sessions": [
        {
          "timestamp": "2026-03-19T...",
          "text": "Hello, Lingling.",
          "overall_score": 80.1,
          "error_phonemes": [
            {"phone": "ah", "word": "Lingling", "score": 60.5},
            {"phone": "ng", "word": "Lingling", "score": 32.3}
          ]
        }
      ],
      "weak_phonemes": {
        "ah": {"count": 5, "errors": 3, "total_score": 320},
        "ng": {"count": 3, "errors": 2, "total_score": 150}
      },
      "overall_stats": {
        "total_sessions": 10,
        "total_phonemes": 120,
        "total_errors": 15,
        "avg_score": 78.5
      },
      "current_plan": {
        "focus_phonemes": ["ah", "ng"],
        "practice_words": ["sing", "long", "song"]
      }
    }
```

---

## 第三层：LLM Agent

Agent 每次会话调用 LLM 两次：

### 调用 1：生成反馈

```
系统提示: 你是一位面向中国儿童的英语发音教练...

用户:
  评估结果: "Hello, Lingling." 得分=80.1
  错误: /ah/ (得分=60.5), /ng/ (得分=32.3)
  用户历史: 3 次会话，薄弱音素: /ah/ (50% 错误率), /ng/ (67% 错误率)

LLM → "你的Hello发得很好！Lingling里的/ah/需要张大嘴巴..."
```

### 调用 2：生成结构化计划

```
LLM → {
  "focus_phonemes": ["ah", "ng"],
  "practice_words": ["sing", "long", "song"],
  "tips": ["张大嘴巴练习/ah/...", "舌头抵上牙龈练习/ng/..."]
}
```

---

## 训练

### 数据

| 数据集 | 样本数 | 音素数 | 来源 |
|--------|--------|--------|------|
| eval_log.xlsx | 1,000 句子 | 12,438 | 儿童英语朗读 |
| word_eval_log.xlsx | 10,601 单词 | 41,488 | 儿童英语单词 |
| **合计** | **11,601** | **53,926** | 专业标注（pherr + score）|

### 训练配置

```
骨干网络: WavLM-Large (316M 参数)
  - 冻结底部 18 层，微调顶部 6 层（76.6M 可训练参数）
  - 学习率: 1e-5

MLP 头: 1058 → 512 → 512 → 256 → (score, pherr)
  - 学习率: 5e-4

优化器: AdamW, weight_decay=1e-3
调度器: CosineAnnealing
梯度累积: 4 步, 梯度裁剪: max_norm=1.0
早停: patience=8, 在第 24 轮停止

损失: MSE/100 (score) + BCE with pos_weight=5.2 (pherr)
     pos_weight 补偿类别不平衡（仅 16% 为错误）

训练/验证/测试划分: 40K / 5K / 8K 音素（按音频文件划分）
```

### 性能对比

| 方法 | 数据量 | AUC | F1 | Precision | Recall | Pearson | MAE |
|------|--------|-----|-----|-----------|--------|---------|-----|
| GOP 阈值 (v1.0) | 1K | 0.738 | 0.476 | 0.379 | 0.638 | 0.372 | 27.44 |
| E2E MLP 冻结骨干 (v2) | 1K | 0.814 | 0.565 | 0.500 | 0.650 | 0.528 | 22.57 |
| 音素对比法 | 1K | 0.691 | 0.492 | 0.379 | 0.703 | N/A | N/A |
| wav2vec2-large 微调 | 11K | 0.844 | 0.548 | 0.527 | 0.571 | 0.574 | 17.72 |
| **WavLM-Large 微调** | **11K** | **0.870** | **0.595** | **0.592** | **0.598** | **0.645** | **16.47** |

---

## 项目结构

```
Voice-correction/
├── pronunciation_agent.py   # Agent: 评估 + LLM 反馈 + 用户记忆
├── pipeline_v2.py           # 核心模型: WavLM 微调评估 (v2.0)
├── pipeline.py              # 基线: GOP 阈值评估 (v1.0)
├── finetune_wavlm.py        # 训练脚本: WavLM 骨干微调
├── finetune_backbone.py     # 训练脚本: wav2vec2 骨干微调
├── phoneme_compare.py       # 实验: 音素对比方法
├── e2e_v2_train.py          # 训练脚本: E2E MLP + 冻结骨干
├── e2e_phoneme_scorer.py    # 训练脚本: E2E MLP v1
├── wavlm_finetuned.pt       # 模型检查点（自动从 HF 下载）
├── eval_log.xlsx            # 标注数据: 1000 句子
├── word_eval_log.xlsx       # 标注数据: 10601 单词
├── audio_files/             # 句子录音 (1000 MP3)
├── words_audio_files/       # 单词录音 (10655 MP3)
├── figures/                 # 架构图
├── user_memory/             # 用户进度数据（gitignored）
├── openclaw-skill/          # OpenClaw 技能定义
│   └── SKILL.md
└── README.md
```

---

## OpenClaw 技能集成

本系统可作为 [OpenClaw](https://docs.openclaw.ai) 技能使用。

```bash
# 安装到 OpenClaw
cp -r openclaw-skill ~/.openclaw/workspace/skills/pronunciation-correction
```

---

## 输出字段说明

| 字段 | 范围 | 说明 |
|------|------|------|
| `overall_score` | 0-100 | 总体得分，越高越好 |
| `pherr_prob` | 0-1 | 音素错误概率，>0.70 视为错误 |
| `score`（每音素）| 0-100 | 模型预测的发音质量分 |
| `GOP` | 负数~正数 | 发音优度，负值表示可能发错 |
| `weak_phonemes` | - | 用户历史上高错误率的薄弱音素 |

---

## HuggingFace 模型

[Jianshu001/wavlm-phoneme-scorer](https://huggingface.co/Jianshu001/wavlm-phoneme-scorer)

## 已知限制

- G2P 生成的音素序列可能与标注不完全匹配
- 不能很好处理数字（"8 o'clock"）和非英语名字
- v2.0 推理需要 GPU（~22GB 显存），v1.0 可在 CPU 上运行；Apple Silicon Mac 可使用 MPS 后端
- LLM 反馈质量取决于所用模型和提示词工程
