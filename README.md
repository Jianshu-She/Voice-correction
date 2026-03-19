# Voice Correction Pipeline

Phoneme-level English pronunciation assessment system. Given a reference text and an audio recording, identifies which phonemes are pronounced correctly and which have errors.

## Pipeline v2.0 (WavLM Fine-tuned) — Recommended

Uses a **fine-tuned WavLM-Large** backbone + MLP scoring head, trained on 11K children's speech samples.

Model weights hosted on HuggingFace: [Jianshu001/wavlm-phoneme-scorer](https://huggingface.co/Jianshu001/wavlm-phoneme-scorer)

```
Reference Text ──→ G2P ──→ Expected phoneme sequence
                                    │
Audio ──→ CTC model ──→ Viterbi Forced Alignment ──→ Frame segments
  │                                                       │
  └──→ WavLM-Large (fine-tuned) ──→ Hidden states ──→ Pool per segment
                                                          │
                                                    + phone embedding
                                                    + GOP score
                                                          │
                                                    MLP scorer head
                                                    ├── phoneme score (0-100)
                                                    └── error probability (0-1)
```

## Pipeline v1.0 (GOP Baseline)

Uses GOP (Goodness of Pronunciation) scoring with a frozen `wav2vec2-xlsr-53-espeak-cv-ft` model.

1. **G2P**: Converts reference text to ARPAbet phoneme sequence using CMU dictionary
2. **Acoustic Model**: `wav2vec2-xlsr-53-espeak-cv-ft` produces frame-level IPA phoneme probabilities
3. **Forced Alignment**: Viterbi algorithm aligns each expected phoneme to audio frames
4. **GOP Scoring**: For each phoneme, computes `GOP = log P(target) - log P(best_other)`. Negative GOP means the model thinks another phoneme fits better → likely mispronunciation
5. **Error Detection**: Phonemes with GOP below threshold (-2.5) are flagged as errors

fig1. System Architecture: A Multi-tiered Approach
The architecture follows a Bottom-to-Top (BT) layered design, ensuring that high-level business logic is decoupled from underlying AI capabilities and data storage.

![architecture_chart](./figures/architecture_chart.png "系统架构")

Layer 1: Foundation & Data Base (The Ground)
User/Memory Database: Stores persistent user profiles, historical performance data, and the current "State" of the learning journey.

Training/Experimental Database: A dedicated repository for raw audio and evaluation logs used for continuous model fine-tuning and iterative testing.

Learning Resource Library: A static repository of educational assets (videos, courseware, and scripts) indexed for retrieval.

Layer 2: Core Technical Base (The AI Engine)
Pronunciation Inference Model: A fine-tuned LLM/Speech model that processes raw audio to detect phoneme-level errors.

Evaluation Engine: Converts model output into structured Evaluation Logs, identifying specific weaknesses (e.g., specific words or intonation issues).

Logic Planner: Acts as a bridge between raw data and pedagogy. It transforms "Logs" into actionable "Learning Plans" based on pre-defined teaching templates.

Layer 3: Intelligent Orchestration (The Agent Hub)
Agent State Machine: Inspired by the LangGraph framework, this is the system's "Brain." It manages the logic flow, deciding whether a user needs more practice on an old task or is ready for new content.
Memory Module: Handles Context Injection. It retrieves past performance to inform current decisions, ensuring the Agent "remembers" that a user struggled with a specific word yesterday.

Scheduler: Enforces pedagogical constraints, such as daily workload limits and curriculum boundaries.
Layer 4: Application Layer (The Surface)
Oral Dialogue Interface: The primary interaction point for real-time speech practice.
Review & Feedback: A module that provides granular explanations of errors.

Resource Recommendation: An upper-level service that fetches relevant videos or exercises from the library based on the Agent’s generated plan.



fig2. Data Flow & Agent State Logic
The system operates on a Stateful Cyclic Workflow, moving beyond simple linear processing to a sophisticated loop of "Assess -> Plan -> Execute -> Remember."

![flow_chart](./figures/flow_chart.png "系统架构")

Phase 1: Input & Assessment (Day N)
User Submission: The user submits audio via the Application Layer.
AI Evaluation: The Technical Base processes the audio, generating a detailed Evaluation Log.
State Update: The Log is fed into the Agent State, updating the user's current mastery level.

Phase 2: Logical Planning & Memory Persistence
Dynamic Planning: The Planner analyzes the Log (e.g., "Apple" mispronounced) and maps it to specific knowledge points (A, B, or C).
Plan Update: A new learning plan is generated. This plan is not just stored in a DB but is committed to the Agent's Memory, influencing the next "turn" of the conversation.
Persistence: The State is saved via a Checkpointer, allowing the system to resume exactly where the user left off.

Phase 3: Feedback & Contextual Execution (Day N+1)
Context Retrieval: Upon the user's return, the Agent retrieves the previous state from Memory.
Output Generation: The Agent executes a hybrid strategy: "Execute Old Plan (Review) + Deploy New Plan (Advance)."
Resource Delivery: The Application Layer presents specific recommended resources based on the refined plan.




## Setup

```bash
# Python 3.10+
pip install torch torchaudio transformers g2p-en huggingface_hub openpyxl

# ffmpeg is required for MP3 decoding
conda install -c conda-forge ffmpeg
# or: apt install ffmpeg
```

Models download automatically on first run from HuggingFace.

## Usage (v2.0 — Recommended)

### Download Model

The model checkpoint (~1.2GB) is automatically downloaded from HuggingFace on first use. You can also download it manually:

```bash
# Option 1: Auto-download (happens on first run)
python pipeline_v2.py --audio audio.mp3 --text "Hello, Peter."

# Option 2: Manual download via huggingface-cli
huggingface-cli download Jianshu001/wavlm-phoneme-scorer wavlm_finetuned.pt --local-dir .

# Option 3: Manual download via Python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="Jianshu001/wavlm-phoneme-scorer", filename="wavlm_finetuned.pt", local_dir=".")
```

### Single File

```bash
python pipeline_v2.py --audio path/to/audio.mp3 --text "Hello, Peter."
```

Output:
```
============================================================
Text: "Hello, Peter."
Overall Score: 85.2/100  (errors: 0/8)
============================================================

  ✓ Hello            score= 87.7  errors=0/4
      /hh  /  score= 98.6  GOP= -0.97  pherr=0.05
      /ah  /  score= 73.1  GOP= -7.40  pherr=0.43
      /l   /  score= 88.6  GOP= +4.00  pherr=0.29
      /ow  /  score= 90.7  GOP= -6.05  pherr=0.13

  ✓ Peter            score= 82.6  errors=0/4
      /p   /  score= 95.7  GOP= +5.40  pherr=0.08
      /iy  /  score= 90.2  GOP= +3.70  pherr=0.12
      /t   /  score= 72.5  GOP= +0.50  pherr=0.55
      /er  /  score= 71.8  GOP= -1.40  pherr=0.61
```

### Python API

```python
from pipeline_v2 import PronunciationAssessorV2

# Auto-download from HuggingFace
assessor = PronunciationAssessorV2.from_pretrained()

# Or with local checkpoint
assessor = PronunciationAssessorV2(checkpoint_path="wavlm_finetuned.pt")

result = assessor.assess("audio.mp3", "Hello, Peter.")

# result structure:
# {
#   "text": "Hello, Peter.",
#   "overall_score": 85.2,
#   "n_phonemes": 8,
#   "n_errors": 0,
#   "error_rate": 0.0,
#   "words": [
#     {
#       "word": "Hello",
#       "score": 87.7,
#       "has_error": false,
#       "phonemes": [
#         {"phone": "hh", "score": 98.6, "gop": -0.97, "pherr_prob": 0.05, "error": false},
#         ...
#       ]
#     },
#     ...
#   ]
# }

# Batch processing
results = assessor.assess_batch([
    ("audio1.mp3", "Hello."),
    ("audio2.mp3", "Good morning."),
])
```

### Batch Mode

```bash
# Process samples from xlsx, save results to JSON
python pipeline_v2.py --batch --input eval_log.xlsx --audio-dir audio_files/ --output results.json --limit 20

# Evaluate against ground truth
python pipeline_v2.py --batch --input eval_log.xlsx --audio-dir audio_files/ --evaluate
```

### Options (v2.0)

| Flag | Description |
|------|-------------|
| `--audio` | Path to audio file (single mode) |
| `--text` | Reference text (single mode) |
| `--checkpoint` | Path to model checkpoint (default: auto-download from HuggingFace) |
| `--threshold` | Pherr probability threshold for error detection (default: 0.70) |
| `--batch` | Enable batch mode |
| `--input` | Input xlsx file (batch mode) |
| `--audio-dir` | Audio files directory (batch mode) |
| `--output` | Output JSON file |
| `--limit` | Max samples to process (0=all) |
| `--evaluate` | Compare with ground truth scores |
| `--json` | Output JSON instead of pretty print |
| `--device` | Device: `cuda:0`, `cpu` |

### v1.0 Usage (GOP Baseline)

```bash
python pipeline.py --audio audio.mp3 --text "Hello, Peter."
```

## Data

- `audio_files/` — 1000 MP3 recordings of English learners (children)
- `eval_log.xlsx` — Ground truth from professional pronunciation assessment engine, with columns:
  - `file_name`: audio filename
  - `content`: reference text
  - `json_content`: detailed scoring (overall accuracy, word-level scores, phoneme-level scores with error flags)

## Performance

| Metric | v1.0 (GOP) | v2.0 (WavLM) |
|--------|-----------|--------------|
| Phoneme error AUC-ROC | 0.738 | **0.870** |
| Phoneme error F1 | 0.476 | **0.595** |
| Phoneme error Precision | 0.379 | **0.592** |
| Phoneme error Recall | 0.638 | 0.598 |
| Phone score Pearson | 0.372 | **0.645** |
| Phone score MAE | 27.44 | **16.47** |

v2.0 evaluated on test set: 8062 phonemes from 1727 audio files (children's speech).

### Known Limitations

- G2P-generated phoneme sequences may not perfectly match the ground truth phoneme sequences
- Does not handle numbers ("8 o'clock") or non-English names well
- v2.0 requires GPU (~22GB VRAM) for inference; v1.0 can run on CPU

## Project Structure

```
Voice-correction/
├── pipeline_v2.py           # Main pipeline v2.0 (WavLM fine-tuned) — recommended
├── pipeline.py              # Pipeline v1.0 (GOP baseline)
├── finetune_wavlm.py        # WavLM backbone fine-tuning script
├── finetune_backbone.py     # wav2vec2 backbone fine-tuning script
├── wavlm_finetuned.pt       # WavLM model checkpoint (or auto-downloaded from HF)
├── eval_log.xlsx            # Ground truth data (1000 sentences)
├── word_eval_log.xlsx       # Ground truth data (10601 words)
├── audio_files/             # 1000 sentence-level MP3 recordings
├── words_audio_files/       # 10655 word-level MP3 recordings
└── README.md
```

## Model on HuggingFace

The fine-tuned WavLM model is hosted at: [Jianshu001/wavlm-phoneme-scorer](https://huggingface.co/Jianshu001/wavlm-phoneme-scorer)
