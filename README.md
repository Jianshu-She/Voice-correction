# Voice Correction Pipeline v1.0

Phoneme-level English pronunciation assessment system. Given a reference text and an audio recording, identifies which phonemes are pronounced correctly and which have errors.

## How It Works

```
Reference Text ──→ G2P (grapheme-to-phoneme) ──→ Expected phoneme sequence
                                                          │
Audio (MP3) ──→ wav2vec2 phoneme model ──→ Frame-level phoneme posteriors
                                                          │
                                              Viterbi Forced Alignment
                                                          │
                                              GOP (Goodness of Pronunciation)
                                                          │
                                              Per-phoneme error detection
```

1. **G2P**: Converts reference text to ARPAbet phoneme sequence using CMU dictionary
2. **Acoustic Model**: `wav2vec2-xlsr-53-espeak-cv-ft` produces frame-level IPA phoneme probabilities
3. **Forced Alignment**: Viterbi algorithm aligns each expected phoneme to audio frames
4. **GOP Scoring**: For each phoneme, computes `GOP = log P(target) - log P(best_other)`. Negative GOP means the model thinks another phoneme fits better → likely mispronunciation
5. **Error Detection**: Phonemes with GOP below threshold (-2.5) are flagged as errors

## Setup

```bash
# Python 3.10+
pip install torch torchaudio transformers g2p-en huggingface_hub openpyxl

# ffmpeg is required for MP3 decoding
conda install -c conda-forge ffmpeg
# or: apt install ffmpeg
```

The wav2vec2 model (~1.2GB) downloads automatically on first run.

## Usage

### Single File

```bash
python pipeline.py --audio path/to/audio.mp3 --text "Hello, Peter."
```

Output:
```
============================================================
Text: "Hello, Peter."
Overall Score: 82.3/100  (errors: 1/8)
============================================================

  ✓ Hello            score= 87.0  errors=0/4
      /hh  /  GOP= +4.90
      /ah  /  GOP= +0.60
      /l   /  GOP= +6.20
      /ow  /  GOP= -0.40

  ✗ Peter            score= 77.5  errors=1/4
      /p   /  GOP= +5.40
      /iy  /  GOP= +3.70
      /t   /  GOP= +0.50
      /er  /  GOP= -3.10 ← ERROR
```

### JSON Output

```bash
python pipeline.py --audio audio.mp3 --text "Hello, Peter." --json
```

### Batch Mode

Process all entries from `eval_log.xlsx`:

```bash
# Process first 20 samples, save results to JSON
python pipeline.py --batch --input eval_log.xlsx --audio-dir audio_files/ --output results.json --limit 20

# Process all samples and evaluate against ground truth
python pipeline.py --batch --input eval_log.xlsx --audio-dir audio_files/ --evaluate
```

### Python API

```python
from pipeline import PronunciationAssessor

assessor = PronunciationAssessor()
result = assessor.assess("audio.mp3", "Hello, Peter.")

# result structure:
# {
#   "text": "Hello, Peter.",
#   "overall_score": 82.3,
#   "n_phonemes": 8,
#   "n_errors": 1,
#   "error_rate": 12.5,
#   "words": [
#     {
#       "word": "Hello",
#       "score": 87.0,
#       "has_error": false,
#       "phonemes": [
#         {"phone": "hh", "gop": 4.9, "error": false},
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

### Options

| Flag | Description |
|------|-------------|
| `--audio` | Path to audio file (single mode) |
| `--text` | Reference text (single mode) |
| `--threshold` | GOP threshold for error detection (default: -2.5) |
| `--batch` | Enable batch mode |
| `--input` | Input xlsx file (batch mode) |
| `--audio-dir` | Audio files directory (batch mode) |
| `--output` | Output JSON file |
| `--limit` | Max samples to process (0=all) |
| `--evaluate` | Compare with ground truth scores |
| `--json` | Output JSON instead of pretty print |
| `--device` | Device: `cuda:0`, `cpu` |

## Data

- `audio_files/` — 1000 MP3 recordings of English learners (children)
- `eval_log.xlsx` — Ground truth from professional pronunciation assessment engine, with columns:
  - `file_name`: audio filename
  - `content`: reference text
  - `json_content`: detailed scoring (overall accuracy, word-level scores, phoneme-level scores with error flags)

## Current Performance (v1.0)

Evaluated on 1000 samples against ground truth:

| Metric | Value |
|--------|-------|
| Phoneme error detection AUC-ROC | 0.738 |
| Best F1 (threshold=-2.5) | 0.476 |
| Precision @ best F1 | 0.379 |
| Recall @ best F1 | 0.638 |
| GOP vs GT phone score Pearson | 0.372 |

### Known Limitations

- The acoustic model is trained on adult speech; performance on children's speech is degraded
- G2P-generated phoneme sequences may not perfectly match the ground truth phoneme sequences
- Simple threshold-based error detection; no per-phoneme calibration
- Does not handle numbers ("8 o'clock") or non-English names well

## Project Structure

```
Voice-correction/
├── pipeline.py              # Main pipeline (v1.0) — use this
├── eval_log.xlsx            # Ground truth data
├── audio_files/             # 1000 MP3 recordings
├── phoneme_gop_eval.py      # Evaluation script (full analysis)
├── gop_pipeline.py          # Earlier prototype (v1, character-level)
├── gop_pipeline_v2.py       # Earlier prototype (v2, feature engineering)
├── gop_pipeline_v3.py       # Earlier prototype (v3, with Whisper ASR)
├── phoneme_gop.py           # Earlier prototype (phoneme-level)
└── README.md
```

## Next Steps

- Fine-tune wav2vec2/HuBERT on children's speech data
- Use ground truth phoneme sequences for alignment instead of G2P
- Train a classifier/regressor on wav2vec2 hidden states for per-phoneme scoring
- Per-phoneme threshold calibration
