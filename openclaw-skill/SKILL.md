---
name: pronunciation-correction
description: Assess English pronunciation at the phoneme level and provide AI-powered feedback with personalized learning plans. Uses a fine-tuned WavLM model for assessment and GPT-4o for intelligent coaching.
metadata: {"openclaw": {"emoji": "🗣️", "requires": {"bins": ["python3", "ffmpeg"], "env": ["OPENAI_API_KEY"]}, "install": [{"id": "pip", "kind": "command", "command": "pip install torch torchaudio transformers g2p-en huggingface_hub openai", "label": "Install Python dependencies"}, {"id": "repo", "kind": "command", "command": "git clone https://github.com/Jianshu-She/Voice-correction.git /tmp/voice-correction && cd /tmp/voice-correction && git checkout feature/pronunciation-agent", "label": "Clone Voice Correction repo"}]}}
user-invocable: true
---

# Pronunciation Correction Skill

You are a pronunciation correction agent. When the user asks you to assess, evaluate, check, or correct their English pronunciation, use this skill.

## When to activate

- User asks to check/assess/evaluate pronunciation
- User provides an audio file and reference text for pronunciation checking
- User asks about their pronunciation progress or learning plan
- User asks "how is my pronunciation" or similar

## How to use

### Step 1: Locate the tool

The pronunciation agent is at `/tmp/voice-correction/pronunciation_agent.py`. If it doesn't exist, clone the repo first:

```bash
git clone https://github.com/Jianshu-She/Voice-correction.git /tmp/voice-correction
cd /tmp/voice-correction && git checkout feature/pronunciation-agent
```

### Step 2: Assess pronunciation

When the user provides an audio file and reference text, run:

```bash
cd /tmp/voice-correction && python3 pronunciation_agent.py --audio <AUDIO_PATH> --text "<REFERENCE_TEXT>" --user <USER_ID> --json
```

- `<AUDIO_PATH>`: Path to the user's audio recording (MP3, WAV, etc.)
- `<REFERENCE_TEXT>`: The text the user was supposed to read
- `<USER_ID>`: A unique identifier for the user (use "default" if unknown)

The `--json` flag outputs structured JSON with:
- `evaluation`: Phoneme-level scores and error flags
- `feedback`: AI-generated feedback in Chinese
- `plan`: Structured learning plan with focus phonemes and practice words
- `user_stats`: Cumulative user statistics
- `weak_phonemes`: Historically weak phonemes

### Step 3: Assessment only (without LLM feedback)

If only phoneme assessment is needed (faster, no OpenAI API call):

```bash
cd /tmp/voice-correction && python3 pipeline_v2.py --audio <AUDIO_PATH> --text "<REFERENCE_TEXT>" --json
```

### Step 4: Check user progress

To view a user's learning history:

```bash
cd /tmp/voice-correction && python3 -c "
from pronunciation_agent import UserMemory
m = UserMemory('<USER_ID>')
import json
print(json.dumps({
    'stats': m.data['overall_stats'],
    'weak_phonemes': m.get_weak_phonemes(),
    'recent_sessions': m.get_recent_sessions(),
    'current_plan': m.data.get('current_plan')
}, indent=2, ensure_ascii=False))
"
```

### Step 5: Start API server

If the user wants a persistent API endpoint:

```bash
cd /tmp/voice-correction && pip install fastapi uvicorn python-multipart -q
python3 pronunciation_agent.py --serve --port 8000
```

Endpoints:
- `POST /assess` — Upload audio + text + user_id, get full assessment + feedback
- `GET /user/{user_id}` — Get user learning history
- `GET /health` — Health check

## Output interpretation

- **overall_score**: 0-100, higher is better
- **pherr_prob**: 0-1, probability of phoneme error. Above 0.70 = likely error
- **score per phoneme**: 0-100, the model's predicted quality score
- **GOP**: Goodness of Pronunciation, negative means likely mispronounced
- **weak_phonemes**: Phonemes with historically high error rates for this user

## Response format

After running the tool, present results to the user in a friendly way:
1. Show overall score
2. Highlight which words/phonemes had errors
3. Include the AI-generated feedback (from the `feedback` field)
4. Show the learning plan
5. If the user has history, mention their progress

## Requirements

- Python 3.10+
- GPU with ~22GB VRAM (for WavLM model inference)
- `OPENAI_API_KEY` environment variable (for LLM feedback)
- ffmpeg (for audio decoding)
- Model weights auto-download from HuggingFace on first run (~1.2GB)
