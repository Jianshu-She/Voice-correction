"""
GOP (Goodness of Pronunciation) Pipeline
=========================================
Uses wav2vec2 forced alignment + CTC posterior probabilities to score pronunciation.

Steps:
1. Load audio, resample to 16kHz
2. Run wav2vec2 to get frame-level CTC log-posteriors
3. Force-align expected text (character-level) using CTC forced alignment
4. Compute GOP score per character segment: mean log-posterior of the target label
5. Aggregate to word-level and overall scores
"""

import torch
import torchaudio
import torchaudio.functional as F_audio
import numpy as np
import json
import re
import openpyxl
from pathlib import Path
from dataclasses import dataclass


# ============================================================
# 1. Model Setup
# ============================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Loading wav2vec2 model...")
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)
model.eval()
labels = bundle.get_labels()
SAMPLE_RATE = bundle.sample_rate  # 16000

# Build label-to-index mapping
label2idx = {l: i for i, l in enumerate(labels)}
BLANK = label2idx["-"]
SPACE = label2idx["|"]

print(f"Labels: {labels}")
print(f"Blank idx: {BLANK}, Space idx: {SPACE}")


# ============================================================
# 2. Audio Loading
# ============================================================

def load_audio(audio_path: str) -> torch.Tensor:
    """Load audio file and resample to 16kHz mono."""
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = F_audio.resample(waveform, sr, SAMPLE_RATE)
    return waveform


# ============================================================
# 3. Get CTC Log-Posteriors
# ============================================================

@torch.no_grad()
def get_emissions(waveform: torch.Tensor) -> torch.Tensor:
    """Get CTC log-posteriors from wav2vec2. Returns (T, C) tensor."""
    waveform = waveform.to(device)
    emissions, _ = model(waveform)
    emissions = torch.log_softmax(emissions, dim=-1)
    return emissions.squeeze(0).cpu()  # (T, C)


# ============================================================
# 4. CTC Forced Alignment
# ============================================================

@dataclass
class AlignedSegment:
    label: str
    label_idx: int
    start_frame: int
    end_frame: int
    score: float  # mean log-posterior


def text_to_tokens(text: str) -> list:
    """Convert text to wav2vec2 token indices.

    The model uses: '-' (blank), '|' (space), and uppercase letters + apostrophe.
    We insert '|' between words.
    """
    text = text.upper().strip()
    # Remove punctuation except apostrophe
    text = re.sub(r"[^A-Z' ]", "", text)

    tokens = []
    for i, ch in enumerate(text):
        if ch == " ":
            tokens.append(SPACE)
        elif ch in label2idx:
            tokens.append(label2idx[ch])
        # Skip unknown characters
    return tokens


def forced_align(emissions: torch.Tensor, tokens: list) -> list:
    """
    CTC forced alignment using Viterbi algorithm.

    For each token in the target sequence, finds the optimal frame assignment
    that maximizes the total log-posterior probability.

    Uses torchaudio's built-in forced alignment if available,
    otherwise falls back to custom implementation.
    """
    try:
        # Use torchaudio's forced_align (available in newer versions)
        targets = torch.tensor([tokens], dtype=torch.int32)
        aligned_tokens, scores = torchaudio.functional.forced_align(
            emissions.unsqueeze(0), targets, blank=BLANK
        )
        return aligned_tokens.squeeze(0).tolist(), scores.squeeze(0).tolist()
    except AttributeError:
        pass

    # Fallback: custom Viterbi forced alignment
    T, C = emissions.shape
    S = len(tokens)

    # Create token sequence with blanks interleaved
    # [blank, tok0, blank, tok1, blank, ..., tokN, blank]
    extended = [BLANK]
    for t in tokens:
        extended.append(t)
        extended.append(BLANK)
    S_ext = len(extended)

    # Viterbi forward pass
    # dp[t][s] = max log-prob of aligning first t frames to first s extended tokens
    NEG_INF = float("-inf")
    dp = np.full((T, S_ext), NEG_INF, dtype=np.float64)
    bp = np.zeros((T, S_ext), dtype=np.int32)  # backpointer

    # Initialize: can start with blank or first token
    dp[0][0] = emissions[0, extended[0]].item()
    if S_ext > 1:
        dp[0][1] = emissions[0, extended[1]].item()

    for t in range(1, T):
        for s in range(S_ext):
            tok = extended[s]
            emit = emissions[t, tok].item()

            # Option 1: stay at same state
            best = dp[t-1][s]
            best_s = s

            # Option 2: transition from previous state
            if s > 0 and dp[t-1][s-1] > best:
                best = dp[t-1][s-1]
                best_s = s - 1

            # Option 3: skip blank (for non-blank, non-repeat tokens)
            if s > 1 and extended[s] != BLANK and extended[s] != extended[s-2]:
                if dp[t-1][s-2] > best:
                    best = dp[t-1][s-2]
                    best_s = s - 2

            dp[t][s] = best + emit
            bp[t][s] = best_s

    # Backtrace
    # End at last or second-to-last extended token
    if dp[T-1][S_ext-1] >= dp[T-1][S_ext-2]:
        s = S_ext - 1
    else:
        s = S_ext - 2

    path = []
    for t in range(T - 1, -1, -1):
        path.append((t, s, extended[s]))
        s = bp[t][s]
    path.reverse()

    # Extract token-level frame assignments and scores
    frame_labels = [(t, tok_idx) for t, s, tok_idx in path]
    return frame_labels


def align_and_score(emissions: torch.Tensor, text: str) -> dict:
    """
    Full alignment pipeline:
    1. Convert text to tokens
    2. Force-align to get frame assignments
    3. Compute per-character GOP scores
    4. Aggregate to word-level and overall scores
    """
    tokens = text_to_tokens(text)
    if not tokens:
        return {"accuracy": 0, "integrity": 0, "details": []}

    T, C = emissions.shape

    # Get frame-level alignment
    frame_labels = forced_align(emissions, tokens)

    # Parse alignment result based on format
    if isinstance(frame_labels, tuple):
        # torchaudio format: (aligned_tokens, scores)
        aligned_tokens, scores_list = frame_labels
        # Group consecutive frames by token
        segments = []
        current_tok = None
        current_frames = []
        current_scores = []

        for frame_idx, (tok_idx, score) in enumerate(zip(aligned_tokens, scores_list)):
            if tok_idx == BLANK:
                if current_tok is not None:
                    segments.append(AlignedSegment(
                        label=labels[current_tok],
                        label_idx=current_tok,
                        start_frame=current_frames[0],
                        end_frame=current_frames[-1],
                        score=np.mean(current_scores)
                    ))
                    current_tok = None
                    current_frames = []
                    current_scores = []
                continue

            if tok_idx != current_tok:
                if current_tok is not None:
                    segments.append(AlignedSegment(
                        label=labels[current_tok],
                        label_idx=current_tok,
                        start_frame=current_frames[0],
                        end_frame=current_frames[-1],
                        score=np.mean(current_scores)
                    ))
                current_tok = tok_idx
                current_frames = [frame_idx]
                current_scores = [score]
            else:
                current_frames.append(frame_idx)
                current_scores.append(score)

        if current_tok is not None:
            segments.append(AlignedSegment(
                label=labels[current_tok],
                label_idx=current_tok,
                start_frame=current_frames[0],
                end_frame=current_frames[-1],
                score=np.mean(current_scores)
            ))
    else:
        # Custom Viterbi format: list of (frame, token_idx)
        segments = []
        current_tok = None
        current_frames = []
        current_emissions = []

        for frame_idx, tok_idx in frame_labels:
            if tok_idx == BLANK:
                if current_tok is not None:
                    segments.append(AlignedSegment(
                        label=labels[current_tok],
                        label_idx=current_tok,
                        start_frame=current_frames[0],
                        end_frame=current_frames[-1],
                        score=np.mean(current_emissions)
                    ))
                    current_tok = None
                    current_frames = []
                    current_emissions = []
                continue

            emit_score = emissions[frame_idx, tok_idx].item()

            if tok_idx != current_tok:
                if current_tok is not None:
                    segments.append(AlignedSegment(
                        label=labels[current_tok],
                        label_idx=current_tok,
                        start_frame=current_frames[0],
                        end_frame=current_frames[-1],
                        score=np.mean(current_emissions)
                    ))
                current_tok = tok_idx
                current_frames = [frame_idx]
                current_emissions = [emit_score]
            else:
                current_frames.append(frame_idx)
                current_emissions.append(emit_score)

        if current_tok is not None:
            segments.append(AlignedSegment(
                label=labels[current_tok],
                label_idx=current_tok,
                start_frame=current_frames[0],
                end_frame=current_frames[-1],
                score=np.mean(current_emissions)
            ))

    # Group segments into words (split by SPACE)
    words_text = re.sub(r"[^A-Za-z' ]", "", text).split()
    word_details = []
    seg_idx = 0

    for word in words_text:
        char_scores = []
        word_upper = word.upper()

        for ch in word_upper:
            if ch not in label2idx:
                continue
            # Find matching segment
            while seg_idx < len(segments) and segments[seg_idx].label == "|":
                seg_idx += 1

            if seg_idx < len(segments) and segments[seg_idx].label == ch:
                char_scores.append(segments[seg_idx].score)
                seg_idx += 1
            else:
                # Alignment mismatch - try to recover
                char_scores.append(-10.0)  # low score for missing alignment
                # Try to find next matching segment
                for j in range(seg_idx, min(seg_idx + 3, len(segments))):
                    if segments[j].label == ch:
                        seg_idx = j + 1
                        char_scores[-1] = segments[j].score
                        break

        if char_scores:
            # Convert log-posterior to 0-100 score
            # log-posterior ranges from ~-30 (bad) to ~0 (perfect)
            # Map: -10 -> 0, 0 -> 100
            mean_log_post = np.mean(char_scores)
            word_score = max(0, min(100, (mean_log_post + 10) * 10))
        else:
            word_score = 0

        word_details.append({
            "word": word,
            "score": round(word_score, 1),
            "raw_log_posterior": round(float(np.mean(char_scores)) if char_scores else -10, 3),
            "char_scores": [round(s, 3) for s in char_scores],
        })

        # Skip space segment
        while seg_idx < len(segments) and segments[seg_idx].label == "|":
            seg_idx += 1

    # Overall accuracy: mean of word scores
    if word_details:
        overall_accuracy = np.mean([w["score"] for w in word_details])
    else:
        overall_accuracy = 0

    # Integrity: fraction of expected words that have non-zero alignment
    aligned_words = sum(1 for w in word_details if w["score"] > 10)
    integrity = (aligned_words / len(word_details)) * 100 if word_details else 0

    return {
        "accuracy": round(float(overall_accuracy), 1),
        "integrity": round(float(integrity), 1),
        "details": word_details,
        "segments": [(s.label, s.start_frame, s.end_frame, round(s.score, 3)) for s in segments],
    }


# ============================================================
# 5. Load Ground Truth & Run Evaluation
# ============================================================

def load_eval_log(xlsx_path: str) -> list:
    """Load evaluation log from xlsx."""
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb.active

    records = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        fname, content, raw_json = row
        if not fname or not content:
            continue

        # Extract ground truth accuracy
        s = raw_json
        for _ in range(5):
            s = s.replace("\\\\", "\\")
        s = s.replace('\\"', '"')

        accuracy_match = re.search(r'"accuracy":\s*([\d.]+)', s)
        integrity_match = re.search(r'"integrity":\s*([\d.]+)', s)

        gt_accuracy = float(accuracy_match.group(1)) if accuracy_match else None
        gt_integrity = float(integrity_match.group(1)) if integrity_match else None

        records.append({
            "file_name": fname,
            "content": content,
            "gt_accuracy": gt_accuracy,
            "gt_integrity": gt_integrity,
        })

    return records


def main():
    audio_dir = Path("/mnt/weka/home/jianshu.she/Voice-correction/audio_files")
    xlsx_path = "/mnt/weka/home/jianshu.she/Voice-correction/eval_log.xlsx"

    print("\nLoading evaluation log...")
    records = load_eval_log(xlsx_path)
    print(f"Loaded {len(records)} records")

    # Test on first 20 samples
    N = 20
    results = []

    print(f"\nProcessing {N} samples...\n")
    for i, rec in enumerate(records[:N]):
        audio_path = audio_dir / rec["file_name"]
        if not audio_path.exists():
            print(f"  [{i+1}] SKIP - file not found: {rec['file_name']}")
            continue

        try:
            waveform = load_audio(str(audio_path))
            emissions = get_emissions(waveform)
            result = align_and_score(emissions, rec["content"])

            gt_acc = rec["gt_accuracy"]
            pred_acc = result["accuracy"]
            diff = abs(gt_acc - pred_acc) if gt_acc is not None else None

            results.append({
                "file": rec["file_name"][:30],
                "text": rec["content"][:30],
                "gt_accuracy": gt_acc,
                "pred_accuracy": pred_acc,
                "gt_integrity": rec["gt_integrity"],
                "pred_integrity": result["integrity"],
                "diff": diff,
            })

            print(f"  [{i+1:2d}] text='{rec['content'][:25]:25s}' | "
                  f"GT_acc={gt_acc:5.1f}  Pred_acc={pred_acc:5.1f}  diff={diff:5.1f} | "
                  f"GT_int={rec['gt_integrity']:5.1f}  Pred_int={result['integrity']:5.1f}")

            # Print word details for first 3 samples
            if i < 3:
                for wd in result["details"]:
                    print(f"       word='{wd['word']:15s}' score={wd['score']:5.1f}  "
                          f"raw_logpost={wd['raw_log_posterior']:7.3f}  "
                          f"chars={wd['char_scores']}")
                print()

        except Exception as e:
            print(f"  [{i+1}] ERROR - {rec['file_name'][:30]}: {e}")

    # Summary statistics
    if results:
        diffs = [r["diff"] for r in results if r["diff"] is not None]
        gt_accs = [r["gt_accuracy"] for r in results if r["gt_accuracy"] is not None]
        pred_accs = [r["pred_accuracy"] for r in results]

        print(f"\n{'='*60}")
        print(f"SUMMARY ({len(results)} samples)")
        print(f"{'='*60}")
        print(f"  Mean Absolute Error (accuracy): {np.mean(diffs):.1f}")
        print(f"  Median Absolute Error:          {np.median(diffs):.1f}")
        print(f"  GT accuracy range:              {min(gt_accs):.1f} - {max(gt_accs):.1f}")
        print(f"  Pred accuracy range:            {min(pred_accs):.1f} - {max(pred_accs):.1f}")

        # Correlation
        from scipy import stats
        corr, pval = stats.pearsonr(gt_accs, pred_accs)
        spearman, sp_pval = stats.spearmanr(gt_accs, pred_accs)
        print(f"  Pearson correlation:            {corr:.3f} (p={pval:.4f})")
        print(f"  Spearman correlation:           {spearman:.3f} (p={sp_pval:.4f})")


if __name__ == "__main__":
    main()
