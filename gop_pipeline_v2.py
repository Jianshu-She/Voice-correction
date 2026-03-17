"""
GOP Pipeline V2 - Improved Pronunciation Scoring
==================================================
Key improvements over V1:
1. wav2vec2-LARGE model for better acoustic representations
2. Proper GOP scoring: log P(target_phone | audio) - log P(best_other_phone | audio)
3. Score calibration via linear regression on a subset
4. Run on full dataset with train/test split for evaluation
"""

import torch
import torchaudio
import torchaudio.functional as F_audio
import numpy as np
import re
import openpyxl
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1. Model Setup - Use LARGE model
# ============================================================
print("Loading wav2vec2-large model...")
bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_960H
model = bundle.get_model().to(device)
model.eval()
labels = bundle.get_labels()
SAMPLE_RATE = bundle.sample_rate

label2idx = {l: i for i, l in enumerate(labels)}
BLANK = label2idx["-"]
SPACE = label2idx["|"]
print(f"Model loaded. Labels: {len(labels)}")


# ============================================================
# 2. Core Functions
# ============================================================

def load_audio(audio_path: str) -> torch.Tensor:
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = F_audio.resample(waveform, sr, SAMPLE_RATE)
    return waveform


@torch.no_grad()
def get_emissions(waveform: torch.Tensor) -> torch.Tensor:
    waveform = waveform.to(device)
    emissions, _ = model(waveform)
    emissions = torch.log_softmax(emissions, dim=-1)
    return emissions.squeeze(0).cpu()  # (T, C)


def text_to_tokens(text: str) -> list:
    text = text.upper().strip()
    text = re.sub(r"[^A-Z' ]", "", text)
    tokens = []
    for ch in text:
        if ch == " ":
            tokens.append(SPACE)
        elif ch in label2idx:
            tokens.append(label2idx[ch])
    return tokens


def forced_align_viterbi(emissions: torch.Tensor, tokens: list):
    """CTC Viterbi forced alignment. Returns list of (frame_idx, token_idx)."""
    T, C = emissions.shape
    S = len(tokens)

    if S == 0:
        return []

    # Extended token sequence with blanks interleaved
    extended = [BLANK]
    for t in tokens:
        extended.append(t)
        extended.append(BLANK)
    S_ext = len(extended)

    if T < S:  # Audio too short for text
        return []

    NEG_INF = float("-inf")
    dp = np.full((T, S_ext), NEG_INF, dtype=np.float64)
    bp = np.zeros((T, S_ext), dtype=np.int32)

    dp[0][0] = emissions[0, extended[0]].item()
    if S_ext > 1:
        dp[0][1] = emissions[0, extended[1]].item()

    for t in range(1, T):
        for s in range(S_ext):
            tok = extended[s]
            emit = emissions[t, tok].item()

            best = dp[t-1][s]
            best_s = s

            if s > 0 and dp[t-1][s-1] > best:
                best = dp[t-1][s-1]
                best_s = s - 1

            if s > 1 and extended[s] != BLANK and extended[s] != extended[s-2]:
                if dp[t-1][s-2] > best:
                    best = dp[t-1][s-2]
                    best_s = s - 2

            dp[t][s] = best + emit
            bp[t][s] = best_s

    # Backtrace
    if S_ext >= 2 and dp[T-1][S_ext-1] >= dp[T-1][S_ext-2]:
        s = S_ext - 1
    elif S_ext >= 2:
        s = S_ext - 2
    else:
        s = 0

    path = []
    for t in range(T - 1, -1, -1):
        path.append((t, extended[s]))
        s = bp[t][s]
    path.reverse()
    return path


def compute_gop_features(emissions: torch.Tensor, text: str) -> dict:
    """
    Compute multiple GOP-related features for scoring.

    Features per word:
    1. mean_log_posterior: average log P(target_char | audio) over aligned frames
    2. gop_score: log P(target) - log P(best_competing), averaged per char
    3. duration_ratio: actual frames / expected frames per char
    4. min_char_score: worst character in the word
    5. entropy: average entropy over aligned frames (confidence measure)
    """
    tokens = text_to_tokens(text)
    if not tokens:
        return {"features": [], "word_texts": [], "raw_scores": {}}

    T, C = emissions.shape
    path = forced_align_viterbi(emissions, tokens)

    if not path:
        return {"features": [], "word_texts": [], "raw_scores": {}}

    # Group frames by non-blank token
    char_segments = []
    current_tok = None
    current_frames = []

    for frame_idx, tok_idx in path:
        if tok_idx == BLANK:
            if current_tok is not None:
                char_segments.append((current_tok, current_frames))
                current_tok = None
                current_frames = []
            continue
        if tok_idx != current_tok:
            if current_tok is not None:
                char_segments.append((current_tok, current_frames))
            current_tok = tok_idx
            current_frames = [frame_idx]
        else:
            current_frames.append(frame_idx)

    if current_tok is not None:
        char_segments.append((current_tok, current_frames))

    # Compute per-character GOP features
    char_features = []
    for tok_idx, frames in char_segments:
        if tok_idx == SPACE:
            continue

        frame_emissions = emissions[frames]  # (num_frames, C)

        # Feature 1: mean log posterior of target
        target_log_post = frame_emissions[:, tok_idx].mean().item()

        # Feature 2: GOP = log P(target) - max log P(other non-blank, non-target)
        mask = torch.ones(C, dtype=torch.bool)
        mask[BLANK] = False
        mask[tok_idx] = False
        if mask.any():
            best_other = frame_emissions[:, mask].max(dim=-1).values.mean().item()
        else:
            best_other = -30.0
        gop = target_log_post - best_other

        # Feature 3: entropy of posterior distribution
        probs = torch.exp(frame_emissions)
        entropy = -(probs * frame_emissions).sum(dim=-1).mean().item()

        # Feature 4: duration (number of frames)
        duration = len(frames)

        char_features.append({
            "label": labels[tok_idx],
            "log_post": target_log_post,
            "gop": gop,
            "entropy": entropy,
            "duration": duration,
        })

    # Group into words
    words_text = re.sub(r"[^A-Za-z' ]", "", text).split()
    word_features = []
    char_idx = 0

    for word in words_text:
        word_upper = word.upper()
        n_chars = sum(1 for ch in word_upper if ch in label2idx)

        if char_idx + n_chars <= len(char_features):
            word_chars = char_features[char_idx:char_idx + n_chars]
            char_idx += n_chars
        else:
            # Not enough char segments - take what we can
            word_chars = char_features[char_idx:]
            char_idx = len(char_features)

        if word_chars:
            log_posts = [c["log_post"] for c in word_chars]
            gops = [c["gop"] for c in word_chars]
            entropies = [c["entropy"] for c in word_chars]
            durations = [c["duration"] for c in word_chars]

            word_features.append({
                "word": word,
                "mean_log_post": np.mean(log_posts),
                "min_log_post": np.min(log_posts),
                "mean_gop": np.mean(gops),
                "min_gop": np.min(gops),
                "mean_entropy": np.mean(entropies),
                "total_duration": sum(durations),
                "mean_duration_per_char": np.mean(durations),
                "n_chars": len(word_chars),
            })
        else:
            word_features.append({
                "word": word,
                "mean_log_post": -15.0,
                "min_log_post": -15.0,
                "mean_gop": -10.0,
                "min_gop": -10.0,
                "mean_entropy": 5.0,
                "total_duration": 0,
                "mean_duration_per_char": 0,
                "n_chars": 0,
            })

    # Sentence-level features (aggregate)
    if word_features:
        all_log_posts = [w["mean_log_post"] for w in word_features]
        all_gops = [w["mean_gop"] for w in word_features]
        all_min_gops = [w["min_gop"] for w in word_features]
        all_entropies = [w["mean_entropy"] for w in word_features]

        sentence_features = {
            "mean_log_post": np.mean(all_log_posts),
            "min_log_post": np.min(all_log_posts),
            "mean_gop": np.mean(all_gops),
            "min_gop": np.min(all_min_gops),
            "mean_entropy": np.mean(all_entropies),
            "n_words": len(word_features),
            "total_frames": T,
        }
    else:
        sentence_features = {
            "mean_log_post": -15.0, "min_log_post": -15.0,
            "mean_gop": -10.0, "min_gop": -10.0,
            "mean_entropy": 5.0, "n_words": 0, "total_frames": T,
        }

    return {
        "word_features": word_features,
        "sentence_features": sentence_features,
    }


# ============================================================
# 3. Load Data
# ============================================================

def load_eval_log(xlsx_path: str) -> list:
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb.active
    records = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        fname, content, raw_json = row
        if not fname or not content:
            continue
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


# ============================================================
# 4. Main Evaluation
# ============================================================

def main():
    audio_dir = Path("/mnt/weka/home/jianshu.she/Voice-correction/audio_files")
    xlsx_path = "/mnt/weka/home/jianshu.she/Voice-correction/eval_log.xlsx"

    print("\nLoading evaluation log...")
    records = load_eval_log(xlsx_path)
    print(f"Loaded {len(records)} records")

    # Process ALL samples
    print(f"\nProcessing all {len(records)} samples...")
    all_features = []
    all_gt_accuracy = []
    all_gt_integrity = []
    skipped = 0

    for i, rec in enumerate(records):
        audio_path = audio_dir / rec["file_name"]
        if not audio_path.exists():
            skipped += 1
            continue

        try:
            waveform = load_audio(str(audio_path))
            emissions = get_emissions(waveform)
            result = compute_gop_features(emissions, rec["content"])

            sf = result["sentence_features"]
            feature_vec = [
                sf["mean_log_post"],
                sf["min_log_post"],
                sf["mean_gop"],
                sf["min_gop"],
                sf["mean_entropy"],
                sf["n_words"],
            ]
            all_features.append(feature_vec)
            all_gt_accuracy.append(rec["gt_accuracy"])
            all_gt_integrity.append(rec["gt_integrity"])

            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(records)}...")

        except Exception as e:
            skipped += 1
            continue

    print(f"\nProcessed: {len(all_features)}, Skipped: {skipped}")

    X = np.array(all_features)
    y_acc = np.array(all_gt_accuracy)
    y_int = np.array(all_gt_integrity)

    # ============================================================
    # 5. Analysis: Raw Feature Correlations
    # ============================================================
    feature_names = ["mean_log_post", "min_log_post", "mean_gop", "min_gop", "mean_entropy", "n_words"]

    print(f"\n{'='*70}")
    print("RAW FEATURE CORRELATIONS WITH GT ACCURACY")
    print(f"{'='*70}")
    for j, name in enumerate(feature_names):
        corr, pval = stats.pearsonr(X[:, j], y_acc)
        sp, sp_pval = stats.spearmanr(X[:, j], y_acc)
        print(f"  {name:20s}  Pearson={corr:+.3f} (p={pval:.4f})  Spearman={sp:+.3f} (p={sp_pval:.4f})")

    # ============================================================
    # 6. Calibrated Scoring via Linear Regression
    # ============================================================
    print(f"\n{'='*70}")
    print("CALIBRATED SCORING (Linear Regression)")
    print(f"{'='*70}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_acc, test_size=0.3, random_state=42
    )

    reg = LinearRegression()
    reg.fit(X_train, y_train)

    y_pred_train = reg.predict(X_train)
    y_pred_test = reg.predict(X_test)

    # Clip to valid range
    y_pred_train = np.clip(y_pred_train, 0, 100)
    y_pred_test = np.clip(y_pred_test, 0, 100)

    # Train metrics
    train_mae = np.mean(np.abs(y_train - y_pred_train))
    train_corr, _ = stats.pearsonr(y_train, y_pred_train)
    train_spearman, _ = stats.spearmanr(y_train, y_pred_train)

    # Test metrics
    test_mae = np.mean(np.abs(y_test - y_pred_test))
    test_corr, _ = stats.pearsonr(y_test, y_pred_test)
    test_spearman, _ = stats.spearmanr(y_test, y_pred_test)
    test_rmse = np.sqrt(np.mean((y_test - y_pred_test) ** 2))

    print(f"\n  TRAIN ({len(X_train)} samples):")
    print(f"    MAE:      {train_mae:.2f}")
    print(f"    Pearson:  {train_corr:.3f}")
    print(f"    Spearman: {train_spearman:.3f}")

    print(f"\n  TEST ({len(X_test)} samples):")
    print(f"    MAE:      {test_mae:.2f}")
    print(f"    RMSE:     {test_rmse:.2f}")
    print(f"    Pearson:  {test_corr:.3f}")
    print(f"    Spearman: {test_spearman:.3f}")

    print(f"\n  Regression coefficients:")
    for name, coef in zip(feature_names, reg.coef_):
        print(f"    {name:20s}  {coef:+.4f}")
    print(f"    {'intercept':20s}  {reg.intercept_:+.4f}")

    # ============================================================
    # 7. Error Distribution
    # ============================================================
    errors = y_test - y_pred_test
    abs_errors = np.abs(errors)

    print(f"\n{'='*70}")
    print("ERROR DISTRIBUTION (TEST SET)")
    print(f"{'='*70}")
    print(f"  Mean error (bias):     {np.mean(errors):+.2f}")
    print(f"  Std of errors:         {np.std(errors):.2f}")
    print(f"  |error| < 5:           {np.mean(abs_errors < 5) * 100:.1f}%")
    print(f"  |error| < 10:          {np.mean(abs_errors < 10) * 100:.1f}%")
    print(f"  |error| < 15:          {np.mean(abs_errors < 15) * 100:.1f}%")
    print(f"  |error| < 20:          {np.mean(abs_errors < 20) * 100:.1f}%")
    print(f"  Max |error|:           {np.max(abs_errors):.1f}")

    # Show some example predictions
    print(f"\n{'='*70}")
    print("SAMPLE PREDICTIONS (TEST SET)")
    print(f"{'='*70}")
    indices = np.argsort(abs_errors)
    # Show best 5 and worst 5
    print("\n  BEST predictions:")
    for idx in indices[:5]:
        print(f"    GT={y_test[idx]:5.1f}  Pred={y_pred_test[idx]:5.1f}  err={errors[idx]:+5.1f}")

    print("\n  WORST predictions:")
    for idx in indices[-5:]:
        print(f"    GT={y_test[idx]:5.1f}  Pred={y_pred_test[idx]:5.1f}  err={errors[idx]:+5.1f}")

    # ============================================================
    # 8. Score Distribution Analysis
    # ============================================================
    print(f"\n{'='*70}")
    print("SCORE DISTRIBUTION")
    print(f"{'='*70}")
    for pct in [10, 25, 50, 75, 90]:
        print(f"  GT accuracy P{pct:2d}: {np.percentile(y_acc, pct):5.1f}   "
              f"Pred P{pct:2d}: {np.percentile(y_pred_test, pct):5.1f}")


if __name__ == "__main__":
    main()
