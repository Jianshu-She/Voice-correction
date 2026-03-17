"""
GOP Pipeline V3 - With ASR verification + better features
==========================================================
Key improvements over V2:
1. Add Whisper ASR to compute text similarity (catches silent/wrong audio)
2. Use WER/CER as additional features for integrity detection
3. Better feature engineering with non-linear transforms
4. Separate model for integrity prediction
"""

import torch
import torchaudio
import torchaudio.functional as F_audio
import numpy as np
import re
import openpyxl
import whisper
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1. Load Models
# ============================================================
print("Loading wav2vec2-large...")
bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_960H
w2v_model = bundle.get_model().to(device)
w2v_model.eval()
labels = bundle.get_labels()
SAMPLE_RATE = bundle.sample_rate
label2idx = {l: i for i, l in enumerate(labels)}
BLANK = label2idx["-"]
SPACE = label2idx["|"]

print("Loading Whisper-base...")
whisper_model = whisper.load_model("base", device=device)
print("Models loaded.\n")


# ============================================================
# 2. Audio & Wav2Vec2 Functions
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
    emissions, _ = w2v_model(waveform)
    emissions = torch.log_softmax(emissions, dim=-1)
    return emissions.squeeze(0).cpu()


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


def forced_align_viterbi(emissions, tokens):
    T, C = emissions.shape
    S = len(tokens)
    if S == 0 or T < S:
        return []

    extended = [BLANK]
    for t in tokens:
        extended.append(t)
        extended.append(BLANK)
    S_ext = len(extended)

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


# ============================================================
# 3. Whisper ASR + Text Similarity
# ============================================================

def transcribe_whisper(audio_path: str) -> str:
    """Transcribe audio using Whisper."""
    result = whisper_model.transcribe(audio_path, language="en", fp16=True)
    return result["text"].strip()


def compute_text_similarity(reference: str, hypothesis: str) -> dict:
    """Compute text similarity metrics between reference and ASR output."""
    ref_clean = re.sub(r"[^a-z ]", "", reference.lower()).split()
    hyp_clean = re.sub(r"[^a-z ]", "", hypothesis.lower()).split()

    # Word-level similarity
    if not ref_clean:
        return {"word_accuracy": 0, "char_similarity": 0, "word_match_ratio": 0}

    # Simple WER-based accuracy
    matcher = SequenceMatcher(None, ref_clean, hyp_clean)
    word_similarity = matcher.ratio()

    # Character-level similarity
    ref_chars = "".join(ref_clean)
    hyp_chars = "".join(hyp_clean)
    char_matcher = SequenceMatcher(None, ref_chars, hyp_chars)
    char_similarity = char_matcher.ratio()

    # Word match ratio (exact matches / total reference words)
    ref_set = set(ref_clean)
    hyp_set = set(hyp_clean)
    match_ratio = len(ref_set & hyp_set) / len(ref_set) if ref_set else 0

    return {
        "word_accuracy": word_similarity,
        "char_similarity": char_similarity,
        "word_match_ratio": match_ratio,
    }


# ============================================================
# 4. Feature Extraction
# ============================================================

def extract_features(audio_path: str, text: str) -> dict:
    """Extract all features for a single audio-text pair."""
    # 1. Load audio & get emissions
    waveform = load_audio(audio_path)
    emissions = get_emissions(waveform)
    T, C = emissions.shape

    # 2. Forced alignment + GOP features
    tokens = text_to_tokens(text)
    path = forced_align_viterbi(emissions, tokens)

    if not path:
        gop_features = {
            "mean_log_post": -15.0, "min_log_post": -15.0,
            "std_log_post": 0.0,
            "mean_gop": -10.0, "min_gop": -10.0, "std_gop": 0.0,
            "mean_entropy": 5.0,
            "alignment_score": -30.0,
            "frames_per_char": 0.0,
        }
    else:
        # Group frames by character
        char_log_posts = []
        char_gops = []
        char_entropies = []
        current_tok = None
        current_frames = []

        def process_segment(tok_idx, frames):
            if tok_idx == SPACE or tok_idx == BLANK:
                return
            frame_emissions = emissions[frames]
            log_post = frame_emissions[:, tok_idx].mean().item()

            mask = torch.ones(C, dtype=torch.bool)
            mask[BLANK] = False
            mask[tok_idx] = False
            best_other = frame_emissions[:, mask].max(dim=-1).values.mean().item() if mask.any() else -30.0
            gop = log_post - best_other

            probs = torch.exp(frame_emissions)
            entropy = -(probs * frame_emissions).sum(dim=-1).mean().item()

            char_log_posts.append(log_post)
            char_gops.append(gop)
            char_entropies.append(entropy)

        for frame_idx, tok_idx in path:
            if tok_idx != current_tok:
                if current_tok is not None and current_tok != BLANK:
                    process_segment(current_tok, current_frames)
                current_tok = tok_idx
                current_frames = [frame_idx]
            else:
                current_frames.append(frame_idx)
        if current_tok is not None and current_tok != BLANK:
            process_segment(current_tok, current_frames)

        n_chars = len(text_to_tokens(re.sub(r"[^A-Za-z']", "", text)))
        if not n_chars:
            n_chars = 1

        if char_log_posts:
            gop_features = {
                "mean_log_post": np.mean(char_log_posts),
                "min_log_post": np.min(char_log_posts),
                "std_log_post": np.std(char_log_posts) if len(char_log_posts) > 1 else 0,
                "mean_gop": np.mean(char_gops),
                "min_gop": np.min(char_gops),
                "std_gop": np.std(char_gops) if len(char_gops) > 1 else 0,
                "mean_entropy": np.mean(char_entropies),
                "alignment_score": sum(emissions[f, t].item() for f, t in path) / len(path),
                "frames_per_char": T / n_chars,
            }
        else:
            gop_features = {
                "mean_log_post": -15.0, "min_log_post": -15.0,
                "std_log_post": 0.0,
                "mean_gop": -10.0, "min_gop": -10.0, "std_gop": 0.0,
                "mean_entropy": 5.0,
                "alignment_score": -30.0,
                "frames_per_char": 0.0,
            }

    # 3. Whisper ASR + similarity
    asr_text = transcribe_whisper(audio_path)
    sim_features = compute_text_similarity(text, asr_text)

    # 4. Audio-level features
    audio_duration = waveform.shape[1] / SAMPLE_RATE
    energy = (waveform ** 2).mean().item()
    text_len = len(re.sub(r"[^A-Za-z ]", "", text).split())

    audio_features = {
        "duration": audio_duration,
        "energy": energy,
        "text_word_count": text_len,
        "duration_per_word": audio_duration / max(text_len, 1),
    }

    return {
        **gop_features,
        **sim_features,
        **audio_features,
        "asr_text": asr_text,
    }


def features_to_vector(feat: dict) -> list:
    """Convert feature dict to numeric vector for ML model."""
    keys = [
        "mean_log_post", "min_log_post", "std_log_post",
        "mean_gop", "min_gop", "std_gop",
        "mean_entropy",
        "alignment_score",
        "frames_per_char",
        "word_accuracy", "char_similarity", "word_match_ratio",
        "duration", "energy", "text_word_count", "duration_per_word",
    ]
    return [feat.get(k, 0.0) for k in keys]


FEATURE_NAMES = [
    "mean_log_post", "min_log_post", "std_log_post",
    "mean_gop", "min_gop", "std_gop",
    "mean_entropy",
    "alignment_score",
    "frames_per_char",
    "word_accuracy", "char_similarity", "word_match_ratio",
    "duration", "energy", "text_word_count", "duration_per_word",
]


# ============================================================
# 5. Load Data
# ============================================================

def load_eval_log(xlsx_path):
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
        acc = re.search(r'"accuracy":\s*([\d.]+)', s)
        integ = re.search(r'"integrity":\s*([\d.]+)', s)
        records.append({
            "file_name": fname,
            "content": content,
            "gt_accuracy": float(acc.group(1)) if acc else None,
            "gt_integrity": float(integ.group(1)) if integ else None,
        })
    return records


# ============================================================
# 6. Main
# ============================================================

def main():
    audio_dir = Path("/mnt/weka/home/jianshu.she/Voice-correction/audio_files")
    xlsx_path = "/mnt/weka/home/jianshu.she/Voice-correction/eval_log.xlsx"

    print("Loading evaluation log...")
    records = load_eval_log(xlsx_path)
    print(f"Loaded {len(records)} records\n")

    # Extract features for all samples
    all_features = []
    all_gt_acc = []
    all_gt_int = []
    all_asr = []
    skipped = 0

    for i, rec in enumerate(records):
        audio_path = audio_dir / rec["file_name"]
        if not audio_path.exists():
            skipped += 1
            continue

        try:
            feat = extract_features(str(audio_path), rec["content"])
            all_features.append(features_to_vector(feat))
            all_gt_acc.append(rec["gt_accuracy"])
            all_gt_int.append(rec["gt_integrity"])
            all_asr.append(feat["asr_text"])

            if (i + 1) % 50 == 0:
                print(f"  [{i+1:4d}/{len(records)}] "
                      f"text='{rec['content'][:25]:25s}' "
                      f"GT_acc={rec['gt_accuracy']:5.1f} "
                      f"ASR='{feat['asr_text'][:30]}' "
                      f"word_sim={feat['word_accuracy']:.2f} "
                      f"gop={feat['mean_gop']:.2f}")
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"  ERROR [{i+1}]: {e}")
            continue

    print(f"\nProcessed: {len(all_features)}, Skipped: {skipped}\n")

    X = np.array(all_features)
    y_acc = np.array(all_gt_acc)
    y_int = np.array(all_gt_int)

    # ============================================================
    # Feature Correlation Analysis
    # ============================================================
    print(f"{'='*70}")
    print("FEATURE CORRELATIONS WITH GT ACCURACY")
    print(f"{'='*70}")
    for j, name in enumerate(FEATURE_NAMES):
        corr, pval = stats.pearsonr(X[:, j], y_acc)
        sp, _ = stats.spearmanr(X[:, j], y_acc)
        print(f"  {name:22s}  Pearson={corr:+.3f}  Spearman={sp:+.3f}")

    # ============================================================
    # Model Training & Evaluation
    # ============================================================
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y_acc, np.arange(len(X)), test_size=0.3, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Model 1: Ridge Regression
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_train_s, y_train)
    y_pred_ridge = np.clip(ridge.predict(X_test_s), 0, 100)

    # Model 2: Gradient Boosting
    gb = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    gb.fit(X_train, y_train)
    y_pred_gb = np.clip(gb.predict(X_test), 0, 100)

    for name, y_pred in [("Ridge Regression", y_pred_ridge), ("Gradient Boosting", y_pred_gb)]:
        mae = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        corr, _ = stats.pearsonr(y_test, y_pred)
        sp, _ = stats.spearmanr(y_test, y_pred)
        abs_err = np.abs(y_test - y_pred)

        print(f"\n{'='*70}")
        print(f"{name} - TEST RESULTS ({len(X_test)} samples)")
        print(f"{'='*70}")
        print(f"  MAE:      {mae:.2f}")
        print(f"  RMSE:     {rmse:.2f}")
        print(f"  Pearson:  {corr:.3f}")
        print(f"  Spearman: {sp:.3f}")
        print(f"  |err| < 5:  {np.mean(abs_err < 5)*100:.1f}%")
        print(f"  |err| < 10: {np.mean(abs_err < 10)*100:.1f}%")
        print(f"  |err| < 15: {np.mean(abs_err < 15)*100:.1f}%")
        print(f"  |err| < 20: {np.mean(abs_err < 20)*100:.1f}%")

    # Feature importance from GB
    print(f"\n{'='*70}")
    print("GRADIENT BOOSTING FEATURE IMPORTANCE")
    print(f"{'='*70}")
    importances = gb.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for idx in sorted_idx:
        print(f"  {FEATURE_NAMES[idx]:22s}  {importances[idx]:.4f}")

    # Show example predictions
    print(f"\n{'='*70}")
    print("EXAMPLE PREDICTIONS (GB)")
    print(f"{'='*70}")
    abs_err_gb = np.abs(y_test - y_pred_gb)
    sorted_by_err = np.argsort(abs_err_gb)

    print("\n  BEST 10:")
    for idx in sorted_by_err[:10]:
        orig_idx = idx_test[idx]
        print(f"    GT={y_test[idx]:5.1f}  Pred={y_pred_gb[idx]:5.1f}  "
              f"err={y_test[idx]-y_pred_gb[idx]:+5.1f}  "
              f"text='{records[orig_idx]['content'][:30]}'")

    print("\n  WORST 10:")
    for idx in sorted_by_err[-10:]:
        orig_idx = idx_test[idx]
        print(f"    GT={y_test[idx]:5.1f}  Pred={y_pred_gb[idx]:5.1f}  "
              f"err={y_test[idx]-y_pred_gb[idx]:+5.1f}  "
              f"ASR='{all_asr[orig_idx][:30]}'  "
              f"text='{records[orig_idx]['content'][:30]}'")

    # ============================================================
    # Integrity Prediction
    # ============================================================
    print(f"\n{'='*70}")
    print("INTEGRITY PREDICTION (Gradient Boosting)")
    print(f"{'='*70}")

    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
        X, y_int, test_size=0.3, random_state=42
    )
    gb_int = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    gb_int.fit(X_train_i, y_train_i)
    y_pred_int = np.clip(gb_int.predict(X_test_i), 0, 100)

    mae_int = np.mean(np.abs(y_test_i - y_pred_int))
    corr_int, _ = stats.pearsonr(y_test_i, y_pred_int)
    sp_int, _ = stats.spearmanr(y_test_i, y_pred_int)
    print(f"  MAE:      {mae_int:.2f}")
    print(f"  Pearson:  {corr_int:.3f}")
    print(f"  Spearman: {sp_int:.3f}")


if __name__ == "__main__":
    main()
