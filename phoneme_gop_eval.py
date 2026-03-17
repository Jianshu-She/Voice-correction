"""
Phoneme-Level GOP Evaluation - Full Analysis
=============================================
Runs on all 1000 samples, evaluates:
1. Phoneme-level error detection (pherr prediction via GOP threshold)
2. Phoneme-level score regression (GOP → phone score mapping)
3. Per-phoneme analysis: which phonemes are hardest to assess
"""

import torch
import torchaudio
import torchaudio.functional as F_audio
import numpy as np
import re
import json
import openpyxl
from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
from huggingface_hub import hf_hub_download
from g2p_en import G2p
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ============================================================
# Model Setup
# ============================================================
print("Loading wav2vec2 phoneme model...")
feat_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
phone_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft").to(device)
phone_model.eval()

vocab_path = hf_hub_download("facebook/wav2vec2-xlsr-53-espeak-cv-ft", "vocab.json")
with open(vocab_path) as f:
    vocab = json.load(f)
idx2phone = {v: k for k, v in vocab.items()}
phone2idx = vocab
BLANK_IDX = vocab.get("<pad>", 0)
SAMPLE_RATE = 16000

g2p = G2p()
print(f"Model loaded. Vocab: {len(vocab)} phonemes\n")

# ============================================================
# ARPAbet → IPA mapping
# ============================================================
ARPABET_TO_IPA = {
    "aa": ["ɑː", "ɑ", "ɒ", "a"], "ae": ["æ"], "ah": ["ʌ", "ə", "ɐ"],
    "ao": ["ɔː", "ɔ", "ɒ"], "aw": ["aʊ"], "ax": ["ə", "ɐ", "ʌ"], "ay": ["aɪ"],
    "b": ["b"], "ch": ["tʃ"], "d": ["d"], "dh": ["ð"],
    "eh": ["ɛ", "e"], "er": ["ɜː", "ɝ", "ɚ", "ɜ"], "ey": ["eɪ"],
    "f": ["f"], "g": ["ɡ", "g"], "hh": ["h"],
    "ih": ["ɪ", "ᵻ"], "iy": ["iː", "i"],
    "ir": ["ɪɹ"], "jh": ["dʒ"], "k": ["k"], "l": ["l"],
    "m": ["m"], "n": ["n"], "ng": ["ŋ"],
    "ow": ["oʊ", "o", "əʊ"], "oy": ["ɔɪ"],
    "p": ["p"], "r": ["ɹ", "r"], "s": ["s"], "sh": ["ʃ"],
    "t": ["t"], "th": ["θ"], "uh": ["ʊ"], "uw": ["uː", "u"],
    "ur": ["ʊɹ"], "v": ["v"], "w": ["w"], "y": ["j"],
    "z": ["z"], "zh": ["ʒ"],
    "ar": ["ɑːɹ"], "oo": ["ʊ", "uː"], "dr": ["dɹ"], "tr": ["tɹ"],
    "ts": ["ts"], "dz": ["dz"],
}

def arpabet_to_model_idx(ph: str) -> int:
    candidates = ARPABET_TO_IPA.get(ph, [])
    for ipa in candidates:
        if ipa in phone2idx:
            return phone2idx[ipa]
    if ph in phone2idx:
        return phone2idx[ph]
    return -1

# ============================================================
# Audio & Emissions
# ============================================================
def load_audio(path):
    w, sr = torchaudio.load(path)
    if w.shape[0] > 1: w = w.mean(0, keepdim=True)
    if sr != SAMPLE_RATE: w = F_audio.resample(w, sr, SAMPLE_RATE)
    return w

@torch.no_grad()
def get_emissions(waveform):
    inputs = feat_extractor(waveform.squeeze(0).numpy(), sampling_rate=SAMPLE_RATE,
                            return_tensors="pt", padding=True)
    logits = phone_model(inputs.input_values.to(device)).logits
    return torch.log_softmax(logits, dim=-1).squeeze(0).cpu()

# ============================================================
# G2P
# ============================================================
def text_to_arpabet(text):
    phones = g2p(text)
    result = []
    for ph in phones:
        if ph == " ":
            result.append("|")
        elif ph.isalpha() or any(c.isdigit() for c in ph):
            clean = re.sub(r"\d", "", ph).lower()
            if clean:
                result.append(clean)
    return result

# ============================================================
# Forced Alignment (Viterbi)
# ============================================================
def forced_align(emissions, phone_indices):
    T, C = emissions.shape
    S = len(phone_indices)
    if S == 0 or T < S: return []

    extended = [BLANK_IDX]
    for p in phone_indices:
        extended.append(p)
        extended.append(BLANK_IDX)
    S_ext = len(extended)
    if T < 2: return []

    NEG_INF = float("-inf")
    dp = np.full((T, S_ext), NEG_INF, dtype=np.float64)
    bp = np.zeros((T, S_ext), dtype=np.int32)
    dp[0][0] = emissions[0, extended[0]].item()
    if S_ext > 1: dp[0][1] = emissions[0, extended[1]].item()

    for t in range(1, T):
        for s in range(S_ext):
            emit = emissions[t, extended[s]].item()
            best, best_s = dp[t-1][s], s
            if s > 0 and dp[t-1][s-1] > best:
                best, best_s = dp[t-1][s-1], s-1
            if s > 1 and extended[s] != BLANK_IDX and extended[s] != extended[s-2]:
                if dp[t-1][s-2] > best:
                    best, best_s = dp[t-1][s-2], s-2
            dp[t][s] = best + emit
            bp[t][s] = best_s

    s = S_ext-1 if (S_ext >= 2 and dp[T-1][S_ext-1] >= dp[T-1][S_ext-2]) else max(S_ext-2, 0)
    path = []
    for t in range(T-1, -1, -1):
        path.append((t, extended[s]))
        s = bp[t][s]
    path.reverse()
    return path

# ============================================================
# Compute GOP per phoneme
# ============================================================
def compute_gop(audio_path, text):
    arpabet = [p for p in text_to_arpabet(text) if p != "|"]
    if not arpabet: return []

    indices = []
    valid = []
    for ph in arpabet:
        idx = arpabet_to_model_idx(ph)
        if idx >= 0:
            indices.append(idx)
            valid.append(ph)
    if not indices: return []

    waveform = load_audio(audio_path)
    emissions = get_emissions(waveform)
    path = forced_align(emissions, indices)
    if not path: return []

    # Group by phoneme
    segments = []
    cur_tok, cur_frames = None, []
    for f, tok in path:
        if tok == BLANK_IDX:
            if cur_tok is not None:
                segments.append((cur_tok, cur_frames))
                cur_tok, cur_frames = None, []
            continue
        if tok != cur_tok:
            if cur_tok is not None:
                segments.append((cur_tok, cur_frames))
            cur_tok, cur_frames = tok, [f]
        else:
            cur_frames.append(f)
    if cur_tok is not None:
        segments.append((cur_tok, cur_frames))

    results = []
    for i, (ph, expected_idx) in enumerate(zip(valid, indices)):
        if i >= len(segments):
            results.append({"arpabet": ph, "gop": -20.0, "log_post": -20.0})
            continue
        _, frames = segments[i]
        frame_em = emissions[frames]
        target_lp = frame_em[:, expected_idx].mean().item()

        mask = torch.ones(len(vocab), dtype=torch.bool)
        mask[BLANK_IDX] = False
        mask[expected_idx] = False
        best_other = frame_em[:, mask].max(dim=-1).values.mean().item() if mask.any() else -30.0
        gop = target_lp - best_other

        results.append({"arpabet": ph, "gop": gop, "log_post": target_lp})
    return results

# ============================================================
# Load GT
# ============================================================
def load_gt(xlsx_path):
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb.active
    records = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        fname, content, raw = row
        if not fname or not content: continue
        s = raw
        for _ in range(5): s = s.replace("\\\\", "\\")
        s = s.replace('\\"', '"')
        acc = re.search(r'"accuracy":\s*([\d.]+)', s)
        phones = re.findall(r'"char":"([^"]+)","ph2alpha":"([^"]*)".*?"pherr":(\d+).*?"score":([\d.]+)', s)
        gt_phones = [{"phone": p.lower(), "pherr": int(e), "score": float(sc)} for p, _, e, sc in phones]
        records.append({
            "file_name": fname, "content": content,
            "gt_accuracy": float(acc.group(1)) if acc else 0,
            "gt_phones": gt_phones,
        })
    return records

# ============================================================
# Main
# ============================================================
def main():
    audio_dir = Path("/mnt/weka/home/jianshu.she/Voice-correction/audio_files")
    records = load_gt("/mnt/weka/home/jianshu.she/Voice-correction/eval_log.xlsx")
    print(f"Loaded {len(records)} records\n")

    # Collect all phoneme-level data
    all_gop = []       # GOP scores for predicted phonemes
    all_gt_pherr = []  # GT pherr labels (aligned by position)
    all_gt_score = []  # GT phone scores
    per_phone_gop = {}  # GOP scores grouped by ARPAbet phone type
    per_phone_gt = {}   # GT data grouped by phone type
    n_processed = 0
    n_error = 0

    for i, rec in enumerate(records):
        audio_path = audio_dir / rec["file_name"]
        if not audio_path.exists(): continue

        try:
            pred = compute_gop(str(audio_path), rec["content"])
            gt = rec["gt_phones"]

            if not pred or not gt:
                continue

            # Align by position (take min length)
            n = min(len(pred), len(gt))
            for j in range(n):
                gop_val = pred[j]["gop"]
                gt_err = gt[j]["pherr"]
                gt_sc = gt[j]["score"]
                ph_type = pred[j]["arpabet"]

                all_gop.append(gop_val)
                all_gt_pherr.append(gt_err)
                all_gt_score.append(gt_sc)

                if ph_type not in per_phone_gop:
                    per_phone_gop[ph_type] = []
                    per_phone_gt[ph_type] = []
                per_phone_gop[ph_type].append(gop_val)
                per_phone_gt[ph_type].append((gt_err, gt_sc))

            n_processed += 1

        except Exception as e:
            n_error += 1

        if (i+1) % 100 == 0:
            print(f"  Processed {i+1}/{len(records)}  (valid={n_processed}, errors={n_error})")

    all_gop = np.array(all_gop)
    all_gt_pherr = np.array(all_gt_pherr)
    all_gt_score = np.array(all_gt_score)

    print(f"\n{'='*70}")
    print(f"OVERALL STATISTICS")
    print(f"{'='*70}")
    print(f"  Aligned phoneme pairs: {len(all_gop)}")
    print(f"  GT error rate (pherr=1): {all_gt_pherr.mean()*100:.1f}%")
    print(f"  GOP score range: [{all_gop.min():.2f}, {all_gop.max():.2f}]")

    # GOP vs GT score correlation
    corr, pval = stats.pearsonr(all_gop, all_gt_score)
    sp, _ = stats.spearmanr(all_gop, all_gt_score)
    print(f"\n  GOP vs GT phone score:")
    print(f"    Pearson:  {corr:.3f} (p={pval:.2e})")
    print(f"    Spearman: {sp:.3f}")

    # GOP distribution by GT error status
    gop_err = all_gop[all_gt_pherr == 1]
    gop_ok = all_gop[all_gt_pherr == 0]
    print(f"\n  GOP distribution by GT pherr:")
    print(f"    pherr=0 (correct): mean={gop_ok.mean():.2f}, std={gop_ok.std():.2f}, median={np.median(gop_ok):.2f}")
    print(f"    pherr=1 (error):   mean={gop_err.mean():.2f}, std={gop_err.std():.2f}, median={np.median(gop_err):.2f}")

    # ============================================================
    # Threshold search for binary error detection
    # ============================================================
    print(f"\n{'='*70}")
    print(f"THRESHOLD SEARCH FOR PHERR DETECTION")
    print(f"{'='*70}")
    print(f"  {'Threshold':>10s}  {'Prec':>6s}  {'Recall':>6s}  {'F1':>6s}  {'AUC':>6s}")

    best_f1 = 0
    best_thresh = 0
    for thresh in np.arange(-8.0, 4.0, 0.5):
        pred_err = (all_gop < thresh).astype(int)
        if pred_err.sum() == 0 or pred_err.sum() == len(pred_err):
            continue
        prec, rec, f1, _ = precision_recall_fscore_support(all_gt_pherr, pred_err, average="binary", zero_division=0)
        print(f"  {thresh:>10.1f}  {prec:>6.3f}  {rec:>6.3f}  {f1:>6.3f}")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    # AUC
    try:
        auc = roc_auc_score(all_gt_pherr, -all_gop)  # negate: lower GOP = more likely error
        print(f"\n  AUC-ROC: {auc:.3f}")
    except:
        auc = 0

    print(f"\n  Best threshold: {best_thresh:.1f} (F1={best_f1:.3f})")

    # Detailed metrics at best threshold
    pred_best = (all_gop < best_thresh).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(all_gt_pherr, pred_best, average="binary")
    tp = ((pred_best == 1) & (all_gt_pherr == 1)).sum()
    fp = ((pred_best == 1) & (all_gt_pherr == 0)).sum()
    fn = ((pred_best == 0) & (all_gt_pherr == 1)).sum()
    tn = ((pred_best == 0) & (all_gt_pherr == 0)).sum()
    print(f"\n  Confusion matrix at threshold={best_thresh:.1f}:")
    print(f"    TP={tp:4d}  FP={fp:4d}")
    print(f"    FN={fn:4d}  TN={tn:4d}")
    print(f"    Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")

    # ============================================================
    # Per-phoneme analysis
    # ============================================================
    print(f"\n{'='*70}")
    print(f"PER-PHONEME ANALYSIS (top 20 by count)")
    print(f"{'='*70}")
    print(f"  {'Phone':>6s}  {'Count':>6s}  {'GT_err%':>8s}  {'mean_GOP':>9s}  {'corr':>6s}  {'sep':>6s}")

    phone_stats = []
    for ph in sorted(per_phone_gop.keys(), key=lambda x: -len(per_phone_gop[x])):
        gops = np.array(per_phone_gop[ph])
        gt_data = per_phone_gt[ph]
        gt_errs = np.array([d[0] for d in gt_data])
        gt_scores = np.array([d[1] for d in gt_data])

        if len(gops) < 10:
            continue

        gt_err_rate = gt_errs.mean() * 100
        mean_gop = gops.mean()

        # Correlation between GOP and GT score
        if gt_scores.std() > 0 and gops.std() > 0:
            corr_ph, _ = stats.pearsonr(gops, gt_scores)
        else:
            corr_ph = 0

        # Separation: mean GOP for correct - mean GOP for errors
        if gt_errs.sum() > 0 and (1-gt_errs).sum() > 0:
            sep = gops[gt_errs == 0].mean() - gops[gt_errs == 1].mean()
        else:
            sep = 0

        phone_stats.append((ph, len(gops), gt_err_rate, mean_gop, corr_ph, sep))

    for ph, cnt, err_rate, mg, c, s in phone_stats[:20]:
        print(f"  {ph:>6s}  {cnt:>6d}  {err_rate:>7.1f}%  {mg:>+8.2f}  {c:>+5.2f}  {s:>+5.2f}")

    # ============================================================
    # GOP → Score Regression (Linear)
    # ============================================================
    print(f"\n{'='*70}")
    print(f"GOP → PHONE SCORE REGRESSION")
    print(f"{'='*70}")

    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    X_gop = all_gop.reshape(-1, 1)
    y_score = all_gt_score

    X_tr, X_te, y_tr, y_te = train_test_split(X_gop, y_score, test_size=0.3, random_state=42)
    reg = LinearRegression().fit(X_tr, y_tr)
    y_pred = np.clip(reg.predict(X_te), 0, 100)

    mae = np.mean(np.abs(y_te - y_pred))
    rmse = np.sqrt(np.mean((y_te - y_pred) ** 2))
    corr_reg, _ = stats.pearsonr(y_te, y_pred)
    sp_reg, _ = stats.spearmanr(y_te, y_pred)

    print(f"  Linear: y = {reg.coef_[0]:.2f} * GOP + {reg.intercept_:.2f}")
    print(f"  MAE:      {mae:.2f}")
    print(f"  RMSE:     {rmse:.2f}")
    print(f"  Pearson:  {corr_reg:.3f}")
    print(f"  Spearman: {sp_reg:.3f}")

    abs_err = np.abs(y_te - y_pred)
    print(f"  |err| < 10: {np.mean(abs_err < 10)*100:.1f}%")
    print(f"  |err| < 20: {np.mean(abs_err < 20)*100:.1f}%")
    print(f"  |err| < 30: {np.mean(abs_err < 30)*100:.1f}%")


if __name__ == "__main__":
    main()
