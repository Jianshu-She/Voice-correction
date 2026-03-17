"""
Phoneme Comparison Approach
===========================
Instead of GOP scoring, directly compare:
  - Expected phonemes (from G2P on reference text)
  - Recognized phonemes (from wav2vec2 on each aligned segment)

For each aligned phoneme segment, we check what phoneme the model
actually thinks the speaker said, and compare with what they should have said.
"""

import json
import re
import warnings
from pathlib import Path

import numpy as np
import openpyxl
import torch
import torchaudio
import torchaudio.functional as F_audio
from g2p_en import G2p
from huggingface_hub import hf_hub_download
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from scipy import stats

warnings.filterwarnings("ignore")

SAMPLE_RATE = 16000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ============================================================
# IPA ↔ ARPAbet mappings
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

# Build reverse map: IPA → ARPAbet
IPA_TO_ARPABET = {}
for arpa, ipas in ARPABET_TO_IPA.items():
    for ipa in ipas:
        if ipa not in IPA_TO_ARPABET:
            IPA_TO_ARPABET[ipa] = arpa

# Phoneme similarity groups (phones that sound similar)
SIMILAR_GROUPS = {
    "vowel_open": {"aa", "ao", "ah", "ax"},
    "vowel_front": {"ae", "eh", "ih"},
    "vowel_high": {"iy", "ih"},
    "vowel_back": {"uw", "uh", "oo"},
    "diphthong_a": {"ay", "ey"},
    "diphthong_o": {"ow", "aw"},
    "stop_voice": {"b", "d", "g"},
    "stop_unvoice": {"p", "t", "k"},
    "fricative_s": {"s", "z", "sh", "zh"},
    "nasal": {"m", "n", "ng"},
    "rhotic": {"r", "er", "ar"},
}

def are_similar(ph1, ph2):
    """Check if two phones are in the same similarity group."""
    for group in SIMILAR_GROUPS.values():
        if ph1 in group and ph2 in group:
            return True
    return False


# ============================================================
# Model loading
# ============================================================
print("Loading phoneme model...")
feat_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
ctc_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft").to(device)
ctc_model.eval()

vocab_path = hf_hub_download("facebook/wav2vec2-xlsr-53-espeak-cv-ft", "vocab.json")
with open(vocab_path) as f:
    vocab = json.load(f)
idx2phone = {v: k for k, v in vocab.items()}
BLANK_IDX = vocab.get("<pad>", 0)

g2p = G2p()
print(f"Model loaded. Vocab: {len(vocab)} phones")


# ============================================================
# Core functions
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
    logits = ctc_model(inputs.input_values.to(device)).logits
    return torch.log_softmax(logits, dim=-1).squeeze(0).cpu()

def text_to_arpabet(text):
    phones = g2p(text)
    result = []
    for ph in phones:
        if ph == " ": continue
        clean = re.sub(r"\d", "", ph).lower()
        if clean: result.append(clean)
    return result

def arpabet_to_idx(ph):
    for ipa in ARPABET_TO_IPA.get(ph, []):
        if ipa in vocab: return vocab[ipa]
    return vocab.get(ph, -1)

def ipa_to_arpabet(ipa_phone):
    """Convert IPA phone to closest ARPAbet equivalent."""
    if ipa_phone in IPA_TO_ARPABET:
        return IPA_TO_ARPABET[ipa_phone]
    # Try partial matches
    for ipa, arpa in IPA_TO_ARPABET.items():
        if ipa_phone.startswith(ipa) or ipa.startswith(ipa_phone):
            return arpa
    return ipa_phone  # return as-is if no match

def viterbi_align(emissions, phone_indices):
    T, C = emissions.shape
    S = len(phone_indices)
    if S == 0 or T < S: return []
    extended = [BLANK_IDX]
    for p in phone_indices:
        extended.append(p)
        extended.append(BLANK_IDX)
    S_ext = len(extended)
    NEG_INF = float("-inf")
    dp = np.full((T, S_ext), NEG_INF, dtype=np.float64)
    bp = np.zeros((T, S_ext), dtype=np.int32)
    dp[0][0] = emissions[0, extended[0]].item()
    if S_ext > 1: dp[0][1] = emissions[0, extended[1]].item()
    for t in range(1, T):
        for s in range(S_ext):
            emit = emissions[t, extended[s]].item()
            best, best_s = dp[t-1][s], s
            if s > 0 and dp[t-1][s-1] > best: best, best_s = dp[t-1][s-1], s-1
            if s > 1 and extended[s] != BLANK_IDX and extended[s] != extended[s-2]:
                if dp[t-1][s-2] > best: best, best_s = dp[t-1][s-2], s-2
            dp[t][s] = best + emit
            bp[t][s] = best_s
    s = S_ext-1 if (S_ext >= 2 and dp[T-1][S_ext-1] >= dp[T-1][S_ext-2]) else max(S_ext-2, 0)
    path = []
    for t in range(T-1, -1, -1):
        path.append((t, extended[s]))
        s = bp[t][s]
    path.reverse()
    return path


def compare_phonemes(audio_path, text):
    """
    For each expected phoneme:
    1. Force-align to find its audio segment
    2. Look at what the model actually recognizes in that segment
    3. Compare expected vs recognized

    Returns list of:
        {expected, recognized, match, similar, gop, confidence}
    """
    waveform = load_audio(audio_path)
    emissions = get_emissions(waveform)

    arpabet = text_to_arpabet(text)
    indices, valid = [], []
    for ph in arpabet:
        idx = arpabet_to_idx(ph)
        if idx >= 0:
            indices.append(idx)
            valid.append(ph)
    if not indices: return []

    path = viterbi_align(emissions, indices)
    if not path: return []

    # Group frames by phoneme segment
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
    for i, (expected_ph, expected_idx) in enumerate(zip(valid, indices)):
        if i >= len(segments): break
        _, frames = segments[i]
        frame_em = emissions[frames]  # (n_frames, C)

        # What does the model think this segment is?
        # Average emissions over frames, then find top phones (excluding blank)
        avg_em = frame_em.mean(dim=0)  # (C,)
        avg_em[BLANK_IDX] = -float("inf")  # exclude blank

        top5_idx = avg_em.topk(5).indices.tolist()
        top5_prob = torch.softmax(avg_em, dim=0)[top5_idx].tolist()
        top5_phones = [(idx2phone.get(idx, "?"), prob) for idx, prob in zip(top5_idx, top5_prob)]

        # Best recognized phone (in ARPAbet)
        recognized_ipa = idx2phone.get(top5_idx[0], "?")
        recognized_arpa = ipa_to_arpabet(recognized_ipa)

        # GOP
        target_lp = frame_em[:, expected_idx].mean().item()
        mask = torch.ones(emissions.shape[1], dtype=torch.bool)
        mask[BLANK_IDX] = False
        mask[expected_idx] = False
        best_other = frame_em[:, mask].max(dim=-1).values.mean().item()
        gop = target_lp - best_other

        # Compare
        exact_match = (recognized_arpa == expected_ph)
        similar = are_similar(recognized_arpa, expected_ph) if not exact_match else True

        results.append({
            "expected": expected_ph,
            "recognized": recognized_arpa,
            "recognized_ipa": recognized_ipa,
            "match": exact_match,
            "similar": similar,
            "gop": round(gop, 2),
            "confidence": round(top5_prob[0], 3),
            "top3": [(ipa_to_arpabet(ph), round(p, 3)) for ph, p in top5_phones[:3]],
        })

    return results


# ============================================================
# Evaluation
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
        phones = re.findall(r'"char":"([^"]+)","ph2alpha":"([^"]*)".*?"pherr":(\d+).*?"score":([\d.]+)', s)
        gt = [{"phone": p.lower(), "pherr": int(e), "score": float(sc)} for p, _, e, sc in phones]
        records.append({"file_name": fname, "content": content, "gt_phones": gt})
    return records


def main():
    audio_dir = Path("/mnt/weka/home/jianshu.she/Voice-correction/audio_files")
    records = load_gt("/mnt/weka/home/jianshu.she/Voice-correction/eval_log.xlsx")
    print(f"Loaded {len(records)} records\n")

    # Show detailed examples first
    print(f"{'='*70}")
    print("DETAILED EXAMPLES")
    print(f"{'='*70}")

    for i in range(5):
        rec = records[i]
        audio_path = audio_dir / rec["file_name"]
        if not audio_path.exists(): continue

        result = compare_phonemes(str(audio_path), rec["content"])
        gt = rec["gt_phones"]
        n = min(len(result), len(gt))

        print(f"\n  Text: '{rec['content']}'")
        print(f"  {'Expected':>10s}  {'Heard':>10s}  {'Match':>6s}  {'GT_err':>6s}  {'GOP':>6s}  Top-3")
        for j in range(n):
            r = result[j]
            g = gt[j]
            match_str = "✓" if r["match"] else ("~" if r["similar"] else "✗")
            gt_err = "ERR" if g["pherr"] else "ok"
            top3_str = ", ".join(f"{ph}({p:.2f})" for ph, p in r["top3"])
            print(f"  /{r['expected']:>8s}/  /{r['recognized']:>8s}/  {match_str:>6s}  {gt_err:>6s}  {r['gop']:>+5.1f}  {top3_str}")

    # Full evaluation
    print(f"\n\n{'='*70}")
    print("FULL EVALUATION (all samples)")
    print(f"{'='*70}\n")

    all_gt_pherr = []
    all_mismatch = []   # 1 if expected != recognized
    all_not_similar = [] # 1 if not even similar
    all_gop = []
    n_total = 0

    for i, rec in enumerate(records):
        audio_path = audio_dir / rec["file_name"]
        if not audio_path.exists(): continue
        try:
            result = compare_phonemes(str(audio_path), rec["content"])
            gt = rec["gt_phones"]
            n = min(len(result), len(gt))
            for j in range(n):
                all_gt_pherr.append(gt[j]["pherr"])
                all_mismatch.append(0 if result[j]["match"] else 1)
                all_not_similar.append(0 if result[j]["similar"] else 1)
                all_gop.append(result[j]["gop"])
                n_total += 1
        except:
            pass
        if (i+1) % 200 == 0:
            print(f"  Processed {i+1}/{len(records)}...")

    gt = np.array(all_gt_pherr)
    mismatch = np.array(all_mismatch)
    not_similar = np.array(all_not_similar)
    gop = np.array(all_gop)

    print(f"\n  Total aligned phonemes: {n_total}")
    print(f"  GT error rate: {gt.mean()*100:.1f}%")
    print(f"  Mismatch rate (exact): {mismatch.mean()*100:.1f}%")
    print(f"  Mismatch rate (not similar): {not_similar.mean()*100:.1f}%")

    # Method 1: Exact mismatch as error predictor
    print(f"\n  --- Method 1: Exact mismatch → pherr ---")
    prec, rec_m, f1, _ = precision_recall_fscore_support(gt, mismatch, average="binary", zero_division=0)
    auc = roc_auc_score(gt, mismatch) if len(np.unique(gt)) > 1 else 0
    print(f"  Precision={prec:.3f}  Recall={rec_m:.3f}  F1={f1:.3f}  AUC={auc:.3f}")

    # Method 2: Not-similar as error predictor
    print(f"\n  --- Method 2: Not similar → pherr ---")
    prec2, rec2, f12, _ = precision_recall_fscore_support(gt, not_similar, average="binary", zero_division=0)
    auc2 = roc_auc_score(gt, not_similar) if len(np.unique(gt)) > 1 else 0
    print(f"  Precision={prec2:.3f}  Recall={rec2:.3f}  F1={f12:.3f}  AUC={auc2:.3f}")

    # Method 3: Combined - mismatch + GOP
    print(f"\n  --- Method 3: Mismatch AND GOP < threshold ---")
    for thresh in [-4.0, -3.0, -2.5, -2.0, -1.5, -1.0, 0.0]:
        pred = ((mismatch == 1) & (gop < thresh)).astype(int)
        if pred.sum() == 0: continue
        p, r, f, _ = precision_recall_fscore_support(gt, pred, average="binary", zero_division=0)
        print(f"    GOP<{thresh:+.1f}: Prec={p:.3f} Rec={r:.3f} F1={f:.3f}")

    # Comparison table
    print(f"\n{'='*70}")
    print("COMPARISON WITH PREVIOUS METHODS")
    print(f"{'='*70}")
    print(f"  {'Method':<30s}  {'AUC':>6s}  {'F1':>6s}  {'Prec':>6s}  {'Rec':>6s}")
    print(f"  {'GOP threshold (v1.0)':<30s}  {'0.738':>6s}  {'0.476':>6s}  {'0.379':>6s}  {'0.638':>6s}")
    print(f"  {'E2E MLP (v2)':<30s}  {'0.814':>6s}  {'0.565':>6s}  {'0.500':>6s}  {'0.650':>6s}")
    print(f"  {'Phoneme mismatch (exact)':<30s}  {auc:>6.3f}  {f1:>6.3f}  {prec:>6.3f}  {rec_m:>6.3f}")
    print(f"  {'Phoneme mismatch (similar)':<30s}  {auc2:>6.3f}  {f12:>6.3f}  {prec2:>6.3f}  {rec2:>6.3f}")


if __name__ == "__main__":
    main()
