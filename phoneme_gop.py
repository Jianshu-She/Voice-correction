"""
Phoneme-Level GOP (Goodness of Pronunciation) Pipeline
=======================================================
Uses:
1. G2P (grapheme-to-phoneme) to get expected ARPAbet phoneme sequence
2. wav2vec2 phoneme recognition model for frame-level phoneme posteriors
3. Forced alignment at phoneme level
4. GOP scoring per phoneme: log P(target) - log P(best_other)
5. Compare with ground truth pherr labels
"""

import torch
import torchaudio
import torchaudio.functional as F_audio
import numpy as np
import re
import openpyxl
from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
from g2p_en import G2p
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1. Load phoneme-level acoustic model
# ============================================================
print("Loading wav2vec2 phoneme model (espeak)...")
import json
from huggingface_hub import hf_hub_download

# Load feature extractor and model separately to avoid phonemizer dependency
feat_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
phone_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft").to(device)
phone_model.eval()

# Load vocabulary directly
vocab_path = hf_hub_download("facebook/wav2vec2-xlsr-53-espeak-cv-ft", "vocab.json")
with open(vocab_path) as f:
    vocab = json.load(f)

idx2phone = {v: k for k, v in vocab.items()}
phone2idx = vocab
n_phones = len(vocab)
print(f"Phoneme vocab size: {n_phones}")
print(f"Sample phones: {list(vocab.keys())[4:34]}")

BLANK_IDX = vocab.get("<pad>", 0)
print(f"Blank/PAD index: {BLANK_IDX}")

# ============================================================
# 2. G2P + ARPAbet mapping
# ============================================================
g2p = G2p()

# IPA to ARPAbet mapping (approximate)
# The espeak model outputs IPA, we need to map to ARPAbet for comparison
IPA_TO_ARPABET = {
    # Vowels
    "ɑː": "aa", "ɑ": "aa", "æ": "ae", "ʌ": "ah", "ɔː": "ao", "ɔ": "ao",
    "aʊ": "aw", "ə": "ax", "aɪ": "ay",
    "ɛ": "eh", "ɝ": "er", "ɜː": "er", "eɪ": "ey",
    "ɪ": "ih", "iː": "iy", "i": "iy",
    "oʊ": "ow", "ɔɪ": "oy",
    "ʊ": "uh", "uː": "uw", "u": "uw",
    # Consonants
    "b": "b", "tʃ": "ch", "d": "d", "ð": "dh",
    "f": "f", "ɡ": "g", "g": "g", "h": "hh", "dʒ": "jh",
    "k": "k", "l": "l", "m": "m", "n": "n", "ŋ": "ng",
    "p": "p", "ɹ": "r", "r": "r", "s": "s", "ʃ": "sh",
    "t": "t", "θ": "th", "v": "v", "w": "w",
    "j": "y", "z": "z", "ʒ": "zh",
    # Additional
    "ɐ": "ax", "ᵻ": "ih", "ɾ": "d", "ʔ": "t",
    "ɒ": "aa", "ɜ": "er", "e": "eh", "o": "ow",
    "a": "aa", "iə": "ir", "eə": "ar", "ʊə": "ur",
}

# Map from ARPAbet (g2p output) to lowercase for matching
def g2p_to_arpabet(text: str) -> list:
    """Convert text to ARPAbet phoneme list using g2p_en."""
    phones = g2p(text)
    result = []
    for ph in phones:
        if ph == " ":
            result.append("|")  # word boundary
        elif ph.isalpha() or any(c.isdigit() for c in ph):
            # Strip stress markers (0,1,2)
            clean = re.sub(r"\d", "", ph).lower()
            if clean:
                result.append(clean)
    return result


def get_ipa_phoneme_sequence(text: str) -> list:
    """Get IPA phoneme sequence that matches the model's vocabulary."""
    # Use the processor's tokenizer directly on the text
    # But we need to handle this differently - the model expects audio input
    # We use G2P to get the expected phoneme sequence for forced alignment
    arpabet = g2p_to_arpabet(text)
    return arpabet


# ============================================================
# 3. Audio processing & phoneme posteriors
# ============================================================
SAMPLE_RATE = 16000

def load_audio(audio_path: str) -> torch.Tensor:
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = F_audio.resample(waveform, sr, SAMPLE_RATE)
    return waveform


@torch.no_grad()
def get_phone_emissions(waveform: torch.Tensor) -> torch.Tensor:
    """Get frame-level phoneme log-posteriors."""
    inputs = feat_extractor(waveform.squeeze(0).numpy(), sampling_rate=SAMPLE_RATE,
                            return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    logits = phone_model(input_values).logits
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs.squeeze(0).cpu()  # (T, n_phones)


@torch.no_grad()
def get_phone_free_decode(emissions: torch.Tensor) -> list:
    """Greedy decode to get the model's free phoneme prediction."""
    ids = emissions.argmax(dim=-1)
    # CTC decode: remove blanks and consecutive duplicates
    decoded = []
    prev = -1
    for idx in ids.tolist():
        if idx != BLANK_IDX and idx != prev:
            decoded.append(idx2phone.get(idx, "?"))
        prev = idx
    return decoded


# ============================================================
# 4. Forced Alignment at Phoneme Level
# ============================================================

def find_best_phone_idx(target_arpabet: str, vocab: dict) -> int:
    """Find the best matching phoneme index in the model's vocabulary."""
    # Direct match
    if target_arpabet in vocab:
        return vocab[target_arpabet]

    # Try IPA equivalents
    ARPABET_TO_IPA_CANDIDATES = {
        "aa": ["ɑː", "ɑ", "ɒ", "a"],
        "ae": ["æ"],
        "ah": ["ʌ", "ə"],
        "ao": ["ɔː", "ɔ"],
        "aw": ["aʊ"],
        "ax": ["ə", "ɐ"],
        "ay": ["aɪ"],
        "b": ["b"],
        "ch": ["tʃ"],
        "d": ["d"],
        "dh": ["ð"],
        "eh": ["ɛ", "e"],
        "er": ["ɜː", "ɝ", "ɜ"],
        "ey": ["eɪ"],
        "f": ["f"],
        "g": ["ɡ", "g"],
        "hh": ["h"],
        "ih": ["ɪ", "ᵻ"],
        "iy": ["iː", "i"],
        "ir": ["ɪə", "ɪɹ"],
        "jh": ["dʒ"],
        "k": ["k"],
        "l": ["l"],
        "m": ["m"],
        "n": ["n"],
        "ng": ["ŋ"],
        "ow": ["oʊ", "o"],
        "oy": ["ɔɪ"],
        "p": ["p"],
        "r": ["ɹ", "r"],
        "s": ["s"],
        "sh": ["ʃ"],
        "t": ["t"],
        "th": ["θ"],
        "uh": ["ʊ"],
        "uw": ["uː", "u"],
        "ur": ["ʊə", "ʊɹ"],
        "v": ["v"],
        "w": ["w"],
        "y": ["j"],
        "z": ["z"],
        "zh": ["ʒ"],
        "ar": ["ɑːɹ", "ɑɹ"],
        "oo": ["ʊ", "uː"],
        "dr": ["dɹ"],
        "tr": ["tɹ"],
        "ts": ["ts"],
        "dz": ["dz"],
    }

    candidates = ARPABET_TO_IPA_CANDIDATES.get(target_arpabet, [])
    for ipa in candidates:
        if ipa in vocab:
            return vocab[ipa]
    return -1


def forced_align_phones(emissions: torch.Tensor, phone_indices: list) -> list:
    """Viterbi forced alignment at phoneme level."""
    T, C = emissions.shape
    S = len(phone_indices)
    if S == 0 or T < S:
        return []

    # Extended with blanks
    extended = [BLANK_IDX]
    for p in phone_indices:
        extended.append(p)
        extended.append(BLANK_IDX)
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
            if s > 1 and extended[s] != BLANK_IDX and extended[s] != extended[s-2]:
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
    for t in range(T-1, -1, -1):
        path.append((t, extended[s]))
        s = bp[t][s]
    path.reverse()
    return path


# ============================================================
# 5. Phoneme-Level GOP Scoring
# ============================================================

def compute_phoneme_gop(audio_path: str, text: str) -> list:
    """
    Compute GOP score for each phoneme in the expected text.
    Returns list of dicts: {arpabet, gop_score, log_posterior, predicted_as}
    """
    # 1. Get expected phoneme sequence via G2P
    arpabet_phones = g2p_to_arpabet(text)
    # Filter out word boundaries for alignment
    phone_list = [p for p in arpabet_phones if p != "|"]

    if not phone_list:
        return []

    # 2. Map ARPAbet to model vocab indices
    phone_indices = []
    valid_phones = []
    for ph in phone_list:
        idx = find_best_phone_idx(ph, phone2idx)
        if idx >= 0:
            phone_indices.append(idx)
            valid_phones.append(ph)

    if not phone_indices:
        return []

    # 3. Get phoneme emissions
    waveform = load_audio(audio_path)
    emissions = get_phone_emissions(waveform)
    T, C = emissions.shape

    # 4. Forced alignment
    path = forced_align_phones(emissions, phone_indices)
    if not path:
        return []

    # 5. Compute GOP per phoneme segment
    # Group frames by non-blank phoneme
    segments = []
    current_idx = None
    current_frames = []

    for frame, tok_idx in path:
        if tok_idx == BLANK_IDX:
            if current_idx is not None:
                segments.append((current_idx, current_frames))
                current_idx = None
                current_frames = []
            continue
        if tok_idx != current_idx:
            if current_idx is not None:
                segments.append((current_idx, current_frames))
            current_idx = tok_idx
            current_frames = [frame]
        else:
            current_frames.append(frame)

    if current_idx is not None:
        segments.append((current_idx, current_frames))

    # Match segments to expected phones
    results = []
    for i, (expected_ph, expected_idx) in enumerate(zip(valid_phones, phone_indices)):
        if i >= len(segments):
            results.append({
                "arpabet": expected_ph,
                "gop_score": -20.0,
                "log_posterior": -20.0,
                "predicted_phone": "?",
                "pherr_predicted": 1,
            })
            continue

        seg_idx, seg_frames = segments[i]
        frame_emissions = emissions[seg_frames]  # (n_frames, C)

        # Log posterior of target phone
        target_log_post = frame_emissions[:, expected_idx].mean().item()

        # Best competing phone (excluding blank and target)
        mask = torch.ones(C, dtype=torch.bool)
        mask[BLANK_IDX] = False
        mask[expected_idx] = False
        if mask.any():
            other_log_posts = frame_emissions[:, mask]
            best_other_val = other_log_posts.max(dim=-1).values.mean().item()
            best_other_idx = other_log_posts.mean(dim=0).argmax().item()
            # Map back to actual index
            all_indices = torch.arange(C)[mask]
            predicted_idx = all_indices[best_other_idx].item()
            predicted_phone = idx2phone.get(predicted_idx, "?")
        else:
            best_other_val = -30.0
            predicted_phone = "?"

        gop = target_log_post - best_other_val

        # Actually predicted phone (including target)
        actual_best_idx = frame_emissions.mean(dim=0).argmax().item()
        actual_predicted = idx2phone.get(actual_best_idx, "?")

        results.append({
            "arpabet": expected_ph,
            "gop_score": round(gop, 3),
            "log_posterior": round(target_log_post, 3),
            "predicted_phone": actual_predicted,
            "pherr_predicted": 1 if gop < -1.5 else 0,  # threshold
        })

    return results


# ============================================================
# 6. Load Ground Truth with Phoneme-Level Labels
# ============================================================

def load_gt_phonemes(xlsx_path: str) -> list:
    """Load ground truth with phoneme-level pherr labels."""
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

        # Extract phoneme-level data
        # Pattern: "char":"PHONE","ph2alpha":"LETTER",...,"pherr":0/1,...,"score":XX
        phones_data = re.findall(
            r'"char":"([^"]+)","ph2alpha":"([^"]*)".*?"pherr":(\d+).*?"score":([\d.]+)',
            s
        )

        gt_phones = []
        for ph_char, ph_alpha, pherr, score in phones_data:
            gt_phones.append({
                "phone": ph_char.lower(),
                "alpha": ph_alpha,
                "pherr": int(pherr),
                "score": float(score),
            })

        records.append({
            "file_name": fname,
            "content": content,
            "gt_accuracy": float(acc.group(1)) if acc else None,
            "gt_phones": gt_phones,
        })

    return records


# ============================================================
# 7. Main Evaluation
# ============================================================

def main():
    audio_dir = Path("/mnt/weka/home/jianshu.she/Voice-correction/audio_files")
    xlsx_path = "/mnt/weka/home/jianshu.she/Voice-correction/eval_log.xlsx"

    print("\nLoading ground truth...")
    records = load_gt_phonemes(xlsx_path)
    print(f"Loaded {len(records)} records")

    # Test on first 50 samples
    N = 50
    all_gt_pherr = []
    all_pred_pherr = []
    all_gop_scores = []
    all_gt_scores = []

    print(f"\nProcessing {N} samples...\n")
    for i, rec in enumerate(records[:N]):
        audio_path = audio_dir / rec["file_name"]
        if not audio_path.exists():
            continue

        try:
            phone_results = compute_phoneme_gop(str(audio_path), rec["content"])

            if not phone_results or not rec["gt_phones"]:
                continue

            # Compare predicted vs GT
            gt_phones = rec["gt_phones"]
            n_gt = len(gt_phones)
            n_pred = len(phone_results)

            # Collect phone-level stats
            for j, pr in enumerate(phone_results):
                all_gop_scores.append(pr["gop_score"])
                all_pred_pherr.append(pr["pherr_predicted"])

            for gp in gt_phones:
                all_gt_pherr.append(gp["pherr"])
                all_gt_scores.append(gp["score"])

            if i < 5:
                print(f"  [{i+1}] text='{rec['content'][:35]}' GT_acc={rec['gt_accuracy']}")
                print(f"       GT phones ({n_gt}): ", end="")
                for gp in gt_phones[:8]:
                    err = "*" if gp["pherr"] else " "
                    print(f"{gp['phone']:4s}({gp['score']:3.0f}){err}", end=" ")
                print("...")
                print(f"       Pred phones ({n_pred}): ", end="")
                for pr in phone_results[:8]:
                    err = "*" if pr["pherr_predicted"] else " "
                    print(f"{pr['arpabet']:4s}({pr['gop_score']:+5.1f}){err}", end=" ")
                print("...\n")

        except Exception as e:
            if i < 5:
                print(f"  [{i+1}] ERROR: {e}")

    # ============================================================
    # Analysis
    # ============================================================
    print(f"\n{'='*70}")
    print("PHONEME-LEVEL ANALYSIS")
    print(f"{'='*70}")
    print(f"  Total GT phonemes collected: {len(all_gt_pherr)}")
    print(f"  Total predicted phonemes:    {len(all_pred_pherr)}")
    print(f"  GT error rate:               {np.mean(all_gt_pherr)*100:.1f}%")
    print(f"  Pred error rate:             {np.mean(all_pred_pherr)*100:.1f}%")

    # GOP score distribution
    gop_arr = np.array(all_gop_scores)
    print(f"\n  GOP score distribution:")
    for pct in [5, 10, 25, 50, 75, 90, 95]:
        print(f"    P{pct:2d}: {np.percentile(gop_arr, pct):+.2f}")

    # Try different thresholds for pherr prediction
    print(f"\n  Threshold analysis (predicting pherr from GOP):")
    print(f"  {'Threshold':>10s}  {'Precision':>10s}  {'Recall':>10s}  {'F1':>10s}  {'Pred_err%':>10s}")

    # Note: we can only do this if we align GT and pred phonemes
    # Since they may not be the same length, we'll analyze the distributions separately
    # and focus on GOP score vs GT phone score correlation

    gt_scores_arr = np.array(all_gt_scores)
    print(f"\n  GT phone score distribution:")
    print(f"    mean={gt_scores_arr.mean():.1f}, std={gt_scores_arr.std():.1f}")
    print(f"    pherr=1 mean score: {gt_scores_arr[np.array(all_gt_pherr)==1].mean():.1f}")
    print(f"    pherr=0 mean score: {gt_scores_arr[np.array(all_gt_pherr)==0].mean():.1f}")

    # GOP score for predicted errors vs non-errors
    print(f"\n  GOP score by predicted error status:")
    pred_arr = np.array(all_pred_pherr)
    if pred_arr.sum() > 0:
        print(f"    pred pherr=1 mean GOP: {gop_arr[pred_arr==1].mean():.2f}")
    if (1-pred_arr).sum() > 0:
        print(f"    pred pherr=0 mean GOP: {gop_arr[pred_arr==0].mean():.2f}")

    # Free decode comparison
    print(f"\n{'='*70}")
    print("DETAILED EXAMPLES")
    print(f"{'='*70}")
    for i, rec in enumerate(records[:5]):
        audio_path = audio_dir / rec["file_name"]
        if not audio_path.exists():
            continue
        try:
            waveform = load_audio(str(audio_path))
            emissions = get_phone_emissions(waveform)
            free_phones = get_phone_free_decode(emissions)
            expected_arpabet = g2p_to_arpabet(rec["content"])

            print(f"\n  text: '{rec['content']}'")
            print(f"  Expected ARPAbet: {[p for p in expected_arpabet if p != '|']}")
            print(f"  Free decode:      {free_phones}")
            print(f"  GT phones:        {[p['phone'] for p in rec['gt_phones']]}")
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
