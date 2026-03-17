"""
End-to-End Phoneme Scorer
=========================
Uses wav2vec2 hidden states (not just CTC logits) to predict:
1. Phone-level score (0-100 regression)
2. Phone-level error flag (pherr binary classification)

Architecture:
  Audio → wav2vec2 (frozen backbone) → hidden states per frame
  → Forced alignment to locate phoneme segments
  → Mean-pool hidden states per phoneme segment → 768-dim vector
  → MLP head → (score, pherr)

Training:
  - Uses ground truth from eval_log.xlsx (phoneme-level labels)
  - wav2vec2 backbone frozen (only MLP head trained)
  - Multi-task loss: MSE for score + BCE for pherr
"""

import json
import re
import warnings
from pathlib import Path

import numpy as np
import openpyxl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as F_audio
from g2p_en import G2p
from huggingface_hub import hf_hub_download
from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import train_test_split
from scipy import stats
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

warnings.filterwarnings("ignore")

SAMPLE_RATE = 16000
HIDDEN_DIM = 1024  # wav2vec2-large hidden size

# ============================================================
# ARPAbet → IPA mapping (same as pipeline.py)
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


# ============================================================
# 1. Feature Extraction (wav2vec2 hidden states + alignment)
# ============================================================
class FeatureExtractor:
    """Extract per-phoneme hidden state vectors from wav2vec2."""

    def __init__(self, device=None):
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.g2p = G2p()

        # Load wav2vec2 backbone (for hidden states)
        print("Loading wav2vec2-large backbone...")
        self.backbone = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-large-960h"
        ).to(self.device)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Load phoneme CTC model (for forced alignment only)
        print("Loading phoneme CTC model (for alignment)...")
        from transformers import Wav2Vec2ForCTC
        self.ctc_model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
        ).to(self.device)
        self.ctc_model.eval()
        for p in self.ctc_model.parameters():
            p.requires_grad = False

        self.feat_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
        )

        vocab_path = hf_hub_download("facebook/wav2vec2-xlsr-53-espeak-cv-ft", "vocab.json")
        with open(vocab_path) as f:
            self.vocab = json.load(f)
        self.blank_idx = self.vocab.get("<pad>", 0)
        print(f"Models loaded. Device: {self.device}")

    def _load_audio(self, path):
        w, sr = torchaudio.load(path)
        if w.shape[0] > 1:
            w = w.mean(0, keepdim=True)
        if sr != SAMPLE_RATE:
            w = F_audio.resample(w, sr, SAMPLE_RATE)
        return w

    def _text_to_arpabet(self, text):
        phones = self.g2p(text)
        result = []
        for ph in phones:
            if ph == " ":
                continue
            clean = re.sub(r"\d", "", ph).lower()
            if clean:
                result.append(clean)
        return result

    def _arpabet_to_idx(self, ph):
        for ipa in ARPABET_TO_IPA.get(ph, []):
            if ipa in self.vocab:
                return self.vocab[ipa]
        return self.vocab.get(ph, -1)

    def _viterbi_align(self, emissions, phone_indices):
        T, C = emissions.shape
        S = len(phone_indices)
        if S == 0 or T < S:
            return []
        extended = [self.blank_idx]
        for p in phone_indices:
            extended.append(p)
            extended.append(self.blank_idx)
        S_ext = len(extended)
        NEG_INF = float("-inf")
        dp = np.full((T, S_ext), NEG_INF, dtype=np.float64)
        bp = np.zeros((T, S_ext), dtype=np.int32)
        dp[0][0] = emissions[0, extended[0]].item()
        if S_ext > 1:
            dp[0][1] = emissions[0, extended[1]].item()
        for t in range(1, T):
            for s in range(S_ext):
                emit = emissions[t, extended[s]].item()
                best, best_s = dp[t-1][s], s
                if s > 0 and dp[t-1][s-1] > best:
                    best, best_s = dp[t-1][s-1], s-1
                if s > 1 and extended[s] != self.blank_idx and extended[s] != extended[s-2]:
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

    @torch.no_grad()
    def extract(self, audio_path, text):
        """
        Extract per-phoneme features from audio.

        Returns list of dicts:
            {arpabet, hidden_state (768-dim), frame_range}
        """
        waveform = self._load_audio(audio_path)

        # 1. Get CTC emissions for alignment
        inputs = self.feat_extractor(
            waveform.squeeze(0).numpy(), sampling_rate=SAMPLE_RATE,
            return_tensors="pt", padding=True,
        )
        input_values = inputs.input_values.to(self.device)
        ctc_logits = self.ctc_model(input_values).logits
        emissions = torch.log_softmax(ctc_logits, dim=-1).squeeze(0).cpu()

        # 2. Get hidden states from backbone
        # wav2vec2-large-960h uses its own feature extractor
        backbone_inputs = self.backbone.feature_extractor(waveform.squeeze(0).to(self.device).unsqueeze(0))
        # Actually use the proper way
        from transformers import Wav2Vec2FeatureExtractor as FE
        fe2 = FE.from_pretrained("facebook/wav2vec2-large-960h")
        inputs2 = fe2(waveform.squeeze(0).numpy(), sampling_rate=SAMPLE_RATE,
                       return_tensors="pt", padding=True)
        hidden_out = self.backbone(inputs2.input_values.to(self.device))
        hidden_states = hidden_out.last_hidden_state.squeeze(0).cpu()  # (T2, 768)

        # 3. G2P + alignment
        arpabet = self._text_to_arpabet(text)
        indices = []
        valid_phones = []
        for ph in arpabet:
            idx = self._arpabet_to_idx(ph)
            if idx >= 0:
                indices.append(idx)
                valid_phones.append(ph)
        if not indices:
            return []

        path = self._viterbi_align(emissions, indices)
        if not path:
            return []

        # Group frames by phoneme
        segments = []
        cur_tok, cur_frames = None, []
        for f, tok in path:
            if tok == self.blank_idx:
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

        # 4. Map alignment frames to hidden state frames
        # CTC model and backbone may have different frame counts
        T_ctc = emissions.shape[0]
        T_hidden = hidden_states.shape[0]
        scale = T_hidden / T_ctc

        results = []
        for i, ph in enumerate(valid_phones):
            if i >= len(segments):
                break
            _, frames = segments[i]
            # Scale frame indices
            h_start = max(0, int(min(frames) * scale))
            h_end = min(T_hidden, int((max(frames) + 1) * scale))
            if h_end <= h_start:
                h_end = h_start + 1

            # Mean pool hidden states over this phoneme's frames
            h_vec = hidden_states[h_start:h_end].mean(dim=0)  # (768,)
            results.append({
                "arpabet": ph,
                "hidden_state": h_vec,
                "frame_range": (h_start, h_end),
            })

        return results


# ============================================================
# 2. Load Ground Truth
# ============================================================
def load_gt_phonemes(xlsx_path):
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb.active
    records = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        fname, content, raw = row
        if not fname or not content:
            continue
        s = raw
        for _ in range(5):
            s = s.replace("\\\\", "\\")
        s = s.replace('\\"', '"')
        phones = re.findall(
            r'"char":"([^"]+)","ph2alpha":"([^"]*)".*?"pherr":(\d+).*?"score":([\d.]+)', s
        )
        gt_phones = [{"phone": p.lower(), "pherr": int(e), "score": float(sc)} for p, _, e, sc in phones]
        records.append({"file_name": fname, "content": content, "gt_phones": gt_phones})
    return records


# ============================================================
# 3. Build Dataset
# ============================================================
def build_dataset(records, audio_dir, extractor, cache_path=None):
    """
    Extract features for all samples and pair with GT labels.
    Returns: X (N, 768), y_score (N,), y_pherr (N,), phone_types (N,)
    """
    if cache_path and Path(cache_path).exists():
        print(f"Loading cached features from {cache_path}")
        data = torch.load(cache_path, weights_only=False)
        return data["X"], data["y_score"], data["y_pherr"], data["phone_types"]

    all_X = []
    all_score = []
    all_pherr = []
    all_phone_types = []
    n_skip = 0

    for i, rec in enumerate(records):
        audio_path = audio_dir / rec["file_name"]
        if not audio_path.exists():
            n_skip += 1
            continue

        try:
            features = extractor.extract(str(audio_path), rec["content"])
            gt = rec["gt_phones"]

            n = min(len(features), len(gt))
            for j in range(n):
                all_X.append(features[j]["hidden_state"])
                all_score.append(gt[j]["score"])
                all_pherr.append(gt[j]["pherr"])
                all_phone_types.append(features[j]["arpabet"])

        except Exception as e:
            n_skip += 1

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(records)}] features={len(all_X)}, skipped={n_skip}")

    X = torch.stack(all_X)  # (N, 768)
    y_score = torch.tensor(all_score, dtype=torch.float32)
    y_pherr = torch.tensor(all_pherr, dtype=torch.float32)

    if cache_path:
        torch.save({
            "X": X, "y_score": y_score, "y_pherr": y_pherr,
            "phone_types": all_phone_types,
        }, cache_path)
        print(f"Cached features to {cache_path}")

    return X, y_score, y_pherr, all_phone_types


class PhonemeDataset(Dataset):
    def __init__(self, X, y_score, y_pherr):
        self.X = X
        self.y_score = y_score
        self.y_pherr = y_pherr

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_score[idx], self.y_pherr[idx]


# ============================================================
# 4. MLP Scorer Model
# ============================================================
class PhonemeScorer(nn.Module):
    """
    MLP that predicts phone score (0-100) and pherr (0/1) from hidden states.
    """
    def __init__(self, input_dim=HIDDEN_DIM, hidden_dim=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        # Score head (regression)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # Error head (binary classification)
        self.pherr_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        shared = self.shared(x)
        score = self.score_head(shared).squeeze(-1)  # (B,)
        pherr_logit = self.pherr_head(shared).squeeze(-1)  # (B,)
        return score, pherr_logit


# ============================================================
# 5. Training
# ============================================================
def train_model(X_train, y_score_train, y_pherr_train,
                X_val, y_score_val, y_pherr_val,
                epochs=50, batch_size=256, lr=1e-3, device=None):
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_ds = PhonemeDataset(X_train, y_score_train, y_pherr_train)
    val_ds = PhonemeDataset(X_val, y_score_val, y_pherr_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = PhonemeScorer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Class weight for pherr (imbalanced: ~22% errors)
    pos_weight = torch.tensor([(1 - y_pherr_train.mean()) / max(y_pherr_train.mean(), 0.01)]).to(device)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        for X_b, y_score_b, y_pherr_b in train_loader:
            X_b = X_b.to(device)
            y_score_b = y_score_b.to(device)
            y_pherr_b = y_pherr_b.to(device)

            pred_score, pred_pherr_logit = model(X_b)

            loss_score = F.mse_loss(pred_score, y_score_b)
            loss_pherr = F.binary_cross_entropy_with_logits(
                pred_pherr_logit, y_pherr_b, pos_weight=pos_weight
            )
            loss = loss_score / 100.0 + loss_pherr  # scale MSE down

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Validate
        model.eval()
        val_losses = []
        all_pred_score = []
        all_pred_pherr = []
        all_gt_score = []
        all_gt_pherr = []

        with torch.no_grad():
            for X_b, y_score_b, y_pherr_b in val_loader:
                X_b = X_b.to(device)
                y_score_b = y_score_b.to(device)
                y_pherr_b = y_pherr_b.to(device)

                pred_score, pred_pherr_logit = model(X_b)
                loss_score = F.mse_loss(pred_score, y_score_b)
                loss_pherr = F.binary_cross_entropy_with_logits(
                    pred_pherr_logit, y_pherr_b, pos_weight=pos_weight
                )
                loss = loss_score / 100.0 + loss_pherr
                val_losses.append(loss.item())

                all_pred_score.append(pred_score.cpu())
                all_pred_pherr.append(torch.sigmoid(pred_pherr_logit).cpu())
                all_gt_score.append(y_score_b.cpu())
                all_gt_pherr.append(y_pherr_b.cpu())

        val_loss = np.mean(val_losses)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            pred_s = torch.cat(all_pred_score).numpy()
            gt_s = torch.cat(all_gt_score).numpy()
            pred_p = torch.cat(all_pred_pherr).numpy()
            gt_p = torch.cat(all_gt_pherr).numpy()

            mae = np.mean(np.abs(gt_s - np.clip(pred_s, 0, 100)))
            corr, _ = stats.pearsonr(gt_s, pred_s)
            try:
                auc = roc_auc_score(gt_p, pred_p)
            except:
                auc = 0

            print(f"  Epoch {epoch+1:3d}  train_loss={np.mean(train_losses):.4f}  "
                  f"val_loss={val_loss:.4f}  "
                  f"score_MAE={mae:.1f}  score_r={corr:.3f}  pherr_AUC={auc:.3f}")

    # Load best model
    model.load_state_dict(best_state)
    return model


# ============================================================
# 6. Evaluation
# ============================================================
def evaluate(model, X_test, y_score_test, y_pherr_test, device=None):
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        X_t = X_test.to(device)
        pred_score, pred_pherr_logit = model(X_t)
        pred_score = pred_score.cpu().numpy()
        pred_pherr_prob = torch.sigmoid(pred_pherr_logit).cpu().numpy()

    gt_score = y_score_test.numpy()
    gt_pherr = y_pherr_test.numpy()

    pred_score_clipped = np.clip(pred_score, 0, 100)

    # Score regression metrics
    mae = np.mean(np.abs(gt_score - pred_score_clipped))
    rmse = np.sqrt(np.mean((gt_score - pred_score_clipped) ** 2))
    corr, _ = stats.pearsonr(gt_score, pred_score_clipped)
    sp, _ = stats.spearmanr(gt_score, pred_score_clipped)

    print(f"\n{'='*60}")
    print(f"PHONE SCORE PREDICTION (regression)")
    print(f"{'='*60}")
    print(f"  MAE:      {mae:.2f}")
    print(f"  RMSE:     {rmse:.2f}")
    print(f"  Pearson:  {corr:.3f}")
    print(f"  Spearman: {sp:.3f}")

    abs_err = np.abs(gt_score - pred_score_clipped)
    for t in [5, 10, 15, 20, 30]:
        print(f"  |err| < {t:2d}: {np.mean(abs_err < t)*100:.1f}%")

    # pherr classification metrics
    try:
        auc = roc_auc_score(gt_pherr, pred_pherr_prob)
    except:
        auc = 0

    print(f"\n{'='*60}")
    print(f"PHONE ERROR DETECTION (classification)")
    print(f"{'='*60}")
    print(f"  AUC-ROC:  {auc:.3f}")

    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        pred_bin = (pred_pherr_prob >= thresh).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(gt_pherr, pred_bin, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh

    pred_best = (pred_pherr_prob >= best_thresh).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(gt_pherr, pred_best, average="binary")
    tp = ((pred_best == 1) & (gt_pherr == 1)).sum()
    fp = ((pred_best == 1) & (gt_pherr == 0)).sum()
    fn = ((pred_best == 0) & (gt_pherr == 1)).sum()
    tn = ((pred_best == 0) & (gt_pherr == 0)).sum()

    print(f"\n  Best threshold: {best_thresh:.2f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  F1:        {f1:.3f}")
    print(f"  TP={tp:4d}  FP={fp:4d}")
    print(f"  FN={fn:4d}  TN={tn:4d}")

    return {
        "score_mae": mae, "score_rmse": rmse, "score_pearson": corr, "score_spearman": sp,
        "pherr_auc": auc, "pherr_f1": best_f1, "pherr_precision": prec, "pherr_recall": rec,
        "pherr_threshold": best_thresh,
    }


# ============================================================
# 7. Comparison with baseline GOP
# ============================================================
def print_comparison(e2e_metrics):
    print(f"\n{'='*60}")
    print(f"COMPARISON: GOP baseline vs E2E model")
    print(f"{'='*60}")
    print(f"  {'Metric':<25s}  {'GOP (v1.0)':>12s}  {'E2E (v2.0)':>12s}  {'Improvement':>12s}")
    print(f"  {'-'*25}  {'-'*12}  {'-'*12}  {'-'*12}")

    gop_baseline = {
        "pherr_auc": 0.738, "pherr_f1": 0.476,
        "pherr_precision": 0.379, "pherr_recall": 0.638,
        "score_pearson": 0.372, "score_mae": 27.44,
    }

    for key, label in [
        ("pherr_auc", "Error detect AUC"),
        ("pherr_f1", "Error detect F1"),
        ("pherr_precision", "Precision"),
        ("pherr_recall", "Recall"),
        ("score_pearson", "Score Pearson r"),
        ("score_mae", "Score MAE"),
    ]:
        old = gop_baseline.get(key, 0)
        new = e2e_metrics.get(key, 0)
        if key == "score_mae":
            imp = f"{old - new:+.1f}"
        else:
            imp = f"{new - old:+.3f}"
        print(f"  {label:<25s}  {old:>12.3f}  {new:>12.3f}  {imp:>12s}")


# ============================================================
# 8. Main
# ============================================================
def main():
    audio_dir = Path("/mnt/weka/home/jianshu.she/Voice-correction/audio_files")
    xlsx_path = "/mnt/weka/home/jianshu.she/Voice-correction/eval_log.xlsx"
    cache_path = Path("/mnt/weka/home/jianshu.she/Voice-correction/feature_cache.pt")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load GT
    print("Loading ground truth...")
    records = load_gt_phonemes(xlsx_path)
    print(f"Loaded {len(records)} records")

    # Extract features
    print("\nExtracting features...")
    extractor = FeatureExtractor(device=device)
    X, y_score, y_pherr, phone_types = build_dataset(
        records, audio_dir, extractor, cache_path=str(cache_path)
    )
    print(f"\nDataset: {len(X)} phoneme samples, {HIDDEN_DIM}-dim features")
    print(f"  Score range: [{y_score.min():.0f}, {y_score.max():.0f}]")
    print(f"  Error rate: {y_pherr.mean()*100:.1f}%")
    print(f"  Unique phone types: {len(set(phone_types))}")

    # Train/test split
    indices = np.arange(len(X))
    idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42)
    idx_train, idx_val = train_test_split(idx_train, test_size=0.15, random_state=42)

    X_train, y_score_train, y_pherr_train = X[idx_train], y_score[idx_train], y_pherr[idx_train]
    X_val, y_score_val, y_pherr_val = X[idx_val], y_score[idx_val], y_pherr[idx_val]
    X_test, y_score_test, y_pherr_test = X[idx_test], y_score[idx_test], y_pherr[idx_test]

    print(f"\n  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Train
    print("\nTraining MLP scorer...")
    model = train_model(
        X_train, y_score_train, y_pherr_train,
        X_val, y_score_val, y_pherr_val,
        epochs=80, batch_size=256, lr=1e-3, device=device,
    )

    # Evaluate
    metrics = evaluate(model, X_test, y_score_test, y_pherr_test, device=device)

    # Comparison
    print_comparison(metrics)

    # Save model
    save_path = Path("/mnt/weka/home/jianshu.she/Voice-correction/phoneme_scorer.pt")
    torch.save({
        "model_state": model.state_dict(),
        "metrics": metrics,
        "config": {"input_dim": HIDDEN_DIM, "hidden_dim": 256},
    }, save_path)
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()
