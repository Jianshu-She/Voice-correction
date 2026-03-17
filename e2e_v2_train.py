"""
E2E Phoneme Scorer v2 - Improved training
==========================================
Improvements:
1. Add phoneme identity embedding as extra input
2. Stronger regularization (larger dropout, weight decay)
3. Larger model with batch normalization
4. Data augmentation: Gaussian noise on hidden states
5. Multi-layer hidden state aggregation (concat last 4 layers instead of just last)
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
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from scipy import stats
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, Wav2Vec2ForCTC

warnings.filterwarnings("ignore")

SAMPLE_RATE = 16000

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

# All possible ARPAbet phonemes
ALL_PHONES = sorted(ARPABET_TO_IPA.keys())
PHONE_TO_ID = {ph: i for i, ph in enumerate(ALL_PHONES)}
N_PHONE_TYPES = len(ALL_PHONES)


# ============================================================
# Feature extraction with multi-layer hidden states
# ============================================================
class MultiLayerExtractor:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.g2p = G2p()

        print("Loading wav2vec2-large (with hidden states)...")
        self.backbone = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-large-960h",
            output_hidden_states=True,
        ).to(self.device)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.fe_backbone = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-960h")

        print("Loading phoneme CTC (for alignment)...")
        self.ctc_model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
        ).to(self.device)
        self.ctc_model.eval()
        for p in self.ctc_model.parameters():
            p.requires_grad = False

        self.fe_ctc = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")

        vocab_path = hf_hub_download("facebook/wav2vec2-xlsr-53-espeak-cv-ft", "vocab.json")
        with open(vocab_path) as f:
            self.vocab = json.load(f)
        self.blank_idx = self.vocab.get("<pad>", 0)
        print("Models loaded.")

    def _load_audio(self, path):
        w, sr = torchaudio.load(path)
        if w.shape[0] > 1: w = w.mean(0, keepdim=True)
        if sr != SAMPLE_RATE: w = F_audio.resample(w, sr, SAMPLE_RATE)
        return w

    def _text_to_arpabet(self, text):
        phones = self.g2p(text)
        result = []
        for ph in phones:
            if ph == " ": continue
            clean = re.sub(r"\d", "", ph).lower()
            if clean: result.append(clean)
        return result

    def _arpabet_to_idx(self, ph):
        for ipa in ARPABET_TO_IPA.get(ph, []):
            if ipa in self.vocab: return self.vocab[ipa]
        return self.vocab.get(ph, -1)

    def _viterbi_align(self, emissions, phone_indices):
        T, C = emissions.shape
        S = len(phone_indices)
        if S == 0 or T < S: return []
        extended = [self.blank_idx]
        for p in phone_indices:
            extended.append(p)
            extended.append(self.blank_idx)
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
                if s > 1 and extended[s] != self.blank_idx and extended[s] != extended[s-2]:
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

    @torch.no_grad()
    def extract(self, audio_path, text):
        waveform = self._load_audio(audio_path)
        wav_np = waveform.squeeze(0).numpy()

        # CTC emissions for alignment
        ctc_inputs = self.fe_ctc(wav_np, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        ctc_logits = self.ctc_model(ctc_inputs.input_values.to(self.device)).logits
        emissions = torch.log_softmax(ctc_logits, dim=-1).squeeze(0).cpu()

        # Hidden states from backbone (last 4 layers)
        bb_inputs = self.fe_backbone(wav_np, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        bb_out = self.backbone(bb_inputs.input_values.to(self.device))
        # Concat last 4 hidden layers: (T, 1024*4) = (T, 4096)
        last4 = torch.cat([bb_out.hidden_states[-i].squeeze(0).cpu() for i in range(1, 5)], dim=-1)
        # Also get just last layer for comparison
        last_hidden = bb_out.last_hidden_state.squeeze(0).cpu()  # (T, 1024)

        # G2P + alignment
        arpabet = self._text_to_arpabet(text)
        indices, valid = [], []
        for ph in arpabet:
            idx = self._arpabet_to_idx(ph)
            if idx >= 0:
                indices.append(idx)
                valid.append(ph)
        if not indices: return []

        path = self._viterbi_align(emissions, indices)
        if not path: return []

        # Group frames
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

        T_ctc = emissions.shape[0]
        T_hidden = last_hidden.shape[0]
        scale = T_hidden / T_ctc

        results = []
        for i, ph in enumerate(valid):
            if i >= len(segments): break
            _, frames = segments[i]
            h_start = max(0, int(min(frames) * scale))
            h_end = min(T_hidden, int((max(frames) + 1) * scale))
            if h_end <= h_start: h_end = h_start + 1

            # Last layer features
            h_last = last_hidden[h_start:h_end].mean(dim=0)  # (1024,)
            # Multi-layer features
            h_multi = last4[h_start:h_end].mean(dim=0)  # (4096,)

            # Also compute GOP from CTC emissions for this segment
            expected_idx = indices[i]
            seg_emissions = emissions[frames]
            target_lp = seg_emissions[:, expected_idx].mean().item()
            mask = torch.ones(emissions.shape[1], dtype=torch.bool)
            mask[self.blank_idx] = False
            mask[expected_idx] = False
            best_other = seg_emissions[:, mask].max(dim=-1).values.mean().item()
            gop = target_lp - best_other

            phone_id = PHONE_TO_ID.get(ph, 0)

            results.append({
                "arpabet": ph,
                "phone_id": phone_id,
                "h_last": h_last,
                "h_multi": h_multi,
                "gop": gop,
                "n_frames": len(frames),
            })
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
        phones = re.findall(r'"char":"([^"]+)","ph2alpha":"([^"]*)".*?"pherr":(\d+).*?"score":([\d.]+)', s)
        gt = [{"phone": p.lower(), "pherr": int(e), "score": float(sc)} for p, _, e, sc in phones]
        records.append({"file_name": fname, "content": content, "gt_phones": gt})
    return records


# ============================================================
# Build dataset
# ============================================================
def build_dataset(records, audio_dir, extractor, cache_path=None):
    if cache_path and Path(cache_path).exists():
        print(f"Loading cached features from {cache_path}")
        data = torch.load(cache_path, weights_only=False)
        return data

    all_h_last, all_h_multi = [], []
    all_gop, all_phone_id, all_n_frames = [], [], []
    all_score, all_pherr = [], []
    n_skip = 0

    for i, rec in enumerate(records):
        audio_path = audio_dir / rec["file_name"]
        if not audio_path.exists():
            n_skip += 1
            continue
        try:
            feats = extractor.extract(str(audio_path), rec["content"])
            gt = rec["gt_phones"]
            n = min(len(feats), len(gt))
            for j in range(n):
                all_h_last.append(feats[j]["h_last"])
                all_h_multi.append(feats[j]["h_multi"])
                all_gop.append(feats[j]["gop"])
                all_phone_id.append(feats[j]["phone_id"])
                all_n_frames.append(feats[j]["n_frames"])
                all_score.append(gt[j]["score"])
                all_pherr.append(gt[j]["pherr"])
        except:
            n_skip += 1
        if (i+1) % 100 == 0:
            print(f"  [{i+1}/{len(records)}] samples={len(all_score)}, skip={n_skip}")

    data = {
        "h_last": torch.stack(all_h_last),
        "h_multi": torch.stack(all_h_multi),
        "gop": torch.tensor(all_gop, dtype=torch.float32),
        "phone_id": torch.tensor(all_phone_id, dtype=torch.long),
        "n_frames": torch.tensor(all_n_frames, dtype=torch.float32),
        "y_score": torch.tensor(all_score, dtype=torch.float32),
        "y_pherr": torch.tensor(all_pherr, dtype=torch.float32),
    }

    if cache_path:
        torch.save(data, cache_path)
        print(f"Cached to {cache_path}")
    return data


# ============================================================
# Model v2
# ============================================================
class PhoneScorerV2(nn.Module):
    """
    Input: hidden_state (1024) + phone_embedding (32) + GOP (1) + n_frames (1)
    """
    def __init__(self, hidden_dim=1024, n_phone_types=N_PHONE_TYPES,
                 phone_emb_dim=32, mlp_dim=512):
        super().__init__()
        self.phone_emb = nn.Embedding(n_phone_types, phone_emb_dim)
        input_dim = hidden_dim + phone_emb_dim + 2  # +2 for GOP and n_frames

        self.shared = nn.Sequential(
            nn.Linear(input_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.score_head = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))
        self.pherr_head = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, h, phone_id, gop, n_frames):
        emb = self.phone_emb(phone_id)  # (B, 32)
        x = torch.cat([h, emb, gop.unsqueeze(-1), n_frames.unsqueeze(-1)], dim=-1)
        shared = self.shared(x)
        score = self.score_head(shared).squeeze(-1)
        pherr = self.pherr_head(shared).squeeze(-1)
        return score, pherr


# ============================================================
# Dataset
# ============================================================
class PhoneDatasetV2(Dataset):
    def __init__(self, h, gop, phone_id, n_frames, y_score, y_pherr, augment=False):
        self.h = h
        self.gop = gop
        self.phone_id = phone_id
        self.n_frames = n_frames
        self.y_score = y_score
        self.y_pherr = y_pherr
        self.augment = augment

    def __len__(self): return len(self.h)

    def __getitem__(self, idx):
        h = self.h[idx]
        if self.augment:
            h = h + torch.randn_like(h) * 0.01  # Gaussian noise
        return h, self.gop[idx], self.phone_id[idx], self.n_frames[idx], \
               self.y_score[idx], self.y_pherr[idx]


# ============================================================
# Train
# ============================================================
def train(data, device=None, epochs=100, batch_size=512, lr=5e-4):
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    N = len(data["y_score"])
    idx = np.arange(N)
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)
    idx_train, idx_val = train_test_split(idx_train, test_size=0.15, random_state=42)

    def make_ds(idxs, augment=False):
        return PhoneDatasetV2(
            data["h_last"][idxs], data["gop"][idxs], data["phone_id"][idxs],
            data["n_frames"][idxs], data["y_score"][idxs], data["y_pherr"][idxs],
            augment=augment,
        )

    train_ds = make_ds(idx_train, augment=True)
    val_ds = make_ds(idx_val)
    test_ds = make_ds(idx_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, pin_memory=True)

    model = PhoneScorerV2(hidden_dim=1024).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    pos_weight = torch.tensor([(1 - data["y_pherr"][idx_train].mean()) /
                                max(data["y_pherr"][idx_train].mean(), 0.01)]).to(device)

    print(f"\nTrain={len(idx_train)}, Val={len(idx_val)}, Test={len(idx_test)}")
    print(f"Pos weight for pherr: {pos_weight.item():.2f}\n")

    best_val_loss = float("inf")
    best_state = None
    patience, patience_counter = 15, 0

    for epoch in range(epochs):
        model.train()
        losses = []
        for h, gop, pid, nf, ys, yp in train_loader:
            h, gop, pid, nf = h.to(device), gop.to(device), pid.to(device), nf.to(device)
            ys, yp = ys.to(device), yp.to(device)

            pred_s, pred_p = model(h, pid, gop, nf)
            loss_s = F.mse_loss(pred_s, ys)
            loss_p = F.binary_cross_entropy_with_logits(pred_p, yp, pos_weight=pos_weight)
            loss = loss_s / 100.0 + loss_p

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for h, gop, pid, nf, ys, yp in val_loader:
                h, gop, pid, nf = h.to(device), gop.to(device), pid.to(device), nf.to(device)
                ys, yp = ys.to(device), yp.to(device)
                pred_s, pred_p = model(h, pid, gop, nf)
                loss = F.mse_loss(pred_s, ys) / 100.0 + \
                       F.binary_cross_entropy_with_logits(pred_p, yp, pos_weight=pos_weight)
                val_losses.append(loss.item())

        vl = np.mean(val_losses)
        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}  train={np.mean(losses):.4f}  val={vl:.4f}  "
                  f"best_val={best_val_loss:.4f}  patience={patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    model = model.to(device)

    # ============================================================
    # Test evaluation
    # ============================================================
    model.eval()
    test_loader = DataLoader(test_ds, batch_size=len(test_ds))
    with torch.no_grad():
        for h, gop, pid, nf, ys, yp in test_loader:
            h, gop, pid, nf = h.to(device), gop.to(device), pid.to(device), nf.to(device)
            pred_s, pred_p = model(h, pid, gop, nf)
            pred_s = pred_s.cpu().numpy()
            pred_p = torch.sigmoid(pred_p).cpu().numpy()
            gt_s = ys.numpy()
            gt_p = yp.numpy()

    pred_s_clip = np.clip(pred_s, 0, 100)

    # Score metrics
    mae = np.mean(np.abs(gt_s - pred_s_clip))
    rmse = np.sqrt(np.mean((gt_s - pred_s_clip) ** 2))
    corr, _ = stats.pearsonr(gt_s, pred_s_clip)
    sp, _ = stats.spearmanr(gt_s, pred_s_clip)

    print(f"\n{'='*60}")
    print(f"PHONE SCORE PREDICTION")
    print(f"{'='*60}")
    print(f"  MAE:      {mae:.2f}")
    print(f"  RMSE:     {rmse:.2f}")
    print(f"  Pearson:  {corr:.3f}")
    print(f"  Spearman: {sp:.3f}")
    abs_err = np.abs(gt_s - pred_s_clip)
    for t in [5, 10, 15, 20, 30]:
        print(f"  |err|<{t:2d}: {np.mean(abs_err<t)*100:.1f}%")

    # pherr metrics
    auc = roc_auc_score(gt_p, pred_p) if len(np.unique(gt_p)) > 1 else 0
    best_f1, best_thresh = 0, 0.5
    for th in np.arange(0.1, 0.9, 0.05):
        pb = (pred_p >= th).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(gt_p, pb, average="binary", zero_division=0)
        if f1 > best_f1: best_f1, best_thresh = f1, th

    pb = (pred_p >= best_thresh).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(gt_p, pb, average="binary")
    tp = ((pb==1)&(gt_p==1)).sum()
    fp = ((pb==1)&(gt_p==0)).sum()
    fn = ((pb==0)&(gt_p==1)).sum()
    tn = ((pb==0)&(gt_p==0)).sum()

    print(f"\n{'='*60}")
    print(f"PHONE ERROR DETECTION")
    print(f"{'='*60}")
    print(f"  AUC-ROC:   {auc:.3f}")
    print(f"  Threshold: {best_thresh:.2f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  F1:        {f1:.3f}")
    print(f"  TP={tp:4d}  FP={fp:4d}")
    print(f"  FN={fn:4d}  TN={tn:4d}")

    # Comparison
    print(f"\n{'='*60}")
    print(f"COMPARISON")
    print(f"{'='*60}")
    rows = [
        ("Error AUC",    0.738, auc),
        ("Error F1",     0.476, best_f1),
        ("Precision",    0.379, prec),
        ("Recall",       0.638, rec),
        ("Score Pearson",0.372, corr),
        ("Score MAE",   27.44,  mae),
    ]
    print(f"  {'Metric':<20s}  {'GOP v1.0':>10s}  {'E2E v1':>10s}  {'E2E v2':>10s}")
    e2e_v1 = [0.754, 0.500, 0.467, 0.538, 0.458, 24.93]
    for (name, gop, new), v1 in zip(rows, e2e_v1):
        print(f"  {name:<20s}  {gop:>10.3f}  {v1:>10.3f}  {new:>10.3f}")

    # Save
    save_path = Path("/mnt/weka/home/jianshu.she/Voice-correction/phoneme_scorer_v2.pt")
    torch.save({
        "model_state": model.cpu().state_dict(),
        "metrics": {"mae": mae, "pearson": corr, "auc": auc, "f1": best_f1},
    }, save_path)
    print(f"\nModel saved to {save_path}")

    return model


def main():
    audio_dir = Path("/mnt/weka/home/jianshu.she/Voice-correction/audio_files")
    xlsx_path = "/mnt/weka/home/jianshu.she/Voice-correction/eval_log.xlsx"
    cache_path = "/mnt/weka/home/jianshu.she/Voice-correction/feature_cache_v2.pt"

    device = torch.device("cuda:0")

    records = load_gt(xlsx_path)
    print(f"Loaded {len(records)} records")

    extractor = MultiLayerExtractor(device=device)
    data = build_dataset(records, audio_dir, extractor, cache_path=cache_path)
    print(f"\nDataset: {len(data['y_score'])} phonemes")

    train(data, device=device)


if __name__ == "__main__":
    main()
