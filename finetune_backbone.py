"""
Fine-tune wav2vec2 Backbone for Phoneme Scoring
================================================
Key differences from e2e_v2_train.py:
1. Unfreeze top N layers of wav2vec2-large backbone
2. Use both datasets: eval_log.xlsx (1000 sentences) + word_eval_log.xlsx (10601 words)
3. On-the-fly feature extraction (no caching, since backbone is being updated)
4. Differential learning rate: backbone=1e-5, head=5e-4
5. Gradient accumulation to handle large effective batch size
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

ALL_PHONES = sorted(ARPABET_TO_IPA.keys())
PHONE_TO_ID = {ph: i for i, ph in enumerate(ALL_PHONES)}
N_PHONE_TYPES = len(ALL_PHONES)


# ============================================================
# Model: Backbone + Scoring Head (end-to-end)
# ============================================================
class BackbonePhoneScorer(nn.Module):
    """
    End-to-end model: wav2vec2 backbone → per-phoneme pooling → MLP scorer
    """
    def __init__(self, n_phone_types=N_PHONE_TYPES, phone_emb_dim=32,
                 hidden_dim=1024, mlp_dim=512, unfreeze_top_n=6):
        super().__init__()

        # Load wav2vec2-large backbone (disable time masking for fine-tuning)
        self.backbone = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-large-960h",
            output_hidden_states=False,
            mask_time_prob=0.0,
        )
        # Freeze all layers first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze top N transformer layers
        n_layers = len(self.backbone.encoder.layers)
        for i in range(n_layers - unfreeze_top_n, n_layers):
            for param in self.backbone.encoder.layers[i].parameters():
                param.requires_grad = True

        # Also unfreeze layer norm
        if hasattr(self.backbone.encoder, 'layer_norm'):
            for param in self.backbone.encoder.layer_norm.parameters():
                param.requires_grad = True

        self.fe_backbone = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-960h")

        # Phone embedding + MLP scorer (same as PhoneScorerV2)
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
        """Forward pass with pre-extracted hidden states (for batched training)."""
        emb = self.phone_emb(phone_id)
        x = torch.cat([h, emb, gop.unsqueeze(-1), n_frames.unsqueeze(-1)], dim=-1)
        shared = self.shared(x)
        score = self.score_head(shared).squeeze(-1)
        pherr = self.pherr_head(shared).squeeze(-1)
        return score, pherr

    def extract_hidden(self, waveform):
        """Extract hidden states from raw waveform. waveform: (1, T) tensor."""
        # Disable masking to avoid errors with short sequences
        out = self.backbone(waveform, mask_time_indices=None)
        return out.last_hidden_state  # (1, T_frames, 1024)

    def backbone_params(self):
        """Return only the trainable backbone parameters."""
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                yield param

    def head_params(self):
        """Return head parameters (phone_emb + MLP)."""
        yield from self.phone_emb.parameters()
        yield from self.shared.parameters()
        yield from self.score_head.parameters()
        yield from self.pherr_head.parameters()


# ============================================================
# Alignment engine (frozen CTC model)
# ============================================================
class AlignmentEngine:
    def __init__(self, device):
        self.device = device
        self.g2p = G2p()

        print("Loading phoneme CTC model for alignment...")
        self.ctc_model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
        ).to(device)
        self.ctc_model.eval()
        for p in self.ctc_model.parameters():
            p.requires_grad = False

        self.fe_ctc = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
        )

        vocab_path = hf_hub_download("facebook/wav2vec2-xlsr-53-espeak-cv-ft", "vocab.json")
        with open(vocab_path) as f:
            self.vocab = json.load(f)
        self.blank_idx = self.vocab.get("<pad>", 0)

    def text_to_arpabet(self, text):
        phones = self.g2p(text)
        result = []
        for ph in phones:
            if ph == " ":
                continue
            clean = re.sub(r"\d", "", ph).lower()
            if clean:
                result.append(clean)
        return result

    def arpabet_to_idx(self, ph):
        for ipa in ARPABET_TO_IPA.get(ph, []):
            if ipa in self.vocab:
                return self.vocab[ipa]
        return self.vocab.get(ph, -1)

    def viterbi_align(self, emissions, phone_indices):
        T, C = emissions.shape
        S = len(phone_indices)
        if S == 0 or T < S:
            return []
        extended = [self.blank_idx]
        for p in phone_indices:
            extended.append(p)
            extended.append(self.blank_idx)
        S_ext = len(extended)
        dp = np.full((T, S_ext), float("-inf"), dtype=np.float64)
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
    def align(self, audio_path, text):
        """Return alignment info: list of (arpabet, ctc_frames, phone_id, gop)."""
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        if sr != SAMPLE_RATE:
            waveform = F_audio.resample(waveform, sr, SAMPLE_RATE)

        wav_np = waveform.squeeze(0).numpy()

        # CTC emissions
        ctc_inputs = self.fe_ctc(wav_np, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        ctc_logits = self.ctc_model(ctc_inputs.input_values.to(self.device)).logits
        emissions = torch.log_softmax(ctc_logits, dim=-1).squeeze(0).cpu()

        # G2P + alignment
        arpabet = self.text_to_arpabet(text)
        indices, valid = [], []
        for ph in arpabet:
            idx = self.arpabet_to_idx(ph)
            if idx >= 0:
                indices.append(idx)
                valid.append(ph)
        if not indices:
            return [], emissions

        path = self.viterbi_align(emissions, indices)
        if not path:
            return [], emissions

        # Group frames by phone
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

        results = []
        for i, ph in enumerate(valid):
            if i >= len(segments):
                break
            _, frames = segments[i]

            # GOP
            expected_idx = indices[i]
            seg_emissions = emissions[frames]
            target_lp = seg_emissions[:, expected_idx].mean().item()
            mask = torch.ones(emissions.shape[1], dtype=torch.bool)
            mask[self.blank_idx] = False
            mask[expected_idx] = False
            best_other = seg_emissions[:, mask].max(dim=-1).values.mean().item()
            gop = target_lp - best_other

            results.append({
                "arpabet": ph,
                "phone_id": PHONE_TO_ID.get(ph, 0),
                "frames": frames,
                "gop": gop,
                "n_frames": len(frames),
            })

        return results, emissions


# ============================================================
# Dataset: stores pre-aligned info, extracts backbone features on-the-fly
# ============================================================
class PhonemeAlignedDataset(Dataset):
    """
    Each item is one phoneme with its alignment info.
    During __getitem__, we return pre-computed data.
    Backbone features are extracted in batch during training loop.
    """
    def __init__(self, samples, augment=False):
        """
        samples: list of dicts with keys:
            audio_path, phone_id, gop, n_frames, ctc_frames,
            y_score, y_pherr, T_ctc (CTC time steps for this audio)
        """
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "audio_path": s["audio_path"],
            "phone_id": s["phone_id"],
            "gop": s["gop"],
            "n_frames": s["n_frames"],
            "ctc_frames": s["ctc_frames"],
            "T_ctc": s["T_ctc"],
            "y_score": s["y_score"],
            "y_pherr": s["y_pherr"],
        }


# ============================================================
# Pre-align all data and cache alignment info
# ============================================================
def load_gt(xlsx_path):
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb.active
    records = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        fname, content, raw = row
        if not fname or not content or not raw:
            continue
        s = raw
        for _ in range(5):
            s = s.replace("\\\\", "\\")
        s = s.replace('\\"', '"')
        phones = re.findall(
            r'"char":"([^"]+)","ph2alpha":"([^"]*)".*?"pherr":(\d+).*?"score":([\d.]+)', s
        )
        gt = [{"phone": p.lower(), "pherr": int(e), "score": float(sc)} for p, _, e, sc in phones]
        records.append({"file_name": fname, "content": content, "gt_phones": gt})
    wb.close()
    return records


def pre_align_all(records, audio_dir, aligner, cache_path=None):
    """Pre-compute alignment for all samples. Returns list of per-phoneme dicts."""
    if cache_path and Path(cache_path).exists():
        print(f"Loading cached alignments from {cache_path}")
        return torch.load(cache_path, weights_only=False)

    all_samples = []
    n_skip = 0

    for i, rec in enumerate(records):
        audio_path = audio_dir / rec["file_name"]
        if not audio_path.exists():
            n_skip += 1
            continue
        try:
            aligned, emissions = aligner.align(str(audio_path), rec["content"])
            gt = rec["gt_phones"]
            n = min(len(aligned), len(gt))
            T_ctc = emissions.shape[0]

            for j in range(n):
                all_samples.append({
                    "audio_path": str(audio_path),
                    "phone_id": aligned[j]["phone_id"],
                    "gop": aligned[j]["gop"],
                    "n_frames": aligned[j]["n_frames"],
                    "ctc_frames": aligned[j]["frames"],
                    "T_ctc": T_ctc,
                    "y_score": gt[j]["score"],
                    "y_pherr": float(gt[j]["pherr"]),
                })
        except Exception as e:
            n_skip += 1
        if (i+1) % 200 == 0:
            print(f"  [{i+1}/{len(records)}] phonemes={len(all_samples)}, skip={n_skip}")

    print(f"  Total: {len(all_samples)} phonemes from {len(records)-n_skip} files, skip={n_skip}")

    if cache_path:
        torch.save(all_samples, cache_path)
        print(f"  Cached to {cache_path}")

    return all_samples


# ============================================================
# Training with backbone fine-tuning
# ============================================================
def extract_phoneme_features(model, audio_paths, frame_lists, T_ctcs, device, fe):
    """
    Extract backbone hidden states for a batch of phonemes.
    Groups phonemes by audio file to avoid redundant forward passes.
    Returns: (B, 1024) tensor of per-phoneme features.
    """
    # Group by audio path
    path_to_indices = {}
    for i, path in enumerate(audio_paths):
        if path not in path_to_indices:
            path_to_indices[path] = []
        path_to_indices[path].append(i)

    features = torch.zeros(len(audio_paths), 1024, device=device)

    for path, indices in path_to_indices.items():
        # Load audio once
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        if sr != SAMPLE_RATE:
            waveform = F_audio.resample(waveform, sr, SAMPLE_RATE)

        wav_np = waveform.squeeze(0).numpy()
        inputs = fe(wav_np, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        hidden = model.extract_hidden(inputs.input_values.to(device))  # (1, T_h, 1024)
        hidden = hidden.squeeze(0)  # (T_h, 1024)
        T_h = hidden.shape[0]

        for idx in indices:
            T_ctc = T_ctcs[idx]
            scale = T_h / max(T_ctc, 1)
            frames = frame_lists[idx]
            h_start = max(0, int(min(frames) * scale))
            h_end = min(T_h, int((max(frames) + 1) * scale))
            if h_end <= h_start:
                h_end = h_start + 1
            features[idx] = hidden[h_start:h_end].mean(dim=0)

    return features


def collate_fn(batch):
    """Custom collate that handles variable-length frame lists."""
    return {
        "audio_paths": [b["audio_path"] for b in batch],
        "phone_id": torch.tensor([b["phone_id"] for b in batch], dtype=torch.long),
        "gop": torch.tensor([b["gop"] for b in batch], dtype=torch.float32),
        "n_frames": torch.tensor([b["n_frames"] for b in batch], dtype=torch.float32),
        "ctc_frames": [b["ctc_frames"] for b in batch],
        "T_ctc": [b["T_ctc"] for b in batch],
        "y_score": torch.tensor([b["y_score"] for b in batch], dtype=torch.float32),
        "y_pherr": torch.tensor([b["y_pherr"] for b in batch], dtype=torch.float32),
    }


def train(all_samples, device, epochs=30, batch_size=64, lr_backbone=1e-5, lr_head=5e-4,
          unfreeze_top_n=6, grad_accum=4):
    """
    Fine-tune backbone + train head jointly.
    """
    # Split by audio file to avoid data leakage
    audio_files = list(set(s["audio_path"] for s in all_samples))
    files_train, files_test = train_test_split(audio_files, test_size=0.15, random_state=42)
    files_train, files_val = train_test_split(files_train, test_size=0.12, random_state=42)

    train_set = set(files_train)
    val_set = set(files_val)
    test_set = set(files_test)

    train_samples = [s for s in all_samples if s["audio_path"] in train_set]
    val_samples = [s for s in all_samples if s["audio_path"] in val_set]
    test_samples = [s for s in all_samples if s["audio_path"] in test_set]

    print(f"\nSplit by audio file:")
    print(f"  Train: {len(train_samples)} phonemes from {len(files_train)} files")
    print(f"  Val:   {len(val_samples)} phonemes from {len(files_val)} files")
    print(f"  Test:  {len(test_samples)} phonemes from {len(files_test)} files")

    train_ds = PhonemeAlignedDataset(train_samples, augment=True)
    val_ds = PhonemeAlignedDataset(val_samples)
    test_ds = PhonemeAlignedDataset(test_samples)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_fn, num_workers=0)

    # Model
    model = BackbonePhoneScorer(unfreeze_top_n=unfreeze_top_n).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total params: {total_params/1e6:.1f}M")
    print(f"  Trainable params: {trainable_params/1e6:.1f}M")

    # Differential learning rate
    optimizer = torch.optim.AdamW([
        {"params": model.backbone_params(), "lr": lr_backbone},
        {"params": model.head_params(), "lr": lr_head},
    ], weight_decay=1e-3)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Pos weight for imbalanced pherr
    pherr_vals = torch.tensor([s["y_pherr"] for s in train_samples])
    pos_rate = pherr_vals.mean().item()
    pos_weight = torch.tensor([(1 - pos_rate) / max(pos_rate, 0.01)]).to(device)
    print(f"  Pherr pos rate: {pos_rate:.3f}, pos_weight: {pos_weight.item():.2f}")

    best_val_loss = float("inf")
    best_state = None
    patience, patience_counter = 8, 0

    fe = model.fe_backbone

    n_steps_per_epoch = len(train_loader)
    print(f"  Steps per epoch: {n_steps_per_epoch}")

    for epoch in range(epochs):
        model.train()
        losses = []

        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            # Extract backbone features (with gradient for unfrozen layers)
            h = extract_phoneme_features(
                model, batch["audio_paths"], batch["ctc_frames"],
                batch["T_ctc"], device, fe
            )

            # Add noise augmentation
            h = h + torch.randn_like(h) * 0.01

            pid = batch["phone_id"].to(device)
            gop = batch["gop"].to(device)
            nf = batch["n_frames"].to(device)
            ys = batch["y_score"].to(device)
            yp = batch["y_pherr"].to(device)

            pred_s, pred_p = model(h, pid, gop, nf)
            loss_s = F.mse_loss(pred_s, ys)
            loss_p = F.binary_cross_entropy_with_logits(pred_p, yp, pos_weight=pos_weight)
            loss = (loss_s / 100.0 + loss_p) / grad_accum

            loss.backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            losses.append(loss.item() * grad_accum)

            if (step + 1) % 50 == 0:
                avg_loss = np.mean(losses[-50:])
                print(f"    Epoch {epoch+1} Step {step+1}/{n_steps_per_epoch}  loss={avg_loss:.4f}")

        scheduler.step()

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                h = extract_phoneme_features(
                    model, batch["audio_paths"], batch["ctc_frames"],
                    batch["T_ctc"], device, fe
                )
                pid = batch["phone_id"].to(device)
                gop = batch["gop"].to(device)
                nf = batch["n_frames"].to(device)
                ys = batch["y_score"].to(device)
                yp = batch["y_pherr"].to(device)

                pred_s, pred_p = model(h, pid, gop, nf)
                loss = F.mse_loss(pred_s, ys) / 100.0 + \
                       F.binary_cross_entropy_with_logits(pred_p, yp, pos_weight=pos_weight)
                val_losses.append(loss.item())

        vl = np.mean(val_losses)
        tl = np.mean(losses)

        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = " *"
        else:
            patience_counter += 1
            marker = ""

        print(f"  Epoch {epoch+1:3d}/{epochs}  train={tl:.4f}  val={vl:.4f}  "
              f"best={best_val_loss:.4f}  patience={patience_counter}/{patience}{marker}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(best_state)
    model = model.to(device)

    # ============================================================
    # Test evaluation
    # ============================================================
    print(f"\n{'='*60}")
    print("TEST EVALUATION")
    print(f"{'='*60}")

    model.eval()
    all_pred_s, all_pred_p, all_gt_s, all_gt_p = [], [], [], []

    test_loader = DataLoader(test_ds, batch_size=batch_size, collate_fn=collate_fn, num_workers=0)
    with torch.no_grad():
        for batch in test_loader:
            h = extract_phoneme_features(
                model, batch["audio_paths"], batch["ctc_frames"],
                batch["T_ctc"], device, fe
            )
            pid = batch["phone_id"].to(device)
            gop = batch["gop"].to(device)
            nf = batch["n_frames"].to(device)

            pred_s, pred_p = model(h, pid, gop, nf)
            all_pred_s.append(pred_s.cpu())
            all_pred_p.append(torch.sigmoid(pred_p).cpu())
            all_gt_s.append(batch["y_score"])
            all_gt_p.append(batch["y_pherr"])

    pred_s = torch.cat(all_pred_s).numpy()
    pred_p = torch.cat(all_pred_p).numpy()
    gt_s = torch.cat(all_gt_s).numpy()
    gt_p = torch.cat(all_gt_p).numpy()

    pred_s_clip = np.clip(pred_s, 0, 100)

    # Score metrics
    mae = np.mean(np.abs(gt_s - pred_s_clip))
    rmse = np.sqrt(np.mean((gt_s - pred_s_clip) ** 2))
    corr, _ = stats.pearsonr(gt_s, pred_s_clip)
    sp, _ = stats.spearmanr(gt_s, pred_s_clip)

    print(f"\nPHONE SCORE PREDICTION")
    print(f"  MAE:      {mae:.2f}")
    print(f"  RMSE:     {rmse:.2f}")
    print(f"  Pearson:  {corr:.3f}")
    print(f"  Spearman: {sp:.3f}")
    abs_err = np.abs(gt_s - pred_s_clip)
    for t in [5, 10, 15, 20, 30]:
        print(f"  |err|<{t:2d}: {np.mean(abs_err<t)*100:.1f}%")

    # Pherr metrics
    auc = roc_auc_score(gt_p, pred_p) if len(np.unique(gt_p)) > 1 else 0
    best_f1, best_thresh = 0, 0.5
    for th in np.arange(0.1, 0.9, 0.05):
        pb = (pred_p >= th).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(gt_p, pb, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, th

    pb = (pred_p >= best_thresh).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(gt_p, pb, average="binary")
    tp = ((pb==1) & (gt_p==1)).sum()
    fp = ((pb==1) & (gt_p==0)).sum()
    fn = ((pb==0) & (gt_p==1)).sum()
    tn = ((pb==0) & (gt_p==0)).sum()

    print(f"\nPHONE ERROR DETECTION")
    print(f"  AUC-ROC:   {auc:.3f}")
    print(f"  Threshold: {best_thresh:.2f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  F1:        {f1:.3f}")
    print(f"  TP={tp:4d}  FP={fp:4d}")
    print(f"  FN={fn:4d}  TN={tn:4d}")

    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON WITH PREVIOUS METHODS")
    print(f"{'='*60}")
    print(f"  {'Method':<35s}  {'AUC':>6s}  {'F1':>6s}  {'Prec':>6s}  {'Rec':>6s}  {'Pearson':>8s}  {'MAE':>6s}")
    methods = [
        ("GOP threshold (v1.0, 1K data)",   0.738, 0.476, 0.379, 0.638, 0.372, 27.44),
        ("E2E MLP frozen (v2, 1K data)",    0.814, 0.565, 0.500, 0.650, 0.528, 22.57),
        ("Phoneme comparison (1K data)",     0.691, 0.492, 0.379, 0.703, None,  None),
        (f"Backbone finetune (11K data)",    auc,   best_f1, prec, rec, corr,   mae),
    ]
    for name, a, f, p, r, c, m in methods:
        c_str = f"{c:.3f}" if c is not None else "  N/A"
        m_str = f"{m:.2f}" if m is not None else "  N/A"
        print(f"  {name:<35s}  {a:>6.3f}  {f:>6.3f}  {p:>6.3f}  {r:>6.3f}  {c_str:>8s}  {m_str:>6s}")

    # Save model
    save_dir = Path("/mnt/weka/home/jianshu.she/Voice-correction")
    save_path = save_dir / "backbone_finetuned.pt"
    torch.save({
        "model_state": model.state_dict(),
        "unfreeze_top_n": unfreeze_top_n,
        "metrics": {
            "auc": auc, "f1": best_f1, "precision": float(prec),
            "recall": float(rec), "pearson": corr, "mae": mae,
        },
    }, save_path)
    print(f"\nModel saved to {save_path}")

    return model


def main():
    device = torch.device("cuda:0")

    # Load both datasets
    print("Loading datasets...")
    records_sent = load_gt("/mnt/weka/home/jianshu.she/Voice-correction/eval_log.xlsx")
    records_word = load_gt("/mnt/weka/home/jianshu.she/Voice-correction/word_eval_log.xlsx")
    print(f"  Sentence data: {len(records_sent)} records")
    print(f"  Word data: {len(records_word)} records")

    # Tag audio directories
    for r in records_sent:
        r["audio_dir"] = "audio_files"
    for r in records_word:
        r["audio_dir"] = "words_audio_files"

    # Pre-align all data
    aligner = AlignmentEngine(device)

    base_dir = Path("/mnt/weka/home/jianshu.she/Voice-correction")

    cache_sent = base_dir / "align_cache_sentences.pt"
    cache_word = base_dir / "align_cache_words.pt"

    print("\nAligning sentence data...")
    samples_sent = pre_align_all(records_sent, base_dir / "audio_files", aligner, cache_sent)

    print("\nAligning word data...")
    samples_word = pre_align_all(records_word, base_dir / "words_audio_files", aligner, cache_word)

    all_samples = samples_sent + samples_word
    print(f"\nTotal: {len(all_samples)} phonemes")

    pherr_rate = np.mean([s["y_pherr"] for s in all_samples])
    print(f"Overall error rate: {pherr_rate:.3f}")

    # Train
    train(all_samples, device, epochs=30, batch_size=64, unfreeze_top_n=6)


if __name__ == "__main__":
    main()
