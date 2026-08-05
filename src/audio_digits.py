"""
audio_digits.py — spoken-digit recognition under acoustic regimes (audio modality).
==================================================================================

A fourth modality — audio — using the tiny Free Spoken Digit Dataset (FSDD,
~3000 recordings of digits 0-9, 8 kHz; fetched automatically, ~22 MB). Each
waveform is perturbed by one of R **acoustic regimes** (time-stretch, pitch
shift, additive noise, low-pass, gain), which change the spectrogram statistics
while leaving the spoken digit unchanged. The regime is the nuisance the model
must be invariant to; it is identifiable from a global summary of the
log-spectrogram, which CCA reads to adapt its curve per recording.

Features are log-magnitude STFT spectrograms (core `torch.stft`, no torchaudio)
resized to 32x32; the classifier is the same small CNN as the image tasks, so
the only new ingredient is the modality.

Controls: single-regime null (clean audio) + context-shuffle ablation.
Writes results/audio_digits.json (+_single).   Run: python audio_digits.py
"""

import io
import json
import os
import statistics
import time
import urllib.request
import wave
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import metrics
from cca_core import set_cca, mean_coeff_variance
from cca_factory import make_cca2d

SEED = 1337
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 8000
NSAMP = 4096                 # ~0.5 s window (pad/crop)
SPEC = 32                    # spectrogram resized to SPEC x SPEC
BATCH = 128
EPOCHS = int(os.environ.get("EPOCHS", 12))
LR = 2e-3
SEEDS = [int(s) for s in os.environ.get("SEEDS", "1337,1338,1339").split(",")]
FSDD_URL = ("https://github.com/Jakobovski/free-spoken-digit-dataset/"
            "archive/refs/heads/master.zip")
DATA_DIR = "data/fsdd"
os.makedirs("results", exist_ok=True)


def mstd(v):
    return (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0)


# --------------------------------------------------------------------------- #
#  Fetch + decode FSDD wavs (stdlib only)
# --------------------------------------------------------------------------- #
def ensure_fsdd():
    rec = os.path.join(DATA_DIR, "recordings")
    if os.path.isdir(rec) and len(os.listdir(rec)) > 100:
        return rec
    os.makedirs(DATA_DIR, exist_ok=True)
    print("downloading Free Spoken Digit Dataset (~22 MB)...")
    raw = urllib.request.urlopen(FSDD_URL, timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for name in z.namelist():
            if "/recordings/" in name and name.endswith(".wav"):
                fn = os.path.basename(name)
                os.makedirs(rec, exist_ok=True)
                with z.open(name) as src, open(os.path.join(rec, fn), "wb") as dst:
                    dst.write(src.read())
    return rec


def load_wav(path):
    with wave.open(path, "rb") as w:
        n = w.getnframes()
        a = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32)
    a /= 32768.0
    if len(a) >= NSAMP:
        a = a[:NSAMP]
    else:
        a = np.pad(a, (0, NSAMP - len(a)))
    return a


def load_dataset():
    rec = ensure_fsdd()
    wavs, labels = [], []
    for fn in sorted(os.listdir(rec)):
        if fn.endswith(".wav"):
            wavs.append(load_wav(os.path.join(rec, fn)))
            labels.append(int(fn.split("_")[0]))
    X = torch.tensor(np.stack(wavs))
    y = torch.tensor(labels, dtype=torch.long)
    return X, y


# --------------------------------------------------------------------------- #
#  Acoustic regimes (waveform-level perturbations)
# --------------------------------------------------------------------------- #
def _resample(w, factor):
    """Linear-resample a [B,N] waveform by `factor`, then crop/pad to NSAMP."""
    n = w.shape[1]
    g = F.interpolate(w.unsqueeze(1), scale_factor=factor, mode="linear",
                      align_corners=False).squeeze(1)
    if g.shape[1] >= n:
        return g[:, :n]
    return F.pad(g, (0, n - g.shape[1]))


def regimes(multi):
    if not multi:
        return [lambda w: w]
    return [
        lambda w: w,                                   # clean
        lambda w: _resample(w, 0.8),                   # time-stretch (slower)
        lambda w: _resample(w, 1.25),                  # time-stretch (faster)
        lambda w: w + 0.05 * torch.randn_like(w),      # additive noise
        lambda w: torch.tanh(3.0 * w),                 # gain + soft clip
        lambda w: _lowpass(w),                         # low-pass
    ]


def _lowpass(w):
    W = torch.fft.rfft(w, dim=1)
    keep = W.shape[1] // 3
    W[:, keep:] = 0
    return torch.fft.irfft(W, n=w.shape[1], dim=1)


def spectrogram(w):
    """[B,N] waveform -> [B,1,SPEC,SPEC] log-magnitude STFT, normalised."""
    win = torch.hann_window(256, device=w.device)
    S = torch.stft(w, n_fft=256, hop_length=128, win_length=256, window=win,
                   return_complex=True)
    mag = (S.abs() + 1e-6).log()                       # [B, F, T]
    mag = F.interpolate(mag.unsqueeze(1), size=SPEC, mode="bilinear",
                        align_corners=False)
    mag = (mag - mag.mean(dim=(2, 3), keepdim=True)) / (
        mag.std(dim=(2, 3), keepdim=True) + 1e-5)
    return mag


def build_data(multi, X, y):
    regs = regimes(multi)
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(X), generator=g)
    X, y = X[perm], y[perm]
    n_te = len(X) // 5
    splits = {}
    for name, xs, ys in [("test", X[:n_te], y[:n_te]), ("train", X[n_te:], y[n_te:])]:
        r = torch.randint(len(regs), (len(xs),), generator=g)
        out = xs.clone()
        for ri, fn in enumerate(regs):
            m = r == ri
            if m.any():
                out[m] = fn(xs[m])
        splits[name] = (out, ys)
    return splits["train"], splits["test"], len(regs)


# --------------------------------------------------------------------------- #
#  Model: small CNN over the spectrogram (CCA2D reads GAP context internally)
# --------------------------------------------------------------------------- #
class SmallCNN(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.c1 = nn.Conv2d(1, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 10)
        if mode == "cca":
            self.a1, self.a2 = make_cca2d(16), make_cca2d(32)
        else:
            self.a1 = self.a2 = nn.SiLU()

    def forward(self, spec):
        x = F.max_pool2d(self.a1(self.c1(spec)), 2)
        x = F.max_pool2d(self.a2(self.c2(x)), 2)
        return self.fc(x.flatten(1))


def iterate(X, y, gen, train):
    order = torch.randperm(len(X), generator=gen) if train else torch.arange(len(X))
    for i in range(0, len(X), BATCH):
        idx = order[i:i + BATCH]
        wav = X[idx].to(DEVICE)
        yield spectrogram(wav), y[idx].to(DEVICE)


@torch.no_grad()
def evaluate(model, Xte, yte, *, shuffle_ctx=False, collect=False):
    model.eval()
    set_cca(model, collect=collect, shuffle_ctx=shuffle_ctx)
    gen = torch.Generator().manual_seed(SEED)
    correct, total, t0 = 0, 0, time.time()
    for sb, yb in iterate(Xte, yte, gen, False):
        correct += (model(sb).argmax(1) == yb).sum().item()
        total += len(yb)
    dt = time.time() - t0
    set_cca(model, collect=False, shuffle_ctx=False)
    return {"acc": correct / total, "throughput": total / dt,
            "coeff_var": (mean_coeff_variance(model) if collect else None)}


def train(mode, data):
    (Xtr, ytr), _, _ = data
    torch.manual_seed(SEED)
    model = SmallCNN(mode).to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(SEED)
    for _ in range(EPOCHS):
        model.train()
        for sb, yb in iterate(Xtr, ytr, gen, True):
            loss = F.cross_entropy(model(sb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model, n_par


def run(multi, tag, base):
    global SEED
    raw = {"silu": [], "cca": [], "shuffle": []}
    coeff_vars, n_par_ref, n_reg = [], {}, None
    models, data = {}, None
    for sd in SEEDS:
        SEED = sd
        data = build_data(multi, *base)
        n_reg = data[2]
        for mode in ("silu", "cca"):
            model, n_par = train(mode, data)
            ev = evaluate(model, data[1][0], data[1][1], collect=(mode == "cca"))
            models[mode] = model
            n_par_ref[mode] = n_par
            raw[mode].append(ev["acc"])
            if ev["coeff_var"] is not None:
                coeff_vars.append(ev["coeff_var"])
        sh = evaluate(models["cca"], data[1][0], data[1][1], shuffle_ctx=True)
        raw["shuffle"].append(sh["acc"])
    print(f"\n[{tag}] regimes={n_reg} seeds={SEEDS}")
    out = {"task": "audio_digits", "tag": tag, "n_regimes": n_reg,
           "dataset": "Free Spoken Digit Dataset (log-STFT 32x32)",
           "seeds": SEEDS, "models": {}}
    for mode in ("silu", "cca", "shuffle"):
        m, s = mstd(raw[mode])
        out["models"][mode] = {"acc": m, "acc_std": s, "params": n_par_ref.get(mode)}
        print(f"  {mode:7s} acc {m*100:.2f}% +/- {s*100:.2f}")
    if coeff_vars:
        out["coeff_var"] = torch.stack(coeff_vars).mean(0).tolist()
    bm, _ = mstd(raw["silu"])
    cm, _ = mstd(raw["cca"])
    out["err_reduction_pct"] = 100 * ((1 - bm) - (1 - cm)) / (1 - bm) if bm < 1 else 0
    out["acc_gain_pp"] = 100 * (cm - bm)
    print(f"  -> error -{out['err_reduction_pct']:.1f}%  (+{out['acc_gain_pp']:.2f}pp)")
    measure_cost(out, models, data)
    return out


def measure_cost(out, models, data):
    sb = spectrogram(data[1][0][:BATCH].to(DEVICE))
    yb = data[1][1][:BATCH].to(DEVICE)
    base_par = out["models"]["silu"]["params"]
    for mode in ("silu", "cca"):
        model = models[mode]

        def fwd(model=model):
            return model(sb)

        def step(model=model):
            loss = F.cross_entropy(model(sb), yb)
            model.zero_grad(set_to_none=True)
            loss.backward()

        out["models"][mode]["cost"] = metrics.model_cost(
            model, DEVICE, forward_fn=fwd, train_step_fn=step,
            n_items=len(sb), baseline_params=base_par)
    out["viability"] = metrics.viability(
        out["err_reduction_pct"], out["models"]["silu"]["cost"]["throughput"],
        out["models"]["cca"]["cost"]["throughput"])
    v = out["viability"]
    print(f"  cost: CCA {out['models']['cca']['cost']['params']/1e3:.0f}k par "
          f"({out['models']['cca']['cost']['params_overhead_pct']:+.0f}%), "
          f"throughput cost {v['throughput_cost_pct']:.0f}% -> "
          f"R={v['viability_ratio']:.2f} "
          f"({'rentable' if v['rentable'] else 'NOT rentable'})")


def main():
    base = load_dataset()
    print(f"FSDD: {len(base[0])} recordings")
    multi = run(True, "multi-regime", base)
    with open("results/audio_digits.json", "w") as f:
        json.dump(multi, f, indent=2)
    single = run(False, "single-regime", base)
    with open("results/audio_digits_single.json", "w") as f:
        json.dump(single, f, indent=2)
    print("\nsaved results/audio_digits.json (+_single)")


if __name__ == "__main__":
    main()
