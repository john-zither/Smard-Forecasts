"""Walk-forward neural forecast of all five regions at once, for the homelab GPU.

One sample is one delivery day: features [regions x 96 slots x F] (the same
SMARD-free features as the LightGBM model) -> scaled load [regions x 96].
Each block mixes along the day with a dilated convolution (shared by all
regions), then across regions with a learned region-to-region mixing at
every quarter-hour, so the model can use, say, TransnetBW's morning to
inform TenneT's. (Attention across the five regions did the same job about
four times slower on the RTX 2060.) Missing
features (radiation before 2024) are zeroed and flagged.

For each test month: train on every day up to two days before it, early-stop
on the last 42 of those days, ensemble SEEDS models.

    python -m demand.nn          # in the cluster: scripts/demand-train.sh
"""

import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from demand.data import REGIONS, SLOTS
from demand.evaluate import RESULTS, TEST_END, TEST_START, monthly, run, table
from demand.features import build, feature_columns

SEEDS = 3
BATCH = 64
VAL_DAYS = 42
HIDDEN = 64
BLOCKS = 4


def to_tensors(df: pd.DataFrame):
    """(days, X [D, R, 96, F], y [D, R, 96], scale [D, R, 96]) for days present in every region."""
    cols = [c for c in feature_columns(df) if c != 'region']
    per = {r: g.set_index('day') for r, g in df.groupby('region')}
    days = sorted(set.intersection(*(set(g.index.unique()) for g in per.values())))
    days = pd.DatetimeIndex(days)

    def stack(col_list):
        out = []
        for r in range(len(REGIONS)):
            g = per[r].loc[days, col_list].to_numpy(np.float32)
            out.append(g.reshape(len(days), SLOTS, len(col_list)))
        return np.stack(out, axis=1)

    X = stack(cols)
    y = stack(['target'])[..., 0]
    scale = stack(['scale'])[..., 0]
    return days, X, y, scale, cols


class Block(nn.Module):
    def __init__(self, h: int, regions: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(h, h, 5, padding=2 * dilation, dilation=dilation)
        self.mix = nn.Parameter(torch.eye(regions) + 0.01 * torch.randn(regions, regions))
        self.proj = nn.Linear(h, h)
        self.n1, self.n2 = nn.LayerNorm(h), nn.LayerNorm(h)
        self.drop = nn.Dropout(dropout)

    def forward(self, h):                       # h: [B, R, T, H]
        B, R, T, H = h.shape
        t = self.conv(h.reshape(B * R, T, H).transpose(1, 2)).transpose(1, 2).reshape(B, R, T, H)
        h = self.n1(h + self.drop(F.gelu(t)))
        m = torch.einsum('rs,bsth->brth', self.mix, h)
        return self.n2(h + self.drop(F.gelu(self.proj(m))))


class DayNet(nn.Module):
    def __init__(self, features: int, regions: int = len(REGIONS), h: int = HIDDEN, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(features, h)
        self.region = nn.Parameter(torch.zeros(regions, 1, h))
        self.blocks = nn.ModuleList(Block(h, regions, 2 ** (i % 4), dropout) for i in range(BLOCKS))
        self.out = nn.Linear(h, 1)

    def forward(self, x):                       # x: [B, R, T, F] -> [B, R, T]
        h = self.inp(x) + self.region
        for b in self.blocks:
            h = b(h)
        return 1 + self.out(h).squeeze(-1)      # scaled load sits around 1


def prepare(X: np.ndarray, train: np.ndarray):
    """Standardize on the training days; zero NaNs and append missing-value flags."""
    nan = np.isnan(X)
    flags = nan.any(axis=(0, 1, 2))
    mu = np.nanmean(X[train], axis=(0, 1, 2))
    sd = np.nanstd(X[train], axis=(0, 1, 2)) + 1e-6
    Z = np.nan_to_num((X - mu) / sd)
    return np.concatenate([Z, nan[..., flags].astype(np.float32)], axis=-1)


def fit(Z, y, train, val, seed, device, epochs=150, patience=20):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = DayNet(Z.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    amp = device.type == 'cuda'                 # fp16 autocast: ~2x faster on the RTX 2060
    scaler = torch.amp.GradScaler(enabled=amp)
    Zt, yt = torch.tensor(Z, device=device), torch.tensor(y, device=device)
    best, best_state, stale = np.inf, None, 0
    for _ in range(epochs):
        model.train()
        for b in np.array_split(rng.permutation(train), max(1, len(train) // BATCH)):
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                pred = model(Zt[b]).float()
            target = yt[b]
            ok = torch.isfinite(target)
            loss = (pred[ok] - target[ok]).abs().mean()
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        model.eval()
        with torch.no_grad():
            pred, target = model(Zt[val]), yt[val]
            ok = torch.isfinite(target)
            v = (pred[ok] - target[ok]).abs().mean().item()
        if v < best:
            best, best_state, stale = v, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model


def walk_forward(days, X, y, device) -> np.ndarray:
    pred = np.full(y.shape, np.nan, dtype=np.float32)
    for month in pd.date_range(TEST_START, TEST_END, freq='MS'):
        t0 = time.time()
        cutoff = month - pd.Timedelta('2D')
        labeled = np.isfinite(y).all(axis=(1, 2))
        hist = np.flatnonzero((days <= cutoff) & labeled)
        train, val = hist[:-VAL_DAYS], hist[-VAL_DAYS:]
        test = np.flatnonzero((days >= month) & (days < month + pd.offsets.MonthBegin(1)))
        Z = prepare(X, train)
        ens = []
        for seed in range(SEEDS):
            model = fit(Z, y, train, val, seed, device)
            with torch.no_grad():
                ens.append(model(torch.tensor(Z[test], device=device)).cpu().numpy())
        pred[test] = np.mean(ens, axis=0)
        print(f"  nn {month:%Y-%m}: {len(train)} training days, {time.time() - t0:.0f}s", flush=True)
    return pred


def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device {device}", flush=True)
    days, X, y, scale, cols = to_tensors(build())
    print(f"{len(days)} days x {len(REGIONS)} regions x {SLOTS} slots x {len(cols)} features", flush=True)
    pred = walk_forward(days, X, y, device)
    models = {'nn': {region: pd.DataFrame(pred[:, r] * scale[:, r], index=days)
                     for r, region in enumerate(REGIONS)}}
    res = run(models)
    text = (f"nMAE (MAE / mean load), {TEST_START.date()} → {TEST_END.date()}, issue time 12:00 D-1, "
            f"monthly walk-forward, {SEEDS}-seed ensemble\n\n" + table(res) + "\n\nBy month:\n\n" + monthly(models['nn'], 'nn'))
    print(text)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'nn.md').write_text(text + '\n')
    res.to_csv(RESULTS / 'nn.csv', index=False)
    np.savez_compressed(RESULTS / 'nn_pred.npz', days=days.values.astype('datetime64[D]').astype(str), pred=pred * scale)


if __name__ == '__main__':
    main()
