
from __future__ import annotations

import numpy as np
import pandas as pd

from prc.baseb.models.base import Model, register

N_QUANTILES = 1001


def _emb_dim(cardinality: int) -> int:
    return int(min(32, max(2, round(1.6 * cardinality**0.56))))


@register
class MLP(Model):
    name = "mlp"

    def __init__(
        self,
        hidden: tuple[int, ...] | list[int] = (512, 256, 128),
        dropout: float = 0.1,
        learning_rate: float = 2e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 2048,
        max_epochs: int = 40,
        patience: int = 4,
        stop_epoch: int | None = None,
        seed: int | None = None,
        device: str | None = None,
        verbose: bool = True,
    ):
        super().__init__(
            hidden=list(hidden),
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            stop_epoch=stop_epoch,
            seed=seed,
            device=device,
            verbose=verbose,
        )
        self.hidden = [int(h) for h in hidden]
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.stop_epoch = None if stop_epoch is None else int(stop_epoch)
        if self.stop_epoch is not None and not 1 <= self.stop_epoch <= self.max_epochs:
            raise ValueError(f"stop_epoch must be in [1, max_epochs], got {stop_epoch!r}")
        self.seed = 0 if seed is None else int(seed)
        self.device_cfg = device
        self.verbose = verbose


    def _fit_transform(self, X: pd.DataFrame) -> None:
        self.features = list(X.columns)
        self.cat_cols = [c for c in self.features if c.startswith("cat_")]
        self.num_cols = [c for c in self.features if c not in self.cat_cols]
        self.cardinality = {c: int(max(X[c].max(), -1)) + 2 for c in self.cat_cols}
        probs = np.linspace(0.0, 1.0, N_QUANTILES)
        from statistics import NormalDist

        clipped = np.clip(probs, 1e-4, 1 - 1e-4)
        self._normal = np.array([NormalDist().inv_cdf(p) for p in clipped])
        self.quantiles: dict[str, np.ndarray] = {}
        self.has_nan: list[str] = []
        for c in self.num_cols:
            col = self._finite(X[c])
            ok = col[~np.isnan(col)]
            self.quantiles[c] = np.quantile(ok, probs) if len(ok) else np.zeros(N_QUANTILES)
            if np.isnan(col).any():
                self.has_nan.append(c)

    @staticmethod
    def _finite(s: pd.Series) -> np.ndarray:
        col = s.to_numpy(dtype=float, copy=True)
        col[~np.isfinite(col)] = np.nan
        return col

    def _transform(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        n = len(X)
        num = np.zeros((n, len(self.num_cols) + len(self.has_nan)), dtype=np.float32)
        for j, c in enumerate(self.num_cols):
            col = self._finite(X[c])
            q = self.quantiles[c]
            qx = q + np.arange(len(q)) * 1e-9 * max(1.0, float(np.abs(q).max()))
            out = np.interp(col, qx, self._normal)
            out[np.isnan(col)] = 0.0
            num[:, j] = out
        for k, c in enumerate(self.has_nan):
            num[:, len(self.num_cols) + k] = np.isnan(self._finite(X[c]))
        cat = np.zeros((n, len(self.cat_cols)), dtype=np.int64)
        for j, c in enumerate(self.cat_cols):
            codes = X[c].to_numpy(dtype=np.int64) + 1
            codes[(codes < 0) | (codes >= self.cardinality[c])] = 0
            cat[:, j] = codes
        return num, cat


    def _net(self, n_num: int):
        import torch
        from torch import nn

        class Net(nn.Module):
            def __init__(inner, cards: list[int], hidden: list[int], dropout: float):
                super().__init__()
                inner.embs = nn.ModuleList([nn.Embedding(k, _emb_dim(k)) for k in cards])
                width = n_num + sum(_emb_dim(k) for k in cards)
                layers: list[nn.Module] = []
                for h in hidden:
                    layers += [nn.Linear(width, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(dropout)]
                    width = h
                layers.append(nn.Linear(width, 1))
                inner.mlp = nn.Sequential(*layers)

            def forward(inner, num, cat):
                parts = [num] + [e(cat[:, j]) for j, e in enumerate(inner.embs)]
                return inner.mlp(torch.cat(parts, dim=1)).squeeze(1)

        return Net([self.cardinality[c] for c in self.cat_cols], self.hidden, self.dropout)

    def _device(self):
        import torch

        if self.device_cfg:
            return torch.device(self.device_cfg)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _predict_z(self, num: np.ndarray, cat: np.ndarray) -> np.ndarray:
        import torch

        self.net.eval()
        out = []
        step = max(self.batch_size, 16384)
        with torch.no_grad():
            for i in range(0, len(num), step):
                a = torch.from_numpy(num[i : i + step]).to(self.device)
                b = torch.from_numpy(cat[i : i + step]).to(self.device)
                out.append(self.net(a, b).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)


    def fit(self, X, y, meta, valid: tuple[pd.DataFrame, np.ndarray] | None = None):
        import copy
        import time

        import torch

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.device = self._device()
        self._fit_transform(X)
        num, cat = self._transform(X)
        y = np.asarray(y, dtype=float)
        self.y_mean, self.y_std = float(y.mean()), float(y.std() or 1.0)
        z = ((y - self.y_mean) / self.y_std).astype(np.float32)
        if valid is not None:
            vnum, vcat = self._transform(valid[0])
            vz = (np.asarray(valid[1], dtype=float) - self.y_mean) / self.y_std

        self.net = self._net(num.shape[1]).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        steps_per_epoch = max(1, len(z) // self.batch_size)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.learning_rate, total_steps=steps_per_epoch * self.max_epochs, pct_start=0.1
        )
        num_t = torch.from_numpy(num).to(self.device)
        cat_t = torch.from_numpy(cat).to(self.device)
        z_t = torch.from_numpy(z).to(self.device)

        best, best_state, best_epoch, bad = np.inf, None, self.max_epochs, 0
        self.history: list[tuple[int, float, float]] = []
        last = self.stop_epoch or self.max_epochs
        for epoch in range(1, last + 1):
            t0 = time.time()
            self.net.train()
            order = torch.from_numpy(rng.permutation(len(z))).to(self.device)
            total = 0.0
            for s in range(steps_per_epoch):
                idx = order[s * self.batch_size : (s + 1) * self.batch_size]
                loss = torch.mean((self.net(num_t[idx], cat_t[idx]) - z_t[idx]) ** 2)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                total += float(loss.detach())
            train_loss = total / steps_per_epoch
            if valid is None:
                self.history.append((epoch, train_loss, float("nan")))
                continue
            vloss = float(np.mean((self._predict_z(vnum, vcat) - vz) ** 2))
            self.history.append((epoch, train_loss, vloss))
            if self.verbose:
                rmse = np.sqrt(vloss) * self.y_std
                print(f"  mlp epoch {epoch}: train {train_loss:.4f} holdout {vloss:.4f} (~{rmse:.1f} s) "
                      f"{time.time() - t0:.0f} s", flush=True)
            if vloss < best:
                best, best_epoch, bad = vloss, epoch, 0
                best_state = copy.deepcopy(self.net.state_dict())
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.best_iteration = best_epoch if valid is not None else last
        return self

    def __getstate__(self):
        state = dict(self.__dict__)
        net = state.pop("net", None)
        state["net_state"] = None if net is None else {k: v.detach().cpu() for k, v in net.state_dict().items()}
        return state

    def __setstate__(self, state):
        net_state = state.pop("net_state", None)
        self.__dict__.update(state)
        if net_state is not None:
            self.net = self._net(len(self.num_cols) + len(self.has_nan)).to(self.device)
            self.net.load_state_dict(net_state)

    def predict(self, X, meta):
        num, cat = self._transform(X[self.features])
        return self._predict_z(num, cat).astype(float) * self.y_std + self.y_mean
