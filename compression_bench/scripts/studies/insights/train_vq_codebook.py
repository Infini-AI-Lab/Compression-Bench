"""Train the universal 8-d Gaussian k-means codebook (2^16 entries) that run_grids.vq16 quantizes into
(X = randn(N, 8); Lloyd k-means). CPU: N = 4M samples,
16 Lloyd iterations from a random-sample initialisation, seed 0; reports the held-out N(0,1) MSE per dimension.
Output: cb8_16_gauss_retrained.pt in the quant-error scratch directory (common.QUANT_ERROR_DIR,
compression_bench/evolve_space/analysis/quant_error)."""
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch

torch.set_num_threads(int(os.environ.get("NT", "24")))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import QUANT_ERROR_DIR as OUT  # noqa: E402
K, N = 1 << 16, 4_000_000
g = torch.Generator().manual_seed(0)
X = torch.randn(N, 8, generator=g)


def assign(X, C, chunk=1 << 13):
    cn = (C * C).sum(1)
    return torch.cat([(2 * X[i:i + chunk] @ C.t() - cn).argmax(1) for i in range(0, X.shape[0], chunk)])


def kmeans(X, C, iters):
    for it in range(iters):
        a = assign(X, C)
        S = torch.zeros_like(C).index_add_(0, a, X)
        n = torch.zeros(K).index_add_(0, a, torch.ones(a.shape[0]))
        C = torch.where((n > 0)[:, None], S / n.clamp_min(1)[:, None], C)
        print(it, flush=True)
    return C


t0 = time.time()
C = kmeans(X, X[torch.randperm(N, generator=g)[:K]].clone(), 16)
Xt = torch.randn(500_000, 8, generator=g)
mse = ((Xt - C[assign(Xt, C)]) ** 2).mean().item()
OUT.mkdir(parents=True, exist_ok=True)
torch.save(C, OUT / "cb8_16_gauss_retrained.pt")
print(f"trained in {time.time() - t0:.0f}s; held-out N(0,1) MSE per dim {mse:.5f} (train {N}, 16 iterations)", flush=True)
