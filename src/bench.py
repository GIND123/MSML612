"""Throughput of the ACTUAL Sudoku model on whatever GPU this job got.

The class allocation bills an H100 at 144/min and a 5GB MIG slice at 9/min - 16x
apart. That only matters if the MIG slice is less than 16x slower on OUR
workload, which is 81-token sequences and a ~32M-parameter model: a size where an
H100 is largely idle. Measured rather than assumed.
"""
import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Transformer
import diffusion as dfn
from sudoku import CELLS, SudokuTokenizer

dev = "cuda"
tok = SudokuTokenizer()
m = Transformer(len(tok), 512, 10, 8, "ape", causal=False, max_len=CELLS + 8).to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
x = torch.randint(0, 9, (128, CELLS), device=dev)
a = torch.rand(128, CELLS, device=dev) < 0.55

for _ in range(10):                      # warm up
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = dfn.pmd_loss(m, x, a, tok, "full", 8, 1.0)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
torch.cuda.synchronize()

t0 = time.time()
N = 100
for _ in range(N):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = dfn.pmd_loss(m, x, a, tok, "full", 8, 1.0)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
torch.cuda.synchronize()
el = time.time() - t0
name = torch.cuda.get_device_name(0)
print(f"GPU={name}")
print(f"  {N/el:.1f} steps/sec   ({el/N*1000:.1f} ms/step)")
print(f"  60k steps would take {60000*el/N/60:.1f} min")
