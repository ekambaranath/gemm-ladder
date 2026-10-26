"""Plot the GEMM ladder from results/gemm_ladder.csv -> docs/gemm_ladder.png

    python scripts/plot_ladder.py results/gemm_ladder.csv docs/gemm_ladder.png
"""
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "results/gemm_ladder.csv"
dst = sys.argv[2] if len(sys.argv) > 2 else "docs/gemm_ladder.png"

rows = list(csv.DictReader(open(src)))
size = rows[0]["size"]
refs = {r["kernel"]: float(r["tflops"]) for r in rows if r["status"] == "ref"}
rungs = [r for r in rows if r["status"] in ("PASS", "FAIL")]

BLUE, GREEN, GREY, RED = "#2F6DB5", "#3C9D5D", "#6B7785", "#C0504D"
fig, ax = plt.subplots(figsize=(8.5, 4.2))
names = [r["kernel"].replace("k", "k", 1) for r in rungs]
vals = [float(r["tflops"]) for r in rungs]
cols = [RED if r["status"] == "FAIL" else (GREEN if r["baseline"] == "cublas_tc_fp16" else BLUE) for r in rungs]
bars = ax.bar(names, vals, color=cols)
for b, r in zip(bars, rungs):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15,
            f"{float(r['pct_of_cublas']):.0f}%", ha="center", fontsize=8)
if "cublas_fp32" in refs:
    ax.axhline(refs["cublas_fp32"], color=GREY, ls="--", lw=1)
    ax.text(-0.4, refs["cublas_fp32"] + 0.15, f"cuBLAS fp32  {refs['cublas_fp32']:.1f} TFLOP/s", fontsize=8, color=GREY)
if "cublas_tc_fp16" in refs:
    ax.axhline(refs["cublas_tc_fp16"], color=GREEN, ls=":", lw=1)
    ax.text(-0.4, refs["cublas_tc_fp16"] + 0.15, f"cuBLAS fp16 tensor cores  {refs['cublas_tc_fp16']:.1f} TFLOP/s",
            fontsize=8, color=GREEN)
ax.set_ylabel("TFLOP/s (higher is better)")
ax.set_title(f"GEMM ladder, M=N=K={size}  (label = % of the matching cuBLAS baseline)", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
plt.xticks(rotation=20, ha="right", fontsize=8)
fig.tight_layout()
fig.savefig(dst, dpi=160)
print("wrote", dst)
