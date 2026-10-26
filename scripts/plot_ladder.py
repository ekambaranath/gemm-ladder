"""Plot the GEMM ladder from results/gemm_ladder.csv -> docs/gemm_ladder.png

    python scripts/plot_ladder.py results/gemm_ladder.csv docs/gemm_ladder.png

Two panels, because fp32 and tensor-core results live on very different scales:
  left : the fp32 ladder k1..k6 as % of cuBLAS fp32 SGEMM (the optimization story)
  right: absolute TFLOP/s of the best fp32 rung and the tensor-core rung vs both cuBLAS baselines
"""
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "results/gemm_ladder.csv"
dst = sys.argv[2] if len(sys.argv) > 2 else "docs/gemm_ladder.png"

rows = {r["kernel"]: r for r in csv.DictReader(open(src))}
size = next(iter(rows.values()))["size"]
fp32 = [k for k in rows if k.startswith("k") and rows[k]["baseline"] == "cublas_fp32"]
tc = [k for k in rows if k.startswith("k") and rows[k]["baseline"] == "cublas_tc_fp16"]

BLUE, GREEN, GREY, RED = "#2F6DB5", "#3C9D5D", "#9AA5B1", "#C0504D"
plt.rcParams.update({"font.size": 9})
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})

# left: fp32 ladder, % of cuBLAS
pct = [float(rows[k]["pct_of_cublas"]) for k in fp32]
bars = a.bar([k.replace("_", "\n", 1) for k in fp32], pct,
             color=[RED if rows[k]["status"] == "FAIL" else BLUE for k in fp32])
for bar, k in zip(bars, fp32):
    a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
           f"{float(rows[k]['pct_of_cublas']):.0f}%\n{float(rows[k]['tflops']):.2f} TF", ha="center", fontsize=8)
a.axhline(100, color="#333", ls="--", lw=1)
a.text(len(fp32) - 0.5, 102, "cuBLAS fp32 = 100%", ha="right", fontsize=8, color="#333")
a.set_ylim(0, 120)
a.set_ylabel("% of cuBLAS fp32 SGEMM (higher is better)")
a.set_title(f"fp32 ladder, M=N=K={size}", fontsize=10)
a.spines[["top", "right"]].set_visible(False)

# right: absolute TFLOP/s
best_fp32 = max(fp32, key=lambda k: float(rows[k]["tflops"]))
items = [("cuBLAS\nfp32", float(rows["cublas_fp32"]["tflops"]), GREY),
         (best_fp32.replace("_", "\n", 1), float(rows[best_fp32]["tflops"]), BLUE)]
if tc:
    items += [(tc[0].replace("_", "\n", 1), float(rows[tc[0]]["tflops"]), GREEN),
              ("cuBLAS\ntensor cores", float(rows["cublas_tc_fp16"]["tflops"]), GREY)]
bars = b.bar([i[0] for i in items], [i[1] for i in items], color=[i[2] for i in items])
for bar, it in zip(bars, items):
    b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6, f"{it[1]:.1f}", ha="center", fontsize=8)
b.set_ylabel("TFLOP/s")
b.set_title("Best fp32 rung and tensor-core rung", fontsize=10)
b.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(dst, dpi=160)
print("wrote", dst)
