"""Terminal summary of every experiment, for quick status checks."""
import glob, json, os, sys
import numpy as np

RUNS = os.environ.get("RUNS", "runs")


def load(pattern):
    out = []
    for f in glob.glob(f"{RUNS}/{pattern}/result.json"):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        r["name"] = os.path.basename(os.path.dirname(f))
        out.append(r)
    return out


print("=" * 68)
print("ADDITION — exact match at ONE forward pass (grid 11)")
print("=" * 68)
rows = {}
for r in load("g11-*"):
    a = r["args"]
    rows.setdefault((a["digits"], a["mode"]), []).append(r["accuracy_by_passes"])
print(f"{'digits':>7} {'method':<8}{'1 pass':>9}{'16 pass':>9}{'seeds>90%':>11}")
for k in sorted(rows):
    v = rows[k]
    one = [x["1"] for x in v]
    six = [x["16"] for x in v]
    print(f"{k[0]:>7} {k[1]:<8}{np.mean(one)*100:8.1f} {np.mean(six)*100:8.1f} "
          f"{sum(o > 0.9 for o in one):>6}/{len(one)}")

print()
print("=" * 68)
print("INFLECTION — shallow dependencies, where no method is needed (grid 12)")
print("=" * 68)
rows = {}
for r in load("g12-*"):
    a = r["args"]
    rows[(a["lang"], a["mode"])] = r["accuracy_by_passes"]
print(f"{'language':<11}{'mode':<7}{'1 pass':>9}{'32 pass':>9}{'gap':>8}")
for k in sorted(rows):
    v = rows[k]
    one, many = v["1"] * 100, v["32"] * 100
    print(f"{k[0]:<11}{k[1]:<7}{one:8.1f} {many:8.1f} {many - one:+7.1f}")

print()
print("=" * 68)
print("HEAD-TO-HEAD vs INFERENCE-TIME METHODS")
print("=" * 68)
for f in sorted(glob.glob("figures/beat_d*.json")) + sorted(glob.glob("../figures/beat_d*.json")):
    b = json.load(open(f))
    s = b.get("summary", {})
    print(f"{b['digits']}-digit:")
    for k, v in b["inference_time_on_baseline"].items():
        print(f"   {k:<16} 1pass={v[0]*100:6.1f}   best-any-budget={max(v)*100:6.1f}")
    print(f"   {'OURS':<16} 1pass={b['ours']['confidence'][0]*100:6.1f}")
