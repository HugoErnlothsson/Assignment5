# task1.py
import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("images", exist_ok=True)

def get_ground_level(z, bins=200):
    z = z[np.isfinite(z)]
    counts, edges = np.histogram(z, bins=bins)
    i = np.argmax(counts)
    ground = 0.5*(edges[i] + edges[i+1])
    return float(ground), (counts, edges)

def plot_hist(hist, tag):
    counts, edges = hist
    centers = 0.5*(edges[:-1] + edges[1:])
    plt.figure()
    plt.bar(centers, counts, width=(edges[1]-edges[0]))
    plt.xlabel("z")
    plt.ylabel("count")
    plt.title(f"Histogram z ({tag})")
    plt.tight_layout()
    out = f"images/hist_{tag}.png"
    plt.savefig(out, dpi=200)
    plt.close()
    return out

def load_xyz(npy_path):
    arr = np.load(npy_path)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"{npy_path}: expected (N,3)")
    return arr[:, 0], arr[:, 1], arr[:, 2]

def main():
    # Hittar .npy i ./ eller ./data/
    files = sorted(glob.glob("*.npy") + glob.glob("data/*.npy"))
    if not files:
        raise SystemExit("Inga .npy-filer hittades")
    results = {}
    for path in files:
        tag = os.path.splitext(os.path.basename(path))[0]
        _, _, z = load_xyz(path)
        ground, hist = get_ground_level(z, bins=200)
        img = plot_hist(hist, tag)
        results[tag] = {"ground_level": ground, "histogram_png": img}
        print(f"{tag}: ground_level = {ground:.6f}  (bild: {img})")
    with open("results_task1.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
