#%% 
import os
import json
import numpy as np

import matplotlib.pyplot as plt

#%% Functions 
def get_ground_level(z, bins=200):
    z = np.asarray(z)
    z = z[np.isfinite(z)]
    counts, edges = np.histogram(z, bins=bins)
    i = int(np.argmax(counts))
    ground = 0.5*(edges[i] + edges[i+1])
    return float(ground), (counts, edges)

def plot_hist(hist, tag):
    counts, edges = hist
    centers = 0.5*(edges[:-1] + edges[1:])
    plt.figure()
    plt.bar(centers, counts, width=(edges[1]-edges[0]))
    plt.xlabel("z"); plt.ylabel("count"); plt.title(f"Histogram z ({tag})")
    plt.tight_layout()
    plt.show()

#%% Inputs 
files = ["dataset1.npy", "dataset2.npy"]

#%% Compute ground level per file
results = {}
for path in files:
    tag = os.path.splitext(os.path.basename(path))[0]
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"{path}: expected shape (N,3)")
    z = arr[:, 2]
    ground, hist = get_ground_level(z, bins=200)
    results[tag] = {"ground_level": ground}
    print(f"{tag}: ground_level = {ground:.6f}")

#%% Plot histograms
for path in files:
    tag = os.path.splitext(os.path.basename(path))[0]
    arr = np.load(path)
    z = arr[:, 2]
    _, hist = get_ground_level(z, bins=200)
    plot_hist(hist, tag)

#%%
