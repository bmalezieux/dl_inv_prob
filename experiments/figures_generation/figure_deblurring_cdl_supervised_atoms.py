# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('figures_style_small.mplstyle')

data = pd.read_csv("../results/deblurring_cdl_supervised_n_atoms.csv")
# %%

psnrs = []
is_div = []
discrepancy = []
discrepancy_weighted = []
for n_atom in pd.unique(data["n_atoms"]):
    psnrs.append(data[data["n_atoms"] == n_atom]["psnr_rec"].max())
    is_div.append(data[data["n_atoms"] == n_atom]["is_rec"].min())
    discrepancy.append(data[data["n_atoms"] == n_atom]["discrepancy"].max())
    discrepancy_weighted.append(data[data["n_atoms"] == n_atom]["discrepancy_weighted"].max())
# %%

plt.plot(pd.unique(data["n_atoms"]), discrepancy, label="not weighted")
plt.plot(pd.unique(data["n_atoms"]), discrepancy_weighted, label="weighted")
plt.xlabel("Number atoms")
plt.ylabel("Discrepancy")
plt.grid()
plt.legend()

plt.savefig("../figures/deblurring_cdl_supervised_atoms.pdf")
# %%
