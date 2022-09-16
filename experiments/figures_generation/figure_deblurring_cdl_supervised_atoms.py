# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('figures_style_full.mplstyle')

data = pd.read_csv("../results/deblurring_cdl_supervised_n_atoms.csv")
# %%

psnrs = []
is_div = []
discrepancy = []
for n_atom in pd.unique(data["n_atoms"]):
    psnrs.append(data[data["n_atoms"] == n_atom]["psnr_rec"].max())
    is_div.append(data[data["n_atoms"] == n_atom]["is_rec"].min())
    discrepancy.append(data[data["n_atoms"] == n_atom]["discrepancy"].max())
# %%

fig, axs = plt.subplots(1, 3)
axs[0].plot(pd.unique(data["n_atoms"]), psnrs)
axs[0].set_xlabel("n atoms")
axs[0].set_ylabel("PSNR")
axs[0].grid()

axs[1].plot(pd.unique(data["n_atoms"]), is_div)
axs[1].set_xlabel("n atoms")
axs[1].set_ylabel("IS div")
axs[1].grid()

axs[2].plot(pd.unique(data["n_atoms"]), discrepancy)
axs[2].set_xlabel("n atoms")
axs[2].set_ylabel("discrepancy")
axs[2].grid()

plt.savefig("../figures/deblurring_cdl_supervised_atoms.pdf")
# %%
