import numpy as np
import matplotlib.pyplot as plt

images = ["flowers", "forest", "animal", "mushroom"]

for image in images:

    D = np.load(f"../results/dictionaries/inpainting_dict_{image}.npy")

    fig, axs = plt.subplots(8, 5, figsize=(15, 15))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(D[i, :, :], cmap="gray")
        ax.axis("off")

    plt.savefig(f"../figures/inpainting_dict_{image}.png")
