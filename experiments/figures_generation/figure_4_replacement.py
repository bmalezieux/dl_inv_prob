import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from pathlib import Path

def load_jsonl(path):
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]
    return pd.DataFrame(data)

def plot_with_error_bands(df, save_path):
    sns.set(style="whitegrid")
    dim_images = sorted(df['dim_image'].unique())
    palette = sns.color_palette("husl", len(dim_images))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

    for idx, dim in enumerate(dim_images):
        df_dim = df[df['dim_image'] == dim]
        color = palette[idx]

        # Group by sparsity to calculate mean and std
        grouped = df_dim.groupby('s')

        stats = grouped.agg({
            'psnr': ['mean', 'std'],
            'psnr_corrupted': ['mean', 'std'],
            'score_weighted': ['mean', 'std']
        }).reset_index()
        stats.columns = ['s', 'psnr_mean', 'psnr_std', 'psnr_cor_mean', 'psnr_cor_std', 'score_mean', 'score_std']

        # PSNR + PSNR corrupted
        ax = axes[0]
        
        n_patches = ((dim - 10) // 5 + 1) ** 2
        
        ax.plot(stats['s'], stats['psnr_mean'], label=f'PSNR (n. patches={n_patches})', color=color, linewidth=2)
        ax.fill_between(stats['s'], 
                        stats['psnr_mean'] - stats['psnr_std'], 
                        stats['psnr_mean'] + stats['psnr_std'], 
                        color=color, alpha=0.2)

        ax.plot(stats['s'], stats['psnr_cor_mean'], label=f'PSNR_corr (n. patches={n_patches})', linestyle='--', color=color, linewidth=2)
        ax.fill_between(stats['s'], 
                        stats['psnr_cor_mean'] - stats['psnr_cor_std'], 
                        stats['psnr_cor_mean'] + stats['psnr_cor_std'], 
                        color=color, alpha=0.2)

        # Weighted Score
        ax = axes[1]
        ax.plot(stats['s'], stats['score_mean'], label=f'n. patches={n_patches}', color=color, linewidth=2)
        ax.fill_between(stats['s'], 
                        stats['score_mean'] - stats['score_std'], 
                        stats['score_mean'] + stats['score_std'], 
                        color=color, alpha=0.2)

    # Final formatting
    axes[0].set_title('PSNR and PSNR Corrupted vs Sparsity')
    axes[0].set_xlabel('Sparsity (s)')
    axes[0].set_ylabel('PSNR')
    axes[0].legend(loc="upper right")

    axes[1].set_title('Weighted Score vs Sparsity')
    axes[1].set_xlabel('Sparsity (s)')
    axes[1].set_ylabel('Score Weighted')
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path / "figure_4.pdf")

if __name__ == "__main__":

    path_results = Path(__file__).parent.parent / "results" / "inpainting_patches.jsonl"
    save_path = Path(__file__).parent.parent / "figures"
    save_path.mkdir(parents=True, exist_ok=True)
    df = load_jsonl(path_results)
    plot_with_error_bands(df, save_path)
