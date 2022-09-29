# Prior learning in unsupervised inverse problems

## installation
```
pip install -e .
```

## content
* Main code in **dl_inv_prob**
* Experiments in **experiments**: the computations have been performed on a GPU NVIDIA Tesla V100-DGXS 32GB
    - Figure 1: a few hours
    ```
    python experiments/scripts/partial_rec.py
    python experiments/figure_generation/figure_1.py
    ```
    - Figure 2: a few hours
    ```
    python experiments/scripts/num_measurements.py
    python experiments/scripts/inpainting_patches.py
    python experiments/figure_generation/figure_2.py
    ```
    - Figure 3 and C, D: 1 day
    ```
    python experiments/scripts/single_image_inpainting.py
    python experiments/scripts/single_image_inpainting_supervised.py
    python experiments/scripts/pnp_inpainting.py
    python experiments/figure_generation/figure_3.py
    python experiments/figure_generation/figure_3_2.py
    ```
    - Figure 4 and B: 1h
    ```
    python experiments/scripts/inpainting_color.py
    python experiments/figure_generation/figure_4.py
    ```
    - Figure 5: A few hours
    ```
    python experiments/scripts/inpainting_cdl_digits_score.py
    python experiments/figure_generation/figure_5.py
    ```
    - Figure 6 and E, F: 1 day
    ```
    python experiments/scripts/single_image_deblurring.py
    python experiments/scripts/single_image_deblurring_supervised.py
    python experiments/scripts/pnp_deblurring.py
    python experiments/figure_generation/figure_6.py
    python experiments/figure_generation/figure_6_2.py
    ```
    - Figure 7: 1h
    ```
    python experiments/scripts/deblurring_color.py
    python experiments/figure_generation/figure_7.py
    ```
    - Figure 8: A few hours
    ```
    python experiments/scripts/deblurring_pnp_supervised_examples.py
    python experiments/scripts/deblurring_cdl_supervised_n_atoms.py
    python experiments/figure_generation/figure_8_1.py
    python experiments/figure_generation/figure_8_2.py
    ```
    - Figure A: 100h
    ```
    python experiments/scripts/benchmark_invprob_dl.py
    python experiments/figure_generation/figure_A.py
    ```
