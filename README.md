# Change Detection Using DINOv3

Building change detection on satellite image time series, using [SpaceNet-7](https://spacenet.ai/sn7-challenge/) monthly imagery and the satellite-pretrained DINOv3 backbone (`facebook/dinov3-vitl16-pretrain-sat493m`).

A small temporal transformer looks at all monthly DINOv3 features of a location at once and predicts, for every month-to-month step, **which pixels changed** (buildings appearing or disappearing), plus a **building mask** for every month as an auxiliary task. Part 1 also derives per-building change types (stable, new, demolished, expanded, reduced, modified) from the footprints; those are label statistics, not model outputs.

## Pipeline

| Notebook | What it does |
|---|---|
| `00_data_exploration.ipynb` | Explores the raw dataset before any processing: inventory, month coverage, one location over time, building counts for every location, and how cloud masks distort the labels (masked fraction vs. building-count drop: correlation 0.96) |
| `01_config_and_labels.ipynb` | Builds the config, extracts per-pixel and per-building change labels from the monthly footprint GeoJSONs, records which pixels are cloud-masked each month, and creates the train/val/test split |
| `02_feature_extraction_dinov3_sat.ipynb` | Extracts DINOv3 SAT-493M patch features for every monthly image. Default: 1024 px input → 64×64 patches, compressed 1024 → 256 dims with PCA fitted on training locations |
| `03_training.ipynb` | Trains a temporal transformer on the features to predict change between consecutive months (plus building masks), ignoring cloud-masked pixels, with early stopping; then evaluates on held-out test locations |
| `04_embedding_analysis.ipynb` | Uses the DINOv3 embeddings directly: training-free change baselines (cosine distance between months, k-means cluster switches), embedding maps over time, similarity search by example, and a comparison table of every run |
| `05_levir_cd_bitemporal.ipynb` | The same method on a second dataset, LEVIR-CD (0.5 m aerial image pairs, two dates): DINOv3 features at 1024 px with PCA, the difference head and the cross-attention head, threshold on validation, test split scored once |
| `99_reset_training_run.ipynb` | Utility, not a pipeline step: archives or deletes one training run's files so it can be retrained under the same name (see [Retraining and the reset utility](#retraining-and-the-reset-utility)) |

Every notebook is self-contained and runs on **Google Colab**. Data and outputs live on Google Drive.

## Running it

1. **Get the data.** Download `SN7_buildings_train` from the [SpaceNet dataset on AWS Open Data](https://registry.opendata.aws/spacenet/) and put it on your Google Drive at:
   ```
   MyDrive/datasets/spacenet7/SN7_buildings_train/train/
   ```
   It should contain 60 `L15-*` location folders. If you use a different path, change `data_root` in Step 5 of the notebook.
2. **Run the notebooks in order** in Colab, using the badge at the top of each one:
   - **Part 0** is optional and runs on a CPU runtime; it only reads the raw data.
   - **Part 1** runs on a free CPU runtime. Its last steps build the cloud-mask maps and the train/val/test split.
   - **Part 2** needs a GPU runtime (the free **T4** is enough) and a Hugging Face token. See [Data and model licenses](#data-and-model-licenses).
   - **Part 3** needs a GPU runtime (T4) and loads all features (~3 GB) into RAM.
   - **Part 4** runs on a CPU runtime (a GPU helps a little) and also loads all features into RAM.
   - **Part 5** is independent of Parts 1–4. It needs a T4 and `HF_TOKEN`. For the data, either put the LEVIR-CD zip at `MyDrive/datasets/levir_cd_cache/levir-cd.zip`, or add Kaggle credentials to Colab Secrets (`KAGGLE_USERNAME`, `KAGGLE_KEY`) and it downloads the ~2.6 GB dataset once. Its outputs go to `MyDrive/datasets/levir_cd_cache/`.
3. **Outputs** go to `MyDrive/datasets/spacenet7_cache/`:
   - `config.json`: settings shared by all notebooks
   - `labels/<location>_labels.npz`: change labels for each location (Part 1)
   - `masks/<location>_masks.npz`: cloud-masked 128×128 cells for each month (Part 1)
   - `training/split.json`: train / validation / test locations, shared by Parts 2 and 3 (Part 1)
   - `<features folder>/<location>_features.npz` + `pca.npz`: DINOv3 features, ~50 MB per location, ~3 GB total (Part 2). The folder is `features_1024_pca256` by default; `features` holds the original 32×32 run
   - `training/<run name>/`: checkpoints (`best.pt`, `last.pt`), history, test metrics, and figures for each training run (Part 3)

Every notebook can be resumed: if Colab disconnects, run all cells again. Parts 1 and 2 skip locations that are already done, and Part 3 continues the run from its last checkpoint.

### Retraining and the reset utility
Part 3's resume logic has one side effect: a run that has already finished can't be retrained from scratch just by running Part 3 again, because it loads the finished checkpoint and stops. Two ways to handle that:
- **New experiment:** set a new `RUN_NAME`. Each run gets its own folder, and nothing is overwritten. This is the normal path.
- **Same name, from scratch:** run `99_reset_training_run.ipynb` for that run name first. It previews, then archives (or deletes) only that run's six files, never the split, labels, masks, or features.

The reset utility was needed exactly once in this project: after the cloud-mask fix, the first run's results were invalid, so its files were archived and Part 3 was retrained on the corrected labels. The archived files are the "Raw labels (invalid)" row in the results table.

### Experiments
Part 3 selects the features, the model variant, the loss, and a run name in its Step 3, so different settings can be compared on the same split:

| Run name | Features | `MODEL_VARIANT` | `LOSS_REGION` | Idea |
|---|---|---|---|---|
| `grid32` | `features` (512 px, 1024 dims, ~150 m patches) | `baseline` | `dice` | First honest run |
| `grid64_pca256` (default) | `features_1024_pca256` (1024 px, PCA to 256 dims, ~75 m patches) | `baseline` | `dice` | Higher resolution |
| `grid64_xattn_tversky` | `features_1024_pca256` | `xattn` | `tversky` | Each patch of month t+1 attends to all patches of month t (spatial cross-attention) before the difference is taken; Focal-Tversky loss weights misses more than false alarms |

Part 4 adds two **training-free** detectors on the same split and metrics, so the gain from training can be measured: cosine distance between a patch's embeddings in consecutive months, and k-means cluster switches. It also writes `training/results_summary.md` with every run and baseline in one table.

## Data handling notes

Most of the work in this project went into the data, not the model. These are the things that were not obvious from the dataset description and that the notebooks handle explicitly (Part 0 shows each of them in the raw data).

**Cloud masks are the biggest issue, and they're incomplete.**
- `images_masked/` has clouds blacked out, and buildings under those areas are left out of that month's labels. So a building "disappears" when a cloud covers it and "reappears" afterwards. Across all 60 locations, **88% of the raw change-label pixels** fall inside masked areas, which cover only ~6% of the image; the correlation between a month's masked fraction and its drop in building count is **0.96**.
- The cloud-mask rasters in `UDM_masks/` exist **only for cloudy months and only in 44 of the 60 locations**, so they can't be used directly as a per-month mask. Part 1 (Step 13) instead derives the mask from the black pixels of `images_masked/`, which matches the UDM rasters where they exist (99.6% agreement) and also covers the no-data edges. Every month gets a mask, empty where nothing is masked.
- One location (`L15-1848E-0793N`) is 55% masked on average; 21 locations have at least one month where more than 20% of the buildings vanish because of clouds.

**Months are missing, and the gaps are uneven.** Locations have between 18 and 26 months; 34 of them skip at least one calendar month, so consecutive frames can be 1–4 months apart. Part 3 gives the model the actual calendar gap between frames, not just their order, and the labels, features, and masks are all keyed by `YYYY_MM` so they can never be misaligned.

**Images are not all the same size.** Tiles are 1024×1024 or 1023×1024 pixels, while the UDM rasters are always 1024×1024. All masks are built from the image itself, so this never causes an off-by-one, and both images and label masks are resampled to the working grids (512 or 1024 px for DINOv3, 128×128 for the targets).

**Building `Id`s are matched across months, but only inside a location.** A per-building history (new / persisting / demolished) can be read straight from `labels_match/`; `labels/` has the unmatched footprints and the UDM polygons and is not used.

**Google Drive is slow for many small files.** Every file open on a Drive mount costs a network round trip, and the dataset is ~1,400 images plus ~1,400 GeoJSONs. Parts 1 and 2 copy what they need to Colab's local disk with 32 parallel threads before processing, upload results in the background, and skip locations already done, so a disconnected session resumes where it stopped.

**The feature files had to be kept small.** DINOv3 features at 1024 px are 64×64 patches × 1,024 dims, about 12.6 GB for the dataset, more than Colab's RAM. Part 2 compresses them to 256 dims with PCA fitted on training locations only (91.4% of the variance kept), which brings the total to ~3 GB, the same as the 32×32 run.

## Why we looked at the embeddings directly (Part 4)

Parts 2 and 3 only ever use the DINOv3 features as *input to a trained model*. But the features are embeddings: vectors whose distances mean something, which is why they're increasingly used for similarity search, clustering, and as predictors in spatial analysis (see Esri's [*Including Embeddings in Your Spatial Analysis Workflows*](https://www.esri.com/arcgis-blog/products/arcgis-pro/geoai/including-embeddings-in-your-spatial-analysis-workflows)). Part 4 uses them that way, to answer one question: **how much of the trained model's skill comes from the embeddings themselves, and how much from training?**

- **Training-free change detection is nearly useless here.** Cosine distance between a patch's embeddings in consecutive months scores F1 0.013–0.019 on the test locations, 2–3× chance and ~10× below the trained model. Seasons, lighting and sun angle move every patch's embedding (median cosine distance 0.15 for unchanged patches) by about as much as construction does (0.23), with heavily overlapping distributions. k-means cluster switches do no better. So the model's skill is learned: it finds *which* embedding movements mean buildings.
- **The embeddings are still meaningful.** Their top principal components, shown as RGB, are stable month to month and read like a land-cover map, with cloud-masked areas appearing as their own distinct region. Similarity search from a newly built patch finds other built-up patches, and the patch's similarity to its own post-construction embedding jumps from 0.03 to ~0.9 and stays there in every later month. Built-up land is stable and distinct in embedding space; the *moment* of change is what's hard.
- **One table for everything.** Part 4 writes `training/results_summary.md` with every trained run and every baseline on the same split and metrics.

Following the embedding guidance: embeddings are used at the scale of the analysis (one per ~75 m patch), never averaged over space or time, never standardized, with all dimensions weighted equally, and never mixed with other variables in a similarity metric.

## Evaluation

Part 1 splits the 60 labeled locations **by location** into train / validation / test (42 / 9 / 9). Part 2 fits its PCA on training locations only. In Part 3, validation picks the checkpoint, early stopping, and the decision threshold; the test locations are scored once, at the end.

**Cloud masks:** in SpaceNet-7, buildings under cloud-masked parts of a month are left out of that month's labels, so they look like they vanish and reappear. About **88% of raw change-label pixels lie inside masked areas**, which cover only ~6% of the image. A model trained on those labels mostly learns to find the black masked regions. Part 3 therefore **ignores every pixel that is masked in either month of a pair**, in both the loss and the metrics.

Reported scores are **pixel-level precision, recall, F1, and IoU for change between consecutive months, on a 128×128 grid** (~32 m cells), **excluding cloud-masked pixels**. They are **not** the official SpaceNet-7 SCOT metric, which tracks individual buildings at full resolution, so they can't be compared directly with the challenge leaderboard.

## Results

All runs: same 42 / 9 / 9 split, free Colab T4, threshold chosen on validation, test scored once.

| Run | Features | Test change F1 | Test change IoU | Test precision / recall | Test building IoU | Training time |
|---|---|---|---|---|---|---|
| Raw labels (invalid) | 32×32 × 1024 | 0.593 | 0.422 | 0.643 / 0.551 | 0.619 | 12 min |
| Cloud masks ignored | 32×32 × 1024 | 0.137 | 0.073 | 0.151 / 0.125 | 0.623 | 12 min |
| **Cloud masks ignored, higher resolution** | **64×64 × 256 (PCA)** | **0.178** | **0.097** | **0.199 / 0.160** | **0.704** | 20 min (early-stopped at epoch 36, best 26) |
| + spatial cross-attention + Focal-Tversky loss | 64×64 × 256 (PCA) | 0.176 | 0.096 | 0.170 / 0.183 | 0.679 | 32 min (early-stopped at epoch 35, best 25) |
| *No training:* cosine distance between months (tile-median removed) | 64×64 × 256 (PCA) | 0.019 | 0.010 | 0.011 / 0.074 | | seconds |
| *No training:* k-means cluster switch (k = 8) | 64×64 × 256 (PCA) | 0.018 | 0.009 | 0.010 / 0.093 | | seconds |
| *No training:* cosine distance between months (raw) | 64×64 × 256 (PCA) | 0.013 | 0.007 | 0.007 / 0.125 | | seconds |

What the rows mean:
- **Raw labels (invalid):** trained on the labels as extracted. It scored well by finding the black cloud-masked regions, not building change. Reported only as a warning; do not cite it.
- **Cloud masks ignored:** the honest baseline. Change pixels are ~0.3% of the data, so random guessing scores F1 ≈ 0.006; the model is far above chance but misses most single-building changes, because a 512 px input gives ~150 m patches.
- **Higher resolution:** 1024 px input (~75 m patches) with PCA-compressed features, plus stronger regularization (dropout 0.2, weight decay 0.05) and early stopping. +30% relative F1, better building segmentation, and much less overfitting (train/val loss gap 0.11 vs 0.39). Validation F1 (0.219) is higher than test F1 because validation also chose the checkpoint and threshold.

- **Cross-attention + Tversky:** the same F1 within noise (0.176 vs 0.178). The Tversky loss did what it's meant to, trading precision for recall (recall 0.16 → 0.18), but spatial cross-attention added nothing measurable: SpaceNet-7 mosaics are already well co-registered. Not worth its extra cost.
- **Training-free baselines** (Part 4): raw embedding distance between months scores only 2–3× chance (F1 ≈ 0.006). Seasonal and lighting changes move every patch's embedding by about as much as real construction does (median cosine distance 0.15 for unchanged patches vs 0.23 for changed ones, with heavily overlapping distributions). The trained model is ~10× better than the best baseline, so nearly all of its skill comes from learning *which* embedding movements mean buildings.

Building segmentation (IoU 0.70) is solid; detecting the appearance of individual small buildings between two months remains the hard part. Part 4's similarity search shows why embeddings still help: a patch's similarity to its own "after construction" embedding jumps from 0.03 to ~0.9 and stays there in every later month, so built-up land is stable and distinct in embedding space even though the *moment* of change is not.

## Second dataset: LEVIR-CD (Part 5)

To see how the recipe behaves on very different data, Part 5 runs it on [LEVIR-CD](https://justchenhao.github.io/LEVIR/): 637 pairs of 0.5 m aerial images (1024×1024) taken 5–14 years apart, with a clean binary building-change mask per pair, split 445 / 64 / 128 for train / val / test. Compared with SpaceNet-7 it has 8× the ground resolution, two dates instead of a monthly series, no cloud artifacts, and ~15× more change pixels (~4–5%).

The notebook is self-contained and independent of Parts 1–4: it takes the dataset zip from Drive (or downloads it from Kaggle once), extracts DINOv3 SAT-493M features at 1024 px with PCA to 256 dims (fitted on training pairs only), max-pools the masks to the same 128×128 grid (8 m cells here), and trains the **difference head** and the **cross-attention head** with the same loss, schedule, early stopping, and threshold protocol as Part 3. The test split is scored once.

Results on the 128 test pairs (free T4; PCA keeps 87.9% of the variance; features for all 637 pairs took 8 min):

| Head | Parameters | Best epoch | Threshold | Val F1 | Test precision | Test recall | Test F1 | Test IoU |
|---|---|---|---|---|---|---|---|---|
| Difference head | 0.56M | 35 of 39 | 0.35 | 0.908 | 0.907 | 0.914 | **0.910** | 0.835 |
| Cross-attention head | 0.83M | 31 of 39 | 0.35 | 0.909 | 0.908 | 0.908 | 0.908 | 0.831 |

Two things stand out:
- **The same recipe that reaches F1 0.18 on SpaceNet-7 reaches 0.91 on LEVIR-CD.** The method isn't the limit; the data is. LEVIR-CD has 8× the ground resolution, two clean dates years apart, no cloud artifacts, and ~15× more change per pair, so the frozen DINOv3 features separate new buildings from everything else almost perfectly, and training converges in a few minutes.
- **Cross-attention doesn't help here either** (0.908 vs 0.910), on the second dataset in a row, with two very different image sources. For co-registered pairs, comparing each patch with itself is enough.

Note that published LEVIR-CD scores (F1 ≈ 0.90) are pixel-level at full resolution with fully trained networks; these are 128×128 cell-level scores (8 m cells) on frozen features, so they're comparable with the SpaceNet-7 rows above, not with the leaderboard.

## Data and model licenses

This repository contains **code only**. It does not include the dataset, derived labels, or model weights.

- **LEVIR-CD** (Part 5) is provided by its authors for research use; cite:
  > H. Chen, Z. Shi. *A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection.* Remote Sensing, 2020.
- **SpaceNet-7** is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). If you publish labels or results derived from it, the same license applies. Please cite:
  > A. Van Etten, D. Hogan, J. Martinez Manso, J. Shermeyer, N. Weir, R. Lewis. *The Multi-Temporal Urban Development SpaceNet Dataset.* CVPR 2021.
- **DINOv3** weights are gated on Hugging Face under Meta's DINOv3 License. Accept the license on the [model page](https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m) and add a Hugging Face token to **Colab Secrets** as `HF_TOKEN`. Never paste the token into a notebook cell: Colab saves cell outputs when it commits to GitHub.
  > O. Siméoni et al. *DINOv3.* arXiv:2508.10104, 2025.

## License

The code in this repository is released under the [MIT License](LICENSE).
