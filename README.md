# 3D-Lung-Tumor-Segmentation-with-nnU-Net
# 3D Lung Tumor Segmentation with nnU-Net (MSD Task06 Lung)

3D volumetric tumor segmentation on chest CT using the self-configuring **nnU-Net v2**
framework. This is the **3D segmentation pillar** of a medical image segmentation
portfolio. Its 2D counterpart is a hand-built U-Net pipeline for breast-ultrasound
lesion segmentation on the cleaned BUSI dataset. Together they cover 2D and 3D, and a
custom pipeline alongside a standard self-configuring framework.

## Why this project

This project extends the **scope from 2D to 3D**: full volumetric segmentation of lung 
tumors in CT, where context across slices matters and the
network uses 3D convolutions. It demonstrates running a complete 3D medical segmentation
pipeline and reading and reporting its results.

## Dataset

Medical Segmentation Decathlon, Task06 Lung (http://medicaldecathlon.com/).

- 63 labeled training CT volumes (image + expert mask), 32 unlabeled test volumes.
- Single channel (CT). Labels: background = 0, cancer = 1.
- Licence: CC-BY-SA 4.0.
- Test labels are withheld, so evaluation uses nnU-Net's cross-validation on the labeled
  training set, not the test set.

## Method

nnU-Net v2 is a self-configuring segmentation framework. Rather than hand-designing the
network and preprocessing, it analyzes the dataset (a "fingerprint") and automatically
sets the spacing, normalization, patch size, network depth, and training schedule. Its
architecture is a U-Net; the value is the automated, dataset-adaptive configuration.

Configuration used (compute-limited on an 8 GB RTX 5050):

- Configuration: `3d_lowres`
- Fold: 0 (one fold of the 5-fold cross-validation)
- Trainer: `nnUNetTrainer_250epochs` (250 epochs instead of the 1000-epoch default)

These choices trade some peak accuracy for feasible training time on entry hardware and
are reported as such.

### What nnU-Net configured (from the plan)

For `3d_lowres` on this dataset:

- Target spacing: 2.39 x 1.50 x 1.50 mm (resampled from each scan's native spacing)
- Patch size: 80 x 192 x 160
- Batch size: 2
- Normalization: CT scheme, clip to the 0.5 to 99.5 foreground intensity percentiles, then
  z-score with the foreground mean and standard deviation
- Network: 6-stage U-Net, feature maps 32 up to 320, five downsampling steps

Preprocessing applied automatically: crop to the nonzero region, resample to the target
spacing, CT intensity normalization, then foreground-oversampled 3D patch sampling with
on-the-fly augmentation (rotation, scaling, noise, blur, brightness/contrast, gamma,
mirroring).

## Pipeline

Single entry point that runs convert, preprocess, train, and evaluate through the nnU-Net
v2 CLI.

```bash
python pipeline.py                                # full run
python pipeline.py --no-convert --no-preprocess   # train + evaluate only
python pipeline.py --no-train                      # everything except training
```

Key files:

- `pipeline.py`  orchestrator
- `config.py`    paths and run settings (dataset id, configuration, fold, trainer, workers)

## Results

5-fold cross-validation, fold 0, evaluated on the 13 held-out labeled cases that the model
never saw during training.

| Metric        | Value |
|---------------|-------|
| Mean Dice     | 0.772 |
| Mean IoU      | 0.648 |
| Median Dice   | ~0.836 |
| Dice range    | 0.38 to 0.887 |
| Cases >= 0.80 | 9 of 13 |

The mean is pulled down by a few small or ambiguous tumors; the median reflects the
typical case. MSD lung is one of the harder Decathlon tasks (small, sparse tumors in large
volumes), so a mean Dice in this range is competitive. A reduced-budget single-fold run is
expected to sit slightly below a full 5-fold, full-resolution, 1000-epoch run.

The pseudo dice seen during training (~0.80) is nnU-Net's on-the-fly monitoring estimate
on patches. The 0.772 above is the proper validation Dice computed on full volumes and is
the number reported.

### Reading the results

Metrics live in `fold_0/validation/summary.json`:

- `foreground_mean.Dice` is the headline (0.772).
- `metric_per_case` lists each of the 13 cases with Dice, IoU, and the voxel counts
  TP, FP, FN, TN.
- Dice = 2*TP / (2*TP + FP + FN); IoU = TP / (TP + FP + FN). TN is huge because most of the
  volume is background, which is why overlap metrics, not accuracy, are used.


## Output files (fold_0)

- `checkpoint_best.pth`, `checkpoint_final.pth`  trained weights (best epoch and last epoch)
- `summary.json`        validation metrics
- `progress.png`        training curves (loss, pseudo dice, epoch time, learning rate)
- `training_log_*.txt`  per-session logs
- `debug.json`          full run configuration for reproducibility
- `validation/*.nii.gz` predicted masks for the held-out cases

## Limitations

- Single fold of the 5-fold cross-validation, not the full ensemble.
- `3d_lowres`, not `3d_fullres`, and 250 epochs, not the 1000-epoch default. Compute-limited
  on entry hardware.
- Lung CT, not breast ultrasound. 3D breast ultrasound (TDSC-ABUS) is the intended next step.

## References

- Isensee et al. nnU-Net: a self-configuring method for deep learning-based biomedical image
  segmentation. Nature Methods, 2021.
- Antonelli et al. The Medical Segmentation Decathlon. Nature Communications, 2022.
