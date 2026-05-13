# RareGen

RareGen is a medical image synthesis repository for brain cancer MRI generation using hybrid GAN/VAE architectures. The project is based on the research paper `ganpaper.pdf`, which compares three main architectures:

- `WGAN-GP` (Wasserstein GAN with Gradient Penalty)
- `GAN-VAE` hybrid with progressive-resolution training
- `Wavelet-VAE-GAN` hybrid using wavelet decomposition for multi-resolution feature learning

This repository contains the training code, evaluation utilities, batch scripts, generated sample images, and the research paper.

## Research Summary

The paper assesses how synthetic MRI data can help overcome rare brain cancer dataset scarcity while preserving patient privacy and diagnostic features. The authors demonstrate that hybrid models provide a better trade-off between visual realism and structural accuracy than pure GANs or VAEs.

Key contributions:

- Hybrid model designs for synthetic brain MRI generation
- Progressive-resolution training for the GAN-VAE architecture
- Wavelet-based feature decomposition for finer tumor texture preservation
- Comprehensive evaluation with FID, SSIM, PSNR, MS-SSIM, and LPIPS
- Focus on rare brain cancer MRI data scarcity and privacy-friendly synthetic augmentation

## Dataset Description

The paper uses a brain tumor MRI dataset with 5,000 images at 512×512 resolution. The class distribution is:

- Low-grade glioma: 1,650 images (33.0%)
- High-grade glioma: 1,400 images (28.0%)
- Meningioma: 1,050 images (21.0%)
- Pituitary tumor: 900 images (18.0%)

Dataset split:

- Training: 70% (3,500 images)
- Validation: 15% (750 images)
- Testing: 15% (750 images)

Images are standardized to 512×512 and augmented with rotation, translation, zoom, and brightness adjustments for robust training.

## Repository Structure

- `codes/`
  - `main.py` - primary training script for the encoder-decoder-discriminator GAN-VAE model
  - `ganvae1.py` - progressive-resolution GAN-VAE implementation with grayscale MRI training and EMA support
  - `ganwave.py`, `ganwave1.py`, `ganwave2.py` - wavelet-augmented GAN/VAE experiments and workflow variants
  - `evaluate.py` - dataset evaluation utilities and metrics
  - `eval_gan.py` - checkpoint evaluation with FID, SSIM, PSNR, and LPIPS support
- `batch_scripts/`
  - `run_gan.sh` - SLURM batch job example for GPU cluster training
  - `run_eval.sh` - evaluation batch example
  - `run_gw.sh`, `rungw1.sh` - additional workflow scripts
- `generated_images/` - generated sample outputs from trained models
- `ganpaper.pdf` - research paper describing the project

## Model Architectures and Training

### `codes/main.py`

- Encoder/Decoder/Discriminator architecture with residual blocks
- Spectral normalization in the discriminator
- Hinge adversarial loss
- Reconstruction loss, KL divergence, adversarial loss, and feature matching
- Default image size: 512×512
- Latent dimension: 256
- Checkpoints saved every 5 epochs
- Sample outputs saved during training

### `codes/ganvae1.py`

- Grayscale progressive-resolution GAN-VAE
- Starts at 64×64 and progressively grows to 512×512
- Uses InstanceNorm, residual up/down blocks, and EMA for stable generation
- Includes KL warmup and adversarial ramping schedules

### `codes/ganwave.py` and variants

- Wavelet-based VAE-GAN hybrid methods
- Multi-resolution feature extraction using wavelet decomposition
- Designed to preserve fine tumor texture and spatial detail

## Evaluation Metrics

The project evaluates generated MRI samples using the following metrics:

- FID (Fréchet Inception Distance): measures distribution similarity between real and generated images
- SSIM (Structural Similarity Index): measures perceptual and structural quality
- PSNR (Peak Signal-to-Noise Ratio): measures pixel-level reconstruction fidelity
- MS-SSIM: multi-scale structural similarity
- LPIPS: learned perceptual similarity

## Results from the Paper

The paper reports that the GAN-VAE hybrid produced the best balance between realism and structural correctness. Highlights include:

- GAN-VAE achieved the strongest overall performance with FID values in the range of 60–100
- Wavelet-VAE-GAN delivered more stable training and better preservation of fine texture details
- WGAN-GP produced sharper images, but with relatively higher FID and lower structural fidelity

The conclusion states that GAN-VAE is the preferred hybrid model in this work, providing a reliable synthetic MRI generation pipeline for rare cancer image augmentation.

## Generated Output Examples

The following images are selected sample outputs from `generated_images/`. These represent some of the strongest model outputs produced by the project.

![Generated MRI sample 1](generated_images/generated_1.png)

![Generated MRI sample 100](generated_images/generated_100.png)

![Generated MRI sample 200](generated_images/generated_200.png)

![Generated MRI sample 300](generated_images/generated_300.png)

## Installation

Install the required packages:

```bash
pip install torch torchvision pillow tqdm
```

For additional evaluation scripts, install:

```bash
pip install torchmetrics lpips scikit-image pandas
```

## How to Run

### Train the main model

1. Update the path variables in `codes/main.py`:

```python
DATASET_PATH = "/path/to/your/dataset"
SAVE_DIR = "/path/to/save/checkpoints"
SAMPLE_DIR = "/path/to/save/samples"
```

2. Run training:

```bash
python codes/main.py
```

### Run progressive GAN-VAE training

```bash
python codes/ganvae1.py
```

### Evaluate a checkpoint

```bash
python codes/eval_gan.py
```

## Notes and Best Practices

- Ensure the dataset directory contains `.png`, `.jpg`, or `.jpeg` image files.
- Use a GPU-enabled environment for best performance.
- If you use the SLURM script, update `run_gan.sh` with your cluster paths and environment.
- The repository currently does not include an explicit license file.

## Future Work Mentioned in the Paper

- Further integration of diffusion-based priors
- Spatial attention mechanisms for improved texture fidelity
- Larger-scale synthetic datasets for rare tumor subtypes
- Better interpretability and clinical relevance for synthetic medical images

## How to Cite

If you use this repository or the methods from the research paper, please cite the project as a hybrid GAN-VAE/Wavelet-VAE-GAN study for synthetic brain MRI generation focused on rare cancer data augmentation.
