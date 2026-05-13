# RareGen

RareGen is a PyTorch-based generative model repository for training an adversarial autoencoder-style GAN/VAE hybrid on image data. The project contains training and evaluation scripts for image generation, checkpointing, and sample generation.

## Project Structure

- `codes/`
  - `main.py` - main training script for the encoder-decoder-discriminator model.
  - `ganvae1.py`, `ganwave.py`, `ganwave1.py`, `ganwave2.py` - additional GAN/VAE-related experiments and workflows.
  - `evaluate.py`, `eval_gan.py` - evaluation utilities for generated images and models.
- `batch_scripts/`
  - `run_gan.sh` - example SLURM batch script for launching training on a GPU cluster.
  - `run_eval.sh`, `run_gw.sh`, `rungw1.sh` - additional batch job examples.

## Requirements

- Python 3.8+ (recommended)
- PyTorch
- torchvision
- Pillow
- tqdm

You can install the main dependencies with pip:

```bash
pip install torch torchvision pillow tqdm
```

## Usage

1. Update dataset and output paths in `codes/main.py`:

```python
DATASET_PATH = "/path/to/your/dataset"
SAVE_DIR = "/path/to/save/checkpoints"
SAMPLE_DIR = "/path/to/save/samples"
```

2. Make sure the dataset directory contains image files with `.png` or `.jpg` extensions.

3. Run training:

```bash
python codes/main.py
```

4. During training, model checkpoints are saved every 5 epochs and sample images are written to `SAMPLE_DIR`.

## Notes

- `codes/main.py` uses a default image size of `512x512` and a latent dimension of `256`.
- The training script currently hardcodes dataset and checkpoint directories, so update those before running.
- The discriminator uses spectral normalization and hinge losses in the adversarial training loop.

## Customization

- Adjust `IMG_SIZE`, `BATCH_SIZE`, `LATENT_DIM`, `EPOCHS`, and learning rate settings at the top of `codes/main.py`.
- You can add or modify model components in `codes/main.py` to experiment with different generator/discriminator architectures.

## License

This repository does not include a license file. Add a license if you want to share or publish the code.
