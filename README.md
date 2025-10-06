# Solving Inverse Problems with Latent Diffusion Models via Hard Data Consistency (ICLR 2024)

![example](https://github.com/soominkwon/resample/blob/main/figures/resample_ex.png)

## Abstract

In this work, we propose ReSample, an algorithm that can solve general inverse problems with pre-trained latent diffusion models. Our algorithm incorporates data consistency by solving an optimization problem during the reverse sampling process, a concept that we term as hard data consistency. Upon solving this optimization problem, we propose a novel resampling scheme to map the measurement-consistent sample back onto the noisy data manifold.

## Getting Started

### 1) Clone the repository

```
git clone https://github.com/soominkwon/resample.git

cd resample
```

<br />

### 2) Download pretrained checkpoints (autoencoders and model)

```
mkdir -p models/ldm
wget https://ommer-lab.com/files/latent-diffusion/ffhq.zip -P ./models/ldm
unzip models/ldm/ffhq.zip -d ./models/ldm

mkdir -p models/first_stage_models/vq-f4
wget https://ommer-lab.com/files/latent-diffusion/vq-f4.zip -P ./models/first_stage_models/vq-f4
unzip models/first_stage_models/vq-f4/vq-f4.zip -d ./models/first_stage_models/vq-f4
```

<br />

### 3) Set environment

We use the external codes for motion-blurring and non-linear deblurring following the DPS codebase.

```
git clone https://github.com/VinAIResearch/blur-kernel-space-exploring bkse

git clone https://github.com/LeviBorodenko/motionblur motionblur
```

Install dependencies via

```
conda env create -f environment.yaml
```

<br />

### 4) Prepare a custom super-resolution dataset (optional)

If you would like to run super-resolution on your own train/test image
collections, normalise them into the directory structure expected by the
codebase using the helper script below. The script resizes the images to
256×256 pixels (matching the default configuration) and converts them to
PNG files. Pass `--grayscale` when working with black-and-white medical
images so that each sample retains its monochrome intensity while still
providing three channels for the latent diffusion model.

```
python scripts/prepare_super_resolution_dataset.py \
    --train-dir /path/to/highres/train_images \
    --test-dir /path/to/highres/test_images \
    --output-root data/super_resolution \
    --grayscale \
    --write-manifests
```

After running the command the processed images will live under
`data/super_resolution/train` and `data/super_resolution/test`. If
`--write-manifests` is supplied you will also find
`data/super_resolution/train_images.txt` and
`data/super_resolution/test_images.txt`, which point to the processed
assets and can be consumed directly by the PyTorch Lightning training
pipeline. Point the `data.root` entry in
`configs/tasks/super_resolution_config.yaml` to the split you would like
to evaluate.

<br />

### 5) Inference

```
python3 sample_condition.py
```

The code is currently configured to do inference on FFHQ. You can download the corresponding models from https://github.com/CompVis/latent-diffusion/tree/main and modify the checkpoint paths for other datasets and models.


<br />

### Running super-resolution on Run:AI

The updated `sample_condition.py` script accepts a `--split` argument so
you can process different dataset splits in isolation. A typical Run:AI
submission might look like:

```
runai submit superres-eval \
  --image <your-container-image> \
  --project <your-project> \
  --gpu 1 \
  --volume /mnt/datasets/superres:/workspace/data/super_resolution \
  --volume /mnt/outputs/superres:/workspace/results \
  --command -- python sample_condition.py \
      --task_config configs/tasks/super_resolution_config.yaml \
      --data-root data/super_resolution \
      --split test \
      --save_dir results/super_resolution \
      --ddim_steps 200
```

Adjust the `--data-root`, `--save_dir`, image name and resource requests
as needed for your cluster environment. When `--split` is set the script
will resolve both the dataset root and the output directory relative to
the split name, simplifying the process of running multiple jobs (for
example, one per data split).

### Training custom medical super-resolution models on Run:AI

These steps extend the workflow recommended in the original
[ReSample repository](https://github.com/soominkwon/resample) and the
corresponding [ICLR 2024 paper](https://openreview.net/forum?id=j8hdRqOUhN)
so that you can fine-tune the latent diffusion model on grayscale medical
imagery.

1. **Prepare the dataset**
   * Place your high-resolution training and validation/test images in two
     separate folders.
   * Run `scripts/prepare_super_resolution_dataset.py` with `--grayscale`
     and `--write-manifests` to normalise the images, generate PNGs and
     create manifest files listing the processed data.
   * Sync the `data/super_resolution` directory to persistent storage that
     is accessible from your Run:AI cluster (for example an NFS or
     object-store backed volume).

2. **Review the training configuration**
   * Use the existing diffusion configuration in
     `configs/latent-diffusion/ffhq-ldm-vq-4.yaml` as the base model.
   * The new overlay `configs/projects/medical_super_resolution.yaml`
     points the data module to the manifests generated above and ensures
     that a single GPU is used with mixed precision. Update the batch size
     and worker counts to match the resources allocated to your job.
   * If you created only a training split, remove or comment out the
     `validation` section in the overlay or point it to the same manifest.

3. **Launch the Run:AI job**
   * Example submission (adjust paths and resources as required):

     ```
     runai submit medical-sr-train \
       --image <your-container-image> \
       --project <your-project> \
       --gpu 1 \
       --cpu 8 \
       --memory 64G \
       --volume /mnt/datasets/medical_sr:/workspace/data/super_resolution \
       --volume /mnt/outputs/medical_sr:/workspace/logs \
       --command -- bash -lc "\
         conda activate resample && \
         python src/taming-transformers/main.py \
           --base configs/latent-diffusion/ffhq-ldm-vq-4.yaml \
                 configs/projects/medical_super_resolution.yaml \
           --train True \
           --project medical-sr \
           --name runai-medical-sr"
     ```

   * The training script exposes all PyTorch Lightning trainer arguments,
     so you can add flags such as `--max_steps`, `--limit_val_batches 0.1`
     or `--gradient_clip_val 1.0` directly to the `runai submit` command
     if you need finer control.

4. **Monitor training outputs**
   * Checkpoints and tensorboard logs are written under the `logs`
     directory mounted into the container (`/workspace/logs` in the
     example above). Use Run:AI's job dashboard or stream logs with
     `runai logs medical-sr-train` to monitor progress.

5. **Evaluate the fine-tuned model**
   * Once training has converged, point `sample_condition.py` to your new
     checkpoint (using the `--diffusion_config` flag) and run inference as
     outlined earlier to validate the reconstructions on held-out scans.

<br />

## Task Configurations

```
# Linear inverse problems
- configs/tasks/super_resolution_config.yaml
- configs/tasks/gaussian_deblur_config.yaml
- configs/tasks/motion_deblur_config.yaml
- configs/tasks/inpainting_config.yaml

# Non-linear inverse problems
- configs/tasks/nonlinear_deblur_config.yaml
```

<br />

## Hyperparameter Tuning

For the best results, please refer to the hyperparameters reported in the paper. Recall that we use two types of optimizations for hard data consistency: latent space and pixel space optimization. For the fastest inference, one can use just pixel space optimization, but with a degradation in performance. One can change the splits of pixel space and latent space optimization by tuning the index split value in the main DDIM code. We suggest to use both as reported in the main paper. 

<br />


## Citation
If you find our work interesting, please consider citing

```
@inproceedings{
song2024solving,
title={Solving Inverse Problems with Latent Diffusion Models via Hard Data Consistency},
author={Bowen Song and Soo Min Kwon and Zecheng Zhang and Xinyu Hu and Qing Qu and Liyue Shen},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=j8hdRqOUhN}
}
```

