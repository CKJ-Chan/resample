from ldm_inverse.condition_methods import get_conditioning_method
from ldm.models.diffusion.ddim import DDIMSampler
from data.dataloader import get_dataset, get_dataloader
from scripts.utils import clear_color, mask_generator, save_image_grid
import matplotlib.pyplot as plt
from ldm_inverse.measurements import get_noise, get_operator
import torch.nn.functional as F
from functools import partial
import numpy as np
from model_loader import load_model_from_config, load_yaml
import os
import torch
import torchvision.transforms as transforms
import argparse
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from skimage.metrics import peak_signal_noise_ratio as psnr
from src.operators.tile_center import A as A_tile, AT as AT_tile
from src.data.micro_dataset import build_image_dataset
from src.utils.io_helpers import save_tensor_as_image


def get_model(args):
    config = OmegaConf.load(args.ldm_config)
    model = load_model_from_config(config, args.diffusion_config)

    return model


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--ldm_config', default="configs/latent-diffusion/ffhq-ldm-vq-4.yaml", type=str)
    parser.add_argument('--diffusion_config', default="models/ldm/model.ckpt", type=str)
    parser.add_argument('--task_config', default="configs/tasks/gaussian_deblur_config.yaml", type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument("--data-root", type=str, default=None,
                        help="Root directory that contains input images. Overrides the value from the task config.")
    parser.add_argument("--save-root", type=str, default=None)
    parser.add_argument("--s", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--dc-weight", type=float, default=None)
    parser.add_argument("--split", type=str, default=None,
                        help="Optional split name (e.g. train/test). When provided the data root and output directory"
                             " are resolved relative to this split.")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of worker processes for the PyTorch dataloader.")

    parser.add_argument('--ddim_steps', default=500, type=int)
    parser.add_argument('--ddim_eta', default=0.0, type=float)
    parser.add_argument('--n_samples_per_class', default=1, type=int)
    parser.add_argument('--ddim_scale', default=1.0, type=float)
    return parser


def _override_task_config(args, task_config):
    if args.data_root is not None:
        if isinstance(task_config.get("data"), dict):
            split_root = os.path.join(args.data_root, args.split) if args.split else args.data_root
            task_config["data"]["root"] = split_root
        else:
            task_config["data_root"] = os.path.join(args.data_root, args.split) if args.split else args.data_root
    if args.save_root is not None:
        task_config["save_root"] = args.save_root
    if args.s is not None:
        task_config["s"] = args.s
    if args.steps is not None:
        task_config["steps"] = args.steps
    if args.dc_weight is not None:
        task_config["dc_weight"] = args.dc_weight
    return task_config


def run(args):
    task_config = load_yaml(args.task_config)
    task_config = _override_task_config(args, task_config)

    device_str = f"cuda:0" if torch.cuda.is_available() else 'cpu'
    print(f"Device set to {device_str}.")
    device = torch.device(device_str)

    if task_config.get("task_name") == "tile_center":
        dataset = build_image_dataset(root=task_config["data_root"], image_size=task_config["image_size"])
        loader = torch.utils.data.DataLoader(dataset, batch_size=task_config["batch_size"], shuffle=False)
        for batch in loader:
            gt = batch["img"].to(device)
            B, C, H, W = gt.shape
            s = int(task_config["s"])
            with torch.no_grad():
                y = A_tile(gt, s)
            if task_config.get("save_measurements", True):
                save_tensor_as_image(y, out_dir=os.path.join(task_config["save_root"], f"s{s}", "meas"))
            recon = AT_tile(y, s, H, W)
            if task_config.get("save_recon", True):
                save_tensor_as_image(recon, out_dir=os.path.join(task_config["save_root"], f"s{s}", "recon"))
            save_tensor_as_image(gt, out_dir=os.path.join(task_config["save_root"], f"s{s}", "gt"))
        raise SystemExit

    model = get_model(args)
    sampler = DDIMSampler(model)  # Sampling using DDIM

    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    print(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], model, operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    print(f"Conditioning sampler : {task_config['conditioning']['main_sampler']}")

    sample_fn = partial(sampler.posterior_sampler, measurement_cond_fn=measurement_cond_fn, operator_fn=operator.forward,
                        S=args.ddim_steps,
                        cond_method=task_config['conditioning']['main_sampler'],
                        conditioning=None,
                        ddim_use_original_steps=True,
                        batch_size=args.n_samples_per_class,
                        shape=[3, 64, 64],  # Dimension of latent space
                        verbose=False,
                        unconditional_guidance_scale=args.ddim_scale,
                        unconditional_conditioning=None,
                        eta=args.ddim_eta)

    out_path = os.path.join(args.save_dir, args.split) if args.split else args.save_dir
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label', 'panels']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    data_config = task_config['data']
    dataset_root = data_config.get('root', None)
    if dataset_root is None:
        raise ValueError("Task configuration must define data.root when running sample_condition.")
    print(f"Loading dataset '{data_config['name']}' from {dataset_root}")

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=args.num_workers, train=False)

    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(**measure_config['mask_opt'])
    else:
        mask_gen = None

    for i, ref_img in enumerate(loader):
        print(f"Inference for image {i}")
        fname = str(i).zfill(3)
        ref_img = ref_img.to(device)

        if measure_config['operator']['name'] == 'inpainting':
            mask = mask_gen(ref_img)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)
            operator_fn = partial(operator.forward, mask=mask)
            measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
            sample_fn_local = partial(sample_fn, measurement_cond_fn=measurement_cond_fn, operator_fn=operator_fn)

            y = operator_fn(ref_img)
            y_n = noiser(y)
            y_vis = y_n

        else:
            y = operator.forward(ref_img)
            y_n = noiser(y).to(device)
            y_vis = F.interpolate(y, size=ref_img.shape[-2:], mode="nearest")
            sample_fn_local = sample_fn

        samples_ddim, _ = sample_fn_local(measurement=y_n, H=ref_img.shape[-2], W=ref_img.shape[-1])

        x_samples_ddim = model.decode_first_stage(samples_ddim.detach())

        label = clear_color(y_vis)
        reconstructed = clear_color(x_samples_ddim)
        true = clear_color(ref_img)

        plt.imsave(os.path.join(out_path, 'input', fname + '_true.png'), true)
        plt.imsave(os.path.join(out_path, 'label', fname + '_label.png'), label)
        save_image_grid([ref_img[0].detach().cpu(), y_vis[0].detach().cpu(), x_samples_ddim[0].detach().cpu()],
                        os.path.join(out_path, 'panels', f'{fname}_gt_meas_recon.png'))
        plt.imsave(os.path.join(out_path, 'recon', fname + '_recon.png'), reconstructed)

        psnr_cur = psnr(true, reconstructed)

        print('PSNR:', psnr_cur)


def main(cli_args=None):
    parser = build_arg_parser()
    args = parser.parse_args(cli_args)
    run(args)


if __name__ == "__main__":
    main()
