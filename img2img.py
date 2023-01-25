"""make variations of input image"""

import argparse, os, sys, glob, shutil
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

home_dir = '/data/vision/torralba/scratch/aou/vision_project'
exp_count = len(os.listdir(f'{home_dir}/experiment_images'))-1
assert os.path.isdir(f'{home_dir}/experiment_images/experiment_{exp_count:05}/img2img_samples')

config = {
    'prompt': 'a clear photograph of a sitting pug puppy',
    'init-img': f'{home_dir}/classical_images/recovered_extra_noisy_obs.jpeg',
    'outdir': f'{home_dir}/experiment_images/experiment_{exp_count:05}',
    'skip_grid': False,
    'ddim_steps': 50,
    'fixed_code': False,
    'ddim_eta': 0.0,
    'n_iter': 1,
    'C': 4,
    'f': 8,
    'n_samples': 1,
    'scale': 5.0,
    'strength': 0.9,
    'decay_rate': 0.99,
    'min_strength': 0.01,
    'config': f'{home_dir}/stable_diffusion/v1-inference.yaml',
    'ckpt': f'{home_dir}/stable_diffusion/model.ckpt',
    'seed': 42,
    'precision': 'autocast'
}

shutil.copy(config['init-img'], f'{home_dir}/experiment_images/experiment_{exp_count:05}/start_img.jpeg')

with open(f'{home_dir}/experiment_images/experiment_{exp_count:05}/config.txt', 'a') as f: 
    f.writelines([f'{key}: {value}\n' for key, value in config.items()])

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def get_img2img_iterator():
    seed_everything(config['seed'])
    model_config = OmegaConf.load(config['config'])
    model = load_model_from_config(model_config, config['ckpt'])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = DDIMSampler(model)

    outpath = config['outdir']

    batch_size = config['n_samples']
    n_rows = batch_size
    prompt = config['prompt']
    data = [batch_size * [prompt]]
    print('Constructing generator that uses prompt:', data)

    def img2img(iter_num):
        if iter_num > 0:
            init_filename = f'{home_dir}/experiment_images/experiment_{exp_count:05}/grad_descent_samples/{iter_num-1:05}.jpeg'
        else:
            init_filename = config['init-img']
        print('Loading image from:', init_filename)
        assert os.path.isfile(init_filename)
        init_image = load_img(init_filename).to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

        sample_path = os.path.join(outpath, "img2img_samples")

        sampler.make_schedule(ddim_num_steps=config['ddim_steps'], ddim_eta=config['ddim_eta'], verbose=False)
        
        strength = max(config['strength'] * config['decay_rate']**iter_num, config['min_strength'])
        assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(strength * config['ddim_steps'])
        print(f"target t_enc is {t_enc} steps")

        precision_scope = autocast if config['precision'] == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    tic = time.time()
                    all_samples = list()
                    for n in trange(config['n_iter'], desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            uc = None
                            if config['scale'] != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = model.get_learned_conditioning(prompts)

                            # encode (scaled latent)
                            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                            # decode it
                            samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=config['scale'], unconditional_conditioning=uc,)

                            x_samples = model.decode_first_stage(samples)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                            for x_sample in x_samples:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                Image.fromarray(x_sample.astype(np.uint8)).save(
                                    os.path.join(sample_path, f"{iter_num:05}.png"))
                            all_samples.append(x_samples)

                    toc = time.time()

        print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
              f" \nEnjoy.")
    return img2img


if __name__ == "__main__":
    generator = get_img2img_iterator()
    generator(0)
