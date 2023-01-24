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
from base_solver import BaseSolver


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


class Img2ImgSolver(BaseSolver):
    """
    Solver using stablediffusion's img2img
    """

    def __init__(self, config, problem):
        super().__init__(self, problem)
        self.config = config
        seed_everything(config['seed'])
        model_config = OmegaConf.load(config['config'])
        model = load_model_from_config(model_config, config['ckpt'])

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = model.to(self.device)

        self.sampler = DDIMSampler(self.model)


    def solve(self, init_filename, strength=None):
        batch_size = config['n_samples']
        n_rows = batch_size
        prompt = config['prompt']
        data = [batch_size * [prompt]]
        model = self.model
        print('Constructing generator that uses prompt:', data)

        init_image = load_img(init_filename).to(self.device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))

        self.sampler.make_schedule(ddim_num_steps=config['ddim_steps'], ddim_eta=config['ddim_eta'], verbose=self.verbose)
        
        if strength is None:
            strength = self.config['strength']
        assert 0. <= strength <= 1.
        t_enc = int(strength * config['ddim_steps'])

        precision_scope = autocast if config['precision'] == "autocast" else nullcontext
        
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    for n in trange(config['n_iter'], desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            uc = None
                            if config['scale'] != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = model.get_learned_conditioning(prompts)

                            # encode (scaled latent)
                            z_enc = self.sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(self.device))
                            # decode it
                            samples = self.sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=config['scale'], unconditional_conditioning=uc,)

                            x_samples = model.decode_first_stage(samples)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        toc = time.time()
        return x_samples[0]

if __name__ == "__main__":
    home_dir = '/data/vision/torralba/scratch/aou/vision_project'
    config = {
        'prompt': 'a clear photograph of a sitting pug puppy',
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
    init_filename = f'{home_dir}/classical_images/recovered_noisy_obs.jpeg'
    solver = Img2ImgSolver(config, None)
    solver.solve(init_filename)
