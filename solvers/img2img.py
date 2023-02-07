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
from solvers.base_solver import BaseSolver
from solvers.sgd import StochasticGradDescSolver

from my_python_utils.common_utils import *

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

    def __init__(self, problem, config, verbose=False):
        super().__init__(problem, config=config, verbose=verbose)
        seed_everything(config['seed'])
        model_config = OmegaConf.load(config['config'])
        model = load_model_from_config(model_config, config['ckpt'])

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = model.to(self.device)

        self.sampler = DDIMSampler(self.model)

        self.sgd = StochasticGradDescSolver(problem, config, verbose)


    def img2img(self, init_image, strength=None):
        batch_size = self.config['n_samples']
        n_rows = batch_size
        prompt = self.config['prompt']
        data = [batch_size * [prompt]]
        model = self.model
        print('Constructing generator that uses prompt:', data)

        init_image = init_image.to(self.device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))

        self.sampler.make_schedule(ddim_num_steps=self.config['ddim_steps'], ddim_eta=self.config['ddim_eta'], verbose=self.verbose)
        
        if strength is None:
            strength = self.config['strength']
        assert 0. <= strength <= 1.
        t_enc = int(strength * self.config['ddim_steps'])

        precision_scope = autocast if self.config['precision'] == "autocast" else nullcontext
        
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    for n in trange(self.config['n_iter'], desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            uc = None
                            if self.config['scale'] != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = model.get_learned_conditioning(prompts)

                            # encode (scaled latent)
                            z_enc = self.sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(self.device))
                            # decode it
                            samples = self.sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=self.config['scale'], unconditional_conditioning=uc,)

                            x_samples = model.decode_first_stage(samples)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        toc = time.time()
        return x_samples

    def solve(self, obs=None):
        if obs is None:
            obs = self.problem.forward()
        obs = torch.from_numpy(obs).unsqueeze(0)
        img = torch.zeros(obs.shape).to(self.device)
        logs = {'sgd_per_iter': [], 'img2img_per_iter': []}
        for i in range(self.config['iterations']):
            sgd_img = self.sgd.solve(img, obs)
            norm_img = torch.clamp(2*sgd_img-1, min=-1.0, max=1.0)
            strength = max(self.config['strength']*(self.config['decay_rate']**i), self.config['min_strength'])
            img = self.img2img(norm_img, strength=strength).float()
            if i % self.config['log_frequency'] == 0:
                logs['sgd_per_iter'].append(sgd_img)
                logs['img2img_per_iter'].append(img)
        return img[0], {k: torch.cat(v) for k, v in logs.items()}

if __name__ == "__main__":
    from solvers.config import configs, root_dir
    config = configs['img2img']

    init_filename = f'{root_dir}/classical_images/recovered_noisy_obs.jpeg'
    solver = Img2ImgSolver(config, None)
    solver.solve(init_filename)
