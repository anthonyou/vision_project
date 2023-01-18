import numpy as np
import torch
import cv2

import urllib.request
import requests
from PIL import Image
import matplotlib.pyplot as plt
import scipy
import scipy.sparse as sparse
from numpy.random import default_rng
import os

home_dir = '/data/vision/torralba/scratch/aou/vision_project'
exp_count = len(os.listdir(f'{home_dir}/experiment_images'))-1
assert os.path.isdir(f'{home_dir}/experiment_images/experiment_{exp_count:05}/grad_descent_samples')

loss_cutoff = 200000
with open(f'{home_dir}/experiment_images/experiment_{exp_count:05}/config.txt', 'a') as f:
    f.write(f'Loss cutoff: {loss_cutoff}\n')

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

def normalize_and_save_image(img, name='default.jpeg', save=True):
    norm_img = img - np.min(img) 
    norm_img = norm_img / np.max(norm_img)
    resized_img = image_resize(norm_img, 512, 512)
    if save:
        plt.imsave(f'{home_dir}/experiment_images/experiment_{exp_count:05}/grad_descent_samples/{name}', resized_img)
    return norm_img

def move_to_obs(recovered_img, true_obs, A, lr=0.0005):
    recovered_img = torch.from_numpy(recovered_img).cuda().requires_grad_(True)
    true_obs = torch.from_numpy(true_obs).cuda().requires_grad_(False)
    
    values, indices = torch.DoubleTensor(A.data), torch.LongTensor(np.vstack((A.row, A.col)))
    A = torch.sparse.DoubleTensor(indices, values, torch.Size(A.shape))
    A = A.cuda().requires_grad_(False)
    loss = float('inf')
    i = -1
    while loss > loss_cutoff:
        i += 1
        recovered_obs = torch.sparse.mm(A, recovered_img.flatten().unsqueeze(0).T.double())
        recovered_obs = recovered_obs.view(true_obs.shape)
        loss = torch.sum((recovered_obs - true_obs)**2)
        if i % 10 == 0:
            print(loss)
        loss.backward(retain_graph=True)
        recovered_img.data = recovered_img.data - lr * recovered_img.grad.data
        recovered_img.grad.zero_()
    return recovered_img.detach().cpu().numpy()

noisy_obs = np.load(f'{home_dir}/classical_images/obs_with_noise.npy')

N = noisy_obs.flatten().shape[0]
rng = default_rng(123)
A = sparse.random(N, N, density=0.0001, format='coo', dtype=None, random_state=rng, data_rvs=None)

def iterate_move_to_obs(iter_num):
    recovered_img = Image.open(f'{home_dir}/experiment_images/experiment_{exp_count:05}/img2img_samples/{iter_num:05}.png')
    recovered_img = np.array(recovered_img)
    recovered_img = image_resize(recovered_img, 512, 512)
    normalized_recovered_img = normalize_and_save_image(recovered_img, save=False)
    closest_img = move_to_obs(normalized_recovered_img, noisy_obs, A)
    normalized_closest_img = normalize_and_save_image(closest_img, f'{iter_num:05}.jpeg')

