import numpy as np
import cv2

import urllib.request
import requests
from PIL import Image
import matplotlib.pyplot as plt
import scipy
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
from numpy.random import default_rng
import torch

home_dir = '/data/vision/torralba/scratch/aou/vision_project'
device = 0

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


def normalize_and_show_image(img, name='low_dog.jpeg', save=True):
    norm_img = img - np.min(img) 
    norm_img = norm_img / np.max(norm_img)
    resized_img = image_resize(norm_img, 512, 512)
    if save:
        plt.imsave(f'{home_dir}/classical_images/{name}', resized_img)
    return norm_img


def move_to_obs(recovered_img, true_obs, A, lr=0.0005, T = 10000, save_intermediate=False):
    recovered_img = torch.from_numpy(recovered_img).cuda().requires_grad_(True)
    true_obs = torch.from_numpy(true_obs).cuda().requires_grad_(False)
    
    values, indices = torch.DoubleTensor(A.data), torch.LongTensor(np.vstack((A.row, A.col)))
    A = torch.sparse.DoubleTensor(indices, values, torch.Size(A.shape))
    A = A.cuda().requires_grad_(False)
    for i in range(T):
        if i > 0.9 * T:
            lr = lr / 10
        recovered_obs = torch.sparse.mm(A, recovered_img.flatten().unsqueeze(0).T.double())
        recovered_obs = recovered_obs.view(true_obs.shape)
        loss = torch.mean((recovered_obs - true_obs)**2)
        if i % 100 == 0:
            print(loss)
            if save_intermediate:
                saved_img = recovered_img.detach().cpu().numpy()
                normalized_img = normalize_and_show_image(saved_img, 'intermediate_dog.jpeg')
                
        loss.backward(retain_graph=True)
        recovered_img.data = recovered_img.data - lr * recovered_img.grad.data
        recovered_img.grad.zero_()
    return recovered_img.detach().cpu().numpy()

L = 512
url = 'https://i.guim.co.uk/img/media/fe1e34da640c5c56ed16f76ce6f994fa9343d09d/0_174_3408_2046/master/3408.jpg?width=1200&height=1200&quality=85&auto=format&fit=crop&s=67773a9d419786091c958b2ad08eae5e'
I = Image.open(requests.get(url, stream=True).raw)
img = np.array(I)
img = image_resize(img, L, L)
img = img.astype('uint8')
norm_img = normalize_and_show_image(img, 'med_dog.jpeg')

x = norm_img.flatten()
N = x.shape[0]
print(N)
# select random set of indices of A that will be zero, to make A non-invertible and simulate an ill-posed forward problem

rng = default_rng(123)
A = sparse.random(N, N, density=0.0001, format='coo', dtype=None, random_state=rng, data_rvs=None)
obs = A @ x
obs = np.asarray(obs)
obs = obs.reshape(img.shape)
obs_with_noise = obs + np.random.normal(size=obs.shape) * 0.3
np.save(f'{home_dir}/classical_images/obs.npy', obs)
np.save(f'{home_dir}/classical_images/obs_with_noise.npy', obs_with_noise)

norm_obs = normalize_and_show_image(obs, 'new_obs.jpeg')
norm_obs_with_noise = normalize_and_show_image(obs_with_noise, 'obs_with_noise.jpeg')

implicit_invert = True
if implicit_invert:
    start_img = Image.open(f'{home_dir}/classical_images/intermediate_dog.jpeg')
    start_img = np.array(start_img)
    if start_img.shape[0] != start_img.shape[1]:
        min_dim = min(start_img.shape[0], start_img.shape[1])
        start_img = start_img[:min_dim, :min_dim, :]
    start_img = image_resize(start_img, L, L)
    norm_start_img = normalize_and_show_image(start_img, 'start_image.jpeg')
    
    with torch.cuda.device(device):
        inverse = move_to_obs(norm_start_img, obs_with_noise, A, T =100000, lr=5, save_intermediate=True)
    norm_recovered_noisy_obs = normalize_and_show_image(inverse, 'recovered_noisy_obs.jpeg')

explicit_invert = False
if explicit_invert:
    inv = linalg.inv(A)
    recovered_obs = inv @ obs.flatten().transpose()
    recovered_obs = np.asarray(recovered_obs)
    recovered_obs = recovered_obs.reshape(img.shape)
    norm_recovered_obs = normalize_and_show_image(recovered_obs, 'recovered_obs.jpeg')

    recovered_noisy_obs = inv @ obs_with_noise.flatten().transpose()
    recovered_noisy_obs = np.asarray(recovered_noisy_obs)
    recovered_noisy_obs = recovered_noisy_obs.reshape(img.shape)
    norm_recovered_noisy_obs = normalize_and_show_image(recovered_noisy_obs, 'recovered_noisy_obs.jpeg')
