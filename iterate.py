import torch
import os

device = 0

image_dir = '/data/vision/torralba/scratch/aou/vision_project/experiment_images'
assert os.path.isdir(image_dir)
exp_count = len(os.listdir(image_dir)) 
os.makedirs(f'{image_dir}/experiment_{exp_count:05}')
print('samples saved at:', f'{image_dir}/experiment_{exp_count:05}')

with open(f'{image_dir}/experiment_{exp_count:05}/config.txt', 'w') as f:
    f.write(f'Device: {device}\n')

os.makedirs(f'{image_dir}/experiment_{exp_count:05}/grad_descent_samples')
os.makedirs(f'{image_dir}/experiment_{exp_count:05}/img2img_samples')

from align import iterate_move_to_obs
from img2img import get_img2img_iterator

with torch.cuda.device(device):
    print('Currently using device:', torch.cuda.current_device())
    img2img = get_img2img_iterator()
    for i in range(100):
        img2img(i)
        iterate_move_to_obs(i)
