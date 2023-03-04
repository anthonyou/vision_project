from my_python_utils.common_utils import *
from matplotlib.pyplot import imsave
import os
import numpy as np

for file in os.listdir('/data/vision/torralba/scratch/aou/vision_project/synthetic_dataset'):
    img = np.array(cv2_imread(os.path.join('/data/vision/torralba/scratch/aou/vision_project/synthetic_dataset', file)) / 255.0, dtype='float32')
    img = best_centercrop_image(img, 512, 512)
    imsave(os.path.join('/data/vision/torralba/scratch/aou/vision_project/synthetic_baseline/', file), img.transpose([1, 2, 0]))
