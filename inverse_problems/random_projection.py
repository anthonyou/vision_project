from my_python_utils.common_utils import *

from base_problems import *
from scipy import sparse

from numpy.random import default_rng

class RandomProjectionWithGaussianNoise(BaseProblemTransferMatrix2DObsGaussianNoise):
  def construct_A_mat(self):
    rng = default_rng(124)
    self.A_mat = sparse.random(self.C * self.obs_size[0] * self.obs_size[1], self.C * self.img_size[0] * self.img_size[1],
                               density=0.01, format='coo', dtype=None, data_rvs=None, random_state=rng)
    self.A_mat = np.array(self.A_mat)

if __name__ == '__main__':
  img = np.array(cv2_imread('img_examples/pug.png'), dtype='float32')
  img = cv2_resize(img, (16,16)) / 255.0

  noise_std = 0.05

  inverse_problem = RandomProjectionWithGaussianNoise(img=img, obs_size=img.shape[1:], gaussian_noise_std=noise_std)
  obs = inverse_problem.forward_process()
  imshow(img, title='img', biggest_dim=256)
  imshow(obs, title='obs', biggest_dim=256)

  recovered_img = inverse_problem.explicit_solve()
  imshow(recovered_img, title='recovered_img', biggest_dim=256)