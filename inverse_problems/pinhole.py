from my_python_utils.common_utils import *

from base_problem import BaseProblem
import numpy as np

class RandomProjectionWithGaussianNoise(BaseProblem):
  def __init__(self, obs=None, img=None):
    super().__init__(obs=obs, img=img)

    # assumes obs and img are the same dimensionality.

    # TODO: change to rasterization order
    self.A_mat = np.eye(obs.shape)

  def forward_process(self):
    assert not self.img is None
    return self.A_mat @ self.obs.reshape(-1)

  def explicit_solve(self):
    assert not self.obs is None
    inv_mat = np.linalg.inv(self.A_mat)

    return inv_mat @ self.obs.reshape(-1)

if __name__ == '__main__':
  img = cv