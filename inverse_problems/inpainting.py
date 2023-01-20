from base_problems import BaseProblemTransferMatrix2DObsGaussianNoise
from numpy import np

class InpaintingProblem():
  # simply remove pixels at random
  def __init__(self, masking_ratio=0.5, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.masking_ratio = masking_ratio

  def construct_A_mat(self):
    N = self.C * self.obs_size[0] * self.obs_size[1]
    self.A_mat = np.eye(N)



