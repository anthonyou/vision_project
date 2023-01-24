from problems.base_problems import *


class IdentityProblem(BaseProblemTransferMatrix2DObsGaussianNoise):
  def construct_A_mat(self):
    N = self.C * self.obs_size[0] * self.obs_size[1]
    self.A_mat = np.eye(N)
