from problems.base_problems import *
from scipy import sparse
import numpy as np


class BlurProblem(BaseProblem):
  def __init__(self, obs=None, img=None, img_size=None, obs_size=None, N=10):
    super().__init__(obs=obs, img=img, img_size=img_size)

    if obs is None:
      assert not obs_size is None, "Observation size should be provided"
      assert len(obs_size) == 2, "Observation size should be HxW"
      self.obs_size = obs_size

      assert obs_size[0] <= self.img_size[0] and obs_size[1] <= self.img_size[1]

    self.N = N
    self.construct_A_mat()

  def construct_A_mat(self):
    x_coo, y_coo, data = [], [], []

    img_H, img_W = self.img_size
    obs_H, obs_W = self.obs_size
    A_mat_H, A_mat_W = obs_H*obs_W, img_H*img_W

    for c in range(self.C):
      for i in range(img_H):
        for j in range(img_W):
          for n_i in range(self.N):
            for n_j in range(self.N):
              if img_H > i+n_i-self.N//2 >= 0 and img_W > j+n_j-self.N//2 >= 0:
                data.append(1/self.N**2)
                y_coo.append(c*A_mat_W + i*img_W + j)
                x_coo.append(c*A_mat_H + (i+n_i-self.N//2)*obs_W + (j+n_j-self.N//2))
    self.A_mat = sparse.coo_matrix((data, (x_coo, y_coo)), shape=(self.C*A_mat_H, self.C*A_mat_W))
    return self.A_mat

  def forward(self):
    assert not self.img is None
    obs = self.A_mat @ (self.img.reshape(-1))
    return obs.reshape(self.C, *self.obs_size)

if __name__ == '__main__':
  img = np.array(cv2_imread('img_examples/pug.png'), dtype='float32')
  img = cv2_resize(img, (4, 4)) / 255.0

  inverse_problem = BlurProblem(img=img, obs_size=img.shape[1:], N=3)

  obs = inverse_problem.forward()
  print(inverse_problem.A_mat.todense()[0])

