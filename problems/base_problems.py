import numpy as np
import torch

from my_python_utils.common_utils import *


# Base class that defines all problems
class BaseProblem():
  # class to be inherited by other problems
  def __init__(self, img=None, obs=None, img_size=None):
     # assert (img is None) + (obs is None) == 1, "Either img (for simulation) or obs (for real data) should be provided!"

    self.img = img
    self.obs = obs
    self.tensor_type = 'numpy'

    if not self.obs is None:
      assert not img_size is None, "Recovery size should be provided if the recovery is performed from an observation"
      self.img_size = img_size
      self.obs_size = obs.shape[1:]

      assert len(img.shape) == 3, "Img should be a tensor of shape CxHxW, and is {}".format(img.shape)
      self.C = self.obs.shape[0]

    elif not img is None:
      assert len(img.shape) == 3, "Img should be a tensor of shape CxHxW, and is {}".format(img.shape)
      assert img.min() >= 0 and img.max() <= 1 and img.dtype=='float32', "Img should be passed as float array, normalized between 0 and 1"
      self.img_size = self.img.shape[1:]

      self.C = self.img.shape[0]
    else:
      assert img_size is not None and len(img_size)==3, "When run in implicit mode, img_size must be (C,H,W)"
      if obs_size is None:
        obs_size = img_size[1:]
      self.C = self.img_size[0]
      self.img_size = self.img_size[1:]

  def forward(self):
    raise Exception("Must be implemented by children class")

  def explicit_solve(self):
    raise Exception("Must be implemented by children class")

# Problem that is defined by a linear transfer matrix, with 2D observations
class BaseProblemTransferMatrix2DObsGaussianNoise(BaseProblem):
  def __init__(self, obs=None, img=None, img_size=None, obs_size=None, gaussian_noise_std=0.1):
    super().__init__(obs=obs, img=img, img_size=img_size)

    if img is not None and obs is None:
      assert not obs_size is None, "Observation size should be provided"
      assert len(obs_size) == 2, "Observation size should be HxW"
      self.obs_size = obs_size
      assert obs_size[0] <= self.img_size[0] and obs_size[1] <= self.img_size[1]

    self.gaussian_noise_std = gaussian_noise_std
    self.construct_A_mat()

  def construct_A_mat(self):
    raise Exception("Needs to be implemented by base class")

  def forward(self):
    assert not self.img is None
    noiseless_obs = self.A_mat @ (self.img.reshape(-1))
    if self.gaussian_noise_std > 0:
      obs = noiseless_obs + np.random.normal(scale=self.gaussian_noise_std, size=noiseless_obs.shape)
    else:
      obs = noiseless_obs
    obs = obs.reshape(self.C, *self.obs_size)
    return obs

  def explicit_solve(self):
    if self.obs_size == self.img_size:
      if self.obs is None:
        obs = self.forward()
      else:
        obs = self.obs
      if type(self.A_mat) is np.ndarray:
        inv_mat = np.linalg.inv(self.A_mat)
      else:
        import scipy.sparse.linalg as linalg
        inv_mat = linalg.inv(self.A_mat)

      return (inv_mat @ obs.reshape(-1)).reshape((self.C, *self.obs_size))
    else:
      # TODO: solve with regularization when it is underconstrained.
      # e.g. ridge regression
      raise Exception("To be implemented")

  def get_obs(self, batch_img):
    assert not batch_img is None
    batch_size = batch_img.shape[0]
    flatten_img = torch.stack([img.flatten() for img in batch_img]).T
    noiseless_obs = torch.sparse.mm(self.A_torch, flatten_img).T
    if self.gaussian_noise_std > 0:
      noise = self.gaussian_noise_std * torch.randn(noiseless_obs.shape)
      noise = noise.to(self.device)
      obs = noiseless_obs + noise
    else:
      obs = noiseless_obs
    obs = obs.reshape(batch_size, self.C, *self.obs_size)
    return obs
    
  def init_torch(self, device):
    A = self.A_mat
    self.device = device
    values, indices = torch.DoubleTensor(A.data), torch.LongTensor(np.vstack((A.row, A.col)))
    self.A_torch = torch.sparse.DoubleTensor(indices, values, torch.Size(A.shape)).float()
    self.A_torch = self.A_torch.to(device).requires_grad_(False)
    
    if self.img is not None:
        self.img = torch.from_numpy(self.img).to(device).unsqueeze(0)
    if self.obs is not None:
        self.obs = torch.from_numpy(self.obs).to(device).unsqueeze(0)
    self.tensor_type = 'torch'

  def torch_forward(self, batch_img, obs_shape):
    flatten_img = torch.stack([img.flatten() for img in batch_img]).T
    return torch.sparse.mm(self.A_torch, flatten_img).T.view(obs_shape)
