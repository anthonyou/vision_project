from problems.base_problems import *

class InpaintingProblem(BaseProblemTransferMatrix2DObsGaussianNoise):
  # simply remove pixels at random
  def __init__(self, masking_ratio=0.5, *args, **kwargs):
    self.masking_ratio = masking_ratio
    super().__init__(*args, **kwargs)

  def construct_A_mat(self):
    N = self.obs_size[0] * self.obs_size[1]
    # all channels are either visible or masked
    A_mat_without_C = np.eye(N)
    mask = np.random.uniform(size=N) < self.masking_ratio
    A_mat_without_C = mask[:,None] * A_mat_without_C

    A_mat_H, A_mat_W = A_mat_without_C.shape
    self.A_mat = np.zeros((self.C * N, self.C * N))

    for c_i in range(self.C):
      self.A_mat[c_i * A_mat_H:(c_i + 1) * A_mat_H, c_i * A_mat_W:(c_i + 1) * A_mat_W] = A_mat_without_C

if __name__ == '__main__':
  img = np.array(cv2_imread('img_examples/pug.png'), dtype='float32')
  img = cv2_resize(img, (64,64)) / 255.0

  inverse_problem = InpaintingProblem(img=img, obs_size=img.shape[1:], gaussian_noise_std=0.05)
  obs = inverse_problem.forward()
  imshow(img, title='img')
  imshow(obs, title='obs')

  recovered_img = inverse_problem.explicit_solve()
  imshow(recovered_img, title='recovered_img')