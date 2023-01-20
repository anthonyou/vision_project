from base_problems import *

# simply sums over channel dimension with a transfer matrix
class ColorizationProblem(BaseProblemTransferMatrix2DObsGaussianNoise):
  def construct_A_mat(self):
    N = self.obs_size[0] * self.obs_size[1]
    A_mat_single_channel = np.eye(N)
    self.A_mat = np.concatenate([A_mat_single_channel] * 3, axis=1)


if __name__ == '__main__':
  img = np.array(cv2_imread('img_examples/pug.png'), dtype='float32')
  img = cv2_resize(img, (64,64)) / 255.0

  inverse_problem = ColorizationProblem(img=img, obs_size=img.shape[1:], gaussian_noise_std=0.05)
  obs = inverse_problem.forward_process()
  imshow(img, title='img')
  imshow(obs, title='obs')

  recovered_img = inverse_problem.explicit_solve()
  imshow(recovered_img, title='recovered_img')