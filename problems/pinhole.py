from problems.base_problems import *
import numpy as np

class PinholeProblem(BaseProblemTransferMatrix2DObsGaussianNoise):
  def construct_A_mat(self):
    # todo: change to sparse.lil_matrix
    A_mat_without_C = np.zeros((self.obs_size[0] * self.obs_size[1],
                                self.img_size[0] * self.img_size[1]), dtype='float32')

    # then just populate transfer Matrix
    img_H, img_W = self.img_size
    obs_H, obs_W = self.obs_size

    for i in tqdm(range(self.img_size[0])):
      for j in range(self.img_size[1]):
        # TODO: implement for when img_size != obs_size
        # we create a 1 to 1 mapping of the same resolution, and then we downscale, if necessary.
        # this allows the image to be bigger than the observation, so that we can super-resolve

        # not the most efficient approach, but good enough for us
        # cur_obs_mapping = np.zeros(self.img_size)
        #cur_obs_mapping[img_H - i - 1, img_W - j - 1] = 1
        #if self.img_size != self.obs_size:
        #  cur_obs_mapping = cv2_resize(cur_obs_mapping, self.obs_size)
        #cur_non_empty_locs = np.where(cur_obs_mapping)
        A_mat_without_C[(obs_H - i - 1) * obs_W + obs_W - j - 1, i * img_W + j] = 1

    # repeat as many times as channels to recover
    self.A_mat = np.zeros((self.C * A_mat_without_C.shape[0], self.C * A_mat_without_C.shape[1]))

    A_mat_H, A_mat_W = A_mat_without_C.shape
    for c_i in range(self.C):
      self.A_mat[c_i * A_mat_H:(c_i + 1) * A_mat_H, c_i * A_mat_W:(c_i + 1) * A_mat_W] = A_mat_without_C


if __name__ == '__main__':
  img = np.array(cv2_imread('img_examples/pug.png'), dtype='float32')
  img = cv2_resize(img, (64,64)) / 255.0

  inverse_problem = PinholeProblem(img=img, obs_size=img.shape[1:], gaussian_noise_std=0.05)
  obs = inverse_problem.forward()
  imshow(img, title='img')
  imshow(obs, title='obs')

  recovered_img = inverse_problem.explicit_solve()
  imshow(recovered_img, title='recovered_img')