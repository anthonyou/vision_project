import torch

from problems.base_problems import *

def construct_mask_mae(masking_ratio, img_size):
  mask = np.random.uniform(0, 1, size=(14,14))
  mask = mask < masking_ratio

  mask = cv2_resize(mask.astype('float32'), img_size, interpolation=cv2.INTER_NEAREST)

  return mask

def construct_mask_center(masking_ratio, img_size):
  mask_width = np.sqrt(masking_ratio)

  mask = np.ones(img_size)
  mask[int(img_size[0] * (1-mask_width)/2):int(img_size[0] * (1+mask_width)/2)] = 0

  return mask

class InpaintingProblem(BaseProblem):
  # simply remove pixels at random
  def __init__(self, masking_ratio=0.5, mask_type='mae', obs_size=None, *args, **kwargs):
    super().__init__(*args, **kwargs)

    available_mask_types = ['mae', 'center']

    self.masking_ratio = masking_ratio
    assert mask_type in available_mask_types, "mask_type should be one of {}".format(available_mask_types)
    self.mask_type = mask_type
    if self.mask_type == 'mae':
      self.mask = construct_mask_mae(masking_ratio, self.img_size)
    elif self.mask_type == 'center':
      self.mask = construct_mask_center(masking_ratio, self.img_size)

  def forward(self, img):
    if type(img) is torch.Tensor:
      mask = torch.FloatTensor(self.mask).to(img.device)
    else:
      mask = tonumpy(self.mask)

    return img * mask[None,...]

  def get_obs(self, img):
    return self.forward(img=img)

if __name__ == '__main__':
  img = np.array(cv2_imread('img_examples/pug.png'), dtype='float32')
  img = cv2_resize(img, (512,512)) / 255.0

  inverse_problem = InpaintingProblem(img=img, masking_ratio=0.5, mask_type='mae')
  obs = inverse_problem.forward(img=img)
  imshow(img, title='img')
  imshow(obs, title='obs')