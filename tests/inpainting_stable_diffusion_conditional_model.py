from general_vision_prior.analysis.ldm.utils import *
from general_vision_prior.analysis.analysis_utils import get_imgs_to_analyze

from diffusers import DiffusionPipeline
import torch
seed = 1337
torch.manual_seed(seed)
np.random.seed(seed)
img_paths = get_imgs_to_analyze()
device = 'cuda:3'

all_images = []
N_predictions = 10
for img_path in tqdm(img_paths):
  batch = make_batch(img_path, 'mae', device=device, img_size=512, mask_ratio=0.75)
  # inp_img = Image.fromarray(np.array(tonumpy(batch['masked_image'][0] + 1 ) * 255 / 2, dtype='uint8').transpose((1,2,0)))  # loaded with PIL.Image
  full_inp_img = Image.fromarray(np.array(tonumpy(batch['image'][0] + 1 ) * 255 / 2, dtype='uint8').transpose((1,2,0)))  # loaded with PIL.Image

  # mask = Image.fromarray(((1 - np.array([tonumpy(batch['mask'])]*3, dtype='uint8'))* 255).transpose((1,2,0)))
  mask_np = tonumpy(batch['mask']) # np.random.uniform(size=batch['mask'].shape) < 0.5

  inp_img = Image.fromarray(np.array((tonumpy(batch['image'][0]) + 1) * mask_np[None, :,:] * 255 / 2, dtype='uint8').transpose((1,2,0)))  # loaded with PIL.Image
  mask = Image.fromarray(((1 - np.array([mask_np]*3, dtype='uint8')) * 255).transpose((1,2,0)))

  inner_image = inp_img.convert("RGBA")
  pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    custom_pipeline="img2img_inpainting",
    torch_dtype=torch.float32
  )
  pipe = pipe.to(device)
  def dummy(images, **kwargs):
    return images, False
  pipe.safety_checker = dummy

  #pipe.enable_attention_slicing()  # to save some gpu memory in exchange for a small speed decrease
  prompt = "nothing to see here..."
  negative_prompt = None

  images = []
  print("Processing {} samples".format(N_predictions))
  for _ in range(N_predictions):
    result = pipe(prompt=prompt, image=inp_img,
                  inner_image=inner_image,
                  mask_image=mask, negative_prompt=None,
                  num_inference_steps = 50, guidance_scale = 0)

    images.append(np.array(result.images[0]).transpose((2,0,1)))
  image = np.array(images).mean(0)
  var_image = np.array(images).mean(-1).var(0)
  # imshow(np.array(mask), title='mask')
  # imshow(np.array(inp_img), title='original_image_masked')
  # imshow(batch['image'], title='original_image')
  # imshow(np.array(result.images[0]), title='recons')
  tiled_image = tile_images([(tonumpy(batch['image'][0]) + 1) * 255.0/2,
                              np.array(inp_img).transpose((2,0,1)),
                              np.array(result.images[0]).transpose((2,0,1))],
                              tiles_x_y=(3,1), border_pixels=10)

  imshow(tiled_image)
  all_images.append(tiled_image)

imshow(tile_images(all_images[:10], tiles_x_y=(2,5)))