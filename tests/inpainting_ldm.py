from my_python_utils.common_utils import *

conda_name = sys.executable.split('/')[-3]
assert conda_name == 'ldm'

from general_vision_prior.analysis.analysis_utils import *
from general_vision_prior.analysis.ldm.utils import *

from general_vision_prior.latent_diffusion.scripts.sample_diffusion import *

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--steps",
    type=int,
    default=500,
    help="number of ddim sampling steps",
  )
  parser.add_argument(
    "--batch-size",
    type=int,
    default=5,
    help="number of ddim sampling steps",
  )
  parser.add_argument(
    "--debug",
    type=str2bool,
    default="True",
    help="Whether to run in debug mode or not",
  )
  parser.add_argument(
    "--gpu",
    type=int,
    default=0,
    help="GPU to use",
  )
  parser.add_argument(
    "--model-to-test",
    type=str,
    default="ours_s21k",
    choices=['downloaded_lsun', 'ours_s21k', 'stablediffusion'],
    help="What model to test",
  )
  parser.add_argument(
    "--log-every-t",
    type=int,
    default=1,
    help="Log every t steps",
  )
  parser.add_argument(
    "--seed",
    type=int,
    default=1337,
    help="Random seed to use",
  )

  opt = parser.parse_args()
  random.seed(opt.seed)
  torch.manual_seed(opt.seed)

  mask_type = 'mae' #'center' #

  img_paths = get_imgs_to_analyze(imagenet_only=False)
  # img_paths = get_imgs_to_analyze_shaders_mixup()

  is_sd = opt.model_to_test == 'stablediffusion'

  if opt.debug:
    model, vae = get_model(opt.model_to_test, return_vae=True)
  else:
    model = get_model(opt.model_to_test, return_vae=False)

  device = torch.device("cuda:{}".format(opt.gpu)) if torch.cuda.is_available() else torch.device("cpu")
  model = model.to(device)
  if not is_sd:
    model.eval()

  if opt.debug:
    vae = vae.to(device)
    vae.eval()

  all_images = []
  mask = None

  if opt.debug:
    batch_size = opt.batch_size
    custom_steps = 50
    vanilla_sampling = False
    eta = 1.0
    if is_sd:
      prompt = ["running without guidance guidance_scale=0"] * batch_size
      image = [np.array(im).transpose((2,0,1)) for im in model(prompt, guidance_scale=0.0).images]
    else:
      if opt.model_to_test == 'downloaded_imagenet':
        xc = torch.tensor(batch_size*[2])
        c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
        sampler = DDIMSampler(model)

        uc = model.get_learned_conditioning(
          {model.cond_stage_key: torch.tensor(batch_size*[1000]).to(model.device)}
        )
        samples_ddim, _ = sampler.sample(S=50,
                                         conditioning=c,
                                         batch_size=batch_size,
                                         shape=[3, 64, 64],
                                         verbose=False,
                                         unconditional_guidance_scale=10,
                                         unconditional_conditioning=uc,
                                         eta=1.0)

        x_samples_ddim = model.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0,
                                     min=0.0, max=1.0)
        image = x_samples_ddim

      else:
        cond = None
        logs = make_convolutional_sample(model, batch_size=batch_size,
                                         vanilla=vanilla_sampling, custom_steps=custom_steps,
                                         eta=eta, log_every_t=1) #, cond=cond
        image = logs['sample']
    #for k, v in enumerate(logs['intermediates']['pred_x0']):
    #  visdom_histogram(v.clip(-3,3), title='convsample_step_' + str(k))
    #visdom_histogram(np.random.normal(size=10000).clip(-3,3), title='gaussian')

    images = [(im - im.min()) / (im.max() - im.min()) for im in image]
    imshow(tile_images(images), 'Samples {}'.format(opt.model_to_test), biggest_dim=1024)

  with torch.no_grad():
    for img_path in tqdm(img_paths):
      # prepare data
      batch = make_batch(img_path, mask_type, device=device, img_size=512 if is_sd else 256,
                         latent_down_ratio=4 if opt.model_to_test == 'downloaded_imagenet' else 8)

      mask = batch['latent_mask'][None,None]

      image = batch['image']
      full_image_mask = batch['mask']
      masked_image = batch['masked_image']

      # encode masked image and concat downsampled mask
      # replicates the inpainting that is produced during training when logging to test the model being trained.
      # This is not conditional inpainting.

      if is_sd:
        # imshow(model.vae.decode( model.vae.encode(image).latent_dist.sample())[0])
        x0 = model.vae.encode(image).latent_dist.sample()
      else:
        x0 = model.encode_first_stage(image)
        if not type(x0) is torch.Tensor:
          x0 = x0.mean

      # Following same logic as ldm.models.diffusion.ddpm, L1326:L1344
      # make unit variance
      x0 = x0 / x0.std()

      bs = opt.batch_size
      shape = x0.shape[1:]
      # samples, intermediates = ddim.sample(timesteps, batch_size=bs, shape=shape, eta=eta, verbose=False, log_every_t=1)
      ''' replace the previous line'''
      # default args

      ddim_use_original_steps=False;
      # mask=None; x0=None;


      # end of args

      if is_sd:
        scheduler = model.scheduler
      else:
        scheduler = DDIMSampler(model)
        scheduler.make_schedule(ddim_num_steps=opt.steps, ddim_eta=eta, verbose=True)
      # sampling
      C, H, W = shape
      size = (batch_size, C, H, W)
      print(f'Data shape for DDIM sampling is {size}, eta {eta}')

      device = model.device
      b = size[0]
      img = torch.randn(size, device=device)

      if not ddim_use_original_steps:
        if is_sd:
          scheduler_timesteps = tonumpy(scheduler.timesteps)
        else:
          scheduler_timesteps = scheduler.ddim_timesteps
        subset_end = int(min(opt.steps / scheduler_timesteps.shape[0], 1) * scheduler_timesteps.shape[0]) - 1
        timesteps = scheduler_timesteps[:subset_end]

      intermediates = {'x_inter': [img], 'pred_x0': [img]}
      time_range = reversed(range(0,opt.steps)) if ddim_use_original_steps else np.flip(timesteps)
      total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
      print(f"Running DDIM Sampling with {total_steps} timesteps")

      iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

      if opt.model_to_test == 'downloaded_imagenet':
        xc = torch.tensor(batch_size*[int(batch['imagenet_id'])])
        c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
      else:
        c = None

      all_samples = []
      for i, step in enumerate(iterator):
        index = total_steps - i - 1
        ts = torch.full((b,), step, device=device, dtype=torch.long)

        if mask is not None:
          assert x0 is not None
          if is_sd:
            unscaled_noise = torch.randn(size, device=device)
            img_orig = scheduler.add_noise(x0, unscaled_noise, ts)
          else:
            img_orig = scheduler.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
          img = img_orig * mask + (1. - mask) * img

        if is_sd:
          outs = scheduler.step(img, c, ts,
                                         use_original_steps=ddim_use_original_steps,
                                         quantize_denoised=False, temperature=1.0)
        else:
          outs = scheduler.p_sample_ddim(img, c, ts, index=index, use_original_steps=ddim_use_original_steps,
                                       quantize_denoised=False, temperature=1.0)
        img, pred_x0 = outs

        if index % opt.log_every_t == 0 or index == total_steps - 1:
          intermediates['x_inter'].append(img)
          intermediates['pred_x0'].append(pred_x0)

      if mask is not None:
        assert x0 is not None
        img = x0 * mask + (1. - mask) * img

      samples = img

      ''' end of replacement'''

      if not mask is None:
        samples = x0 * mask + (1. - mask) * samples

      imshow(tile_images([k[0,:3] for k in intermediates['x_inter']]))

      final_img = (model.decode_first_stage(samples) / 2 + 0.5).clamp(0, 1)
      # final_img = (model.first_stage_model.decode(samples) / 2 + 0.5).clamp(0, 1)
      imshow(tile_images(final_img), title='diff_recons')
      imshow(final_img[0], title='single_image_recons')
      imshow(masked_image, title='masked_image')
      imshow(final_img.mean(0) * (1 - full_image_mask) + (masked_image + 1 ) * 0.5, title='mean_recons_masked')



# Code for conditional inpainting
  '''

  

  '''



'''
Encode decode


  x0 = model.first_stage_model.encode(image).mean
  if opt.debug:
    visdom_histogram(x0, title='hist_from_model')
    encoded_decoded_image = (model.first_stage_model.decode(x0)[0] / 2 + 0.5).clamp(0, 1)
    imshow(encoded_decoded_image, title='from_model')

    x0_vae = vae.encode(image).mean
    visdom_histogram(x0_vae, title='hist_from_vae')
    encoded_decoded_image = (vae.decode(x0_vae) / 2 + 0.5).clamp(0, 1)
    imshow(encoded_decoded_image, title='from_vae')
    
'''