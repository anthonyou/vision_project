root_dir = '.'
unconditioned = False

if unconditioned:
    config = '/data/vision/torralba/scratch/aou/vision_project/stable_diffusion/synthetic_model_config.yaml'
    ckpt = '/data/vision/torralba/movies_sfm/home/no_training_cnn/general_vision_prior/latent_diffusion/logs/2023-01-17T17-29-04_s21k_mixup-ldm-kl-32x32x4/checkpoints/last.ckpt'
    precision = 'nullcontext'
else:
    config = '/data/vision/torralba/scratch/aou/vision_project/stable_diffusion/v1-inference.yaml'
    ckpt = '/data/vision/torralba/scratch/aou/vision_project/stable_diffusion/model.ckpt'
    precision = 'autocast'

img2img_config = {
    'unconditioned': unconditioned,
    'device': 0,
    'prompt': '',
    'iterations': 11,
    'loss_cutoff': 50000,
    'learning_rate': 0.00005,
    'ddim_steps': 50,
    'ddim_eta': 0.0,
    'n_iter': 1,
    'n_samples': 1,
    'scale': 5.0,
    'strength': 0.5,
    'decay_rate': 0.8,
    'min_strength': 0.01,
    'config': config,
    'ckpt': ckpt,
    'seed': 42,
    'precision': precision,
    'log_frequency': 10
}
