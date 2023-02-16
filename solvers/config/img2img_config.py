home_dir = '/data/vision/torralba/scratch/aou/vision_project'
img2img_config = {
    'unconditioned': True,
    'device': 0,
    'prompt': '',
    'iterations': 11,
    'loss_cutoff': 50000,
    'learning_rate': 0.00005,
    'ddim_steps': 50,
    'ddim_eta': 0.0,
    'n_iter': 1,
    'n_samples': 1,
    'scale': 1.0, # 5.0
    'strength': 0.5,
    'decay_rate': 0.9,
    'min_strength': 0.01,
    'config': f'{home_dir}/stable_diffusion/synthetic_model_config.yaml', #v1-inference.yaml',
    'ckpt': '/data/vision/torralba/movies_sfm/home/no_training_cnn/general_vision_prior/latent_diffusion/logs/2023-01-17T17-29-04_s21k_mixup-ldm-kl-32x32x4/checkpoints/last.ckpt', #f'{home_dir}/stable_diffusion/synthetic_model.ckpt',
    'seed': 42,
    'precision': 'autocast'
}
