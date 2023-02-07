root_dir = '.'
img2img_config = {
    'device': 0,
    'prompt': 'a clear photograph of a sitting pug puppy',
    'iterations': 101,
    'loss_cutoff': 5000,
    'learning_rate': 0.005,
    'ddim_steps': 50,
    'ddim_eta': 0.0,
    'n_iter': 1,
    'n_samples': 1,
    'scale': 5.0,
    'strength': 0.5,
    'decay_rate': 0.99,
    'min_strength': 0.01,
    'config': f'{root_dir}/stable_diffusion/v1-inference.yaml',
    'ckpt': f'{root_dir}/stable_diffusion/model.ckpt',
    'seed': 42,
    'precision': 'autocast',
    'log_frequency': 10
}
