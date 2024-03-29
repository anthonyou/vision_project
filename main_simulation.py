import os, sys
sys.path.append('./dependencies/stable-diffusion')

import shutil
import json

  # Manel: Hardcoded visdom environment. I have this on my path, but we can use the same.
  # you need to launch the following command, if it's not already running:
  # results are displayed in http://visiongpu09.csail.mit.edu:12890/, by selecting the corresponding visdom_environment
  
os.environ['VISDOM_HOST'] = 'visiongpu03'

from my_python_utils.common_utils import *
from matplotlib.pyplot import imsave
from problems import problems
from solvers import solvers
from solvers.config import configs

assert os.path.abspath(os.getcwd()).endswith('/vision_project'), "Should run from vision_project folder, all paths are encoded relatively to this folder (instead of absolute paths)"

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Main method')
  parser.add_argument("--device", type=int, default=0)
  parser.add_argument("--img-file", type=str, default='img_examples/dog.jpeg', help="img path to test")
  parser.add_argument('--img-size', type=int, default=512)
  parser.add_argument('--problem-type', type=str, default='inpainting', choices=problems.keys())
  parser.add_argument('--solver-method', type=str, default='img2img', choices=solvers.keys())
  parser.add_argument('--seed', default=42, type=int)
  parser.add_argument('--iterations', default=11, type=int, help='how many iterations to run')
  parser.add_argument('--loss-cutoff', default=0.0005, type=int, help='sgd l2 loss cutoff (per color pixel)')
  parser.add_argument('--learning-rate', default=0.05, type=float, help='sgd leaarning rate')

  # params for img2img solver, ignored if other method is used
  parser.add_argument("--ddim-steps", type=int, default=50, help="number of ddim sampling steps")
  parser.add_argument('--strength', default=0.9, type=float)
  parser.add_argument('--decay-rate', default=0.8, type=float)
  parser.add_argument("--save-path", type=str, default='/data/vision/torralba/scratch/aou/vision_project/simulation_experiments', help="where to save results, leave empty to not save")

  args = parser.parse_args()
  config = configs[args.solver_method]
  
  config['device'] = args.device
  config['img_file'] = args.img_file
  config['img_size'] = args.img_size
  config['problem_type'] = args.problem_type
  config['solver_method'] = args.solver_method
  config['seed'] = args.seed
  config['ddim_steps'] = args.ddim_steps
  config['iterations'] = args.iterations
  config['loss_cutoff'] = args.loss_cutoff
  config['learning_rate'] = args.learning_rate
  config['strength'] = args.strength
  config['decay_rate'] = args.decay_rate
  config['max_sgd_iters'] = 1000
  config['prompt'] = 'photo of a puppy on a white background'

  img = np.array(cv2_imread(args.img_file) / 255.0, dtype='float32')
  img = best_centercrop_image(img, args.img_size, args.img_size)


  with torch.cuda.device('cuda:{}'.format(args.device)):
    problem = problems[args.problem_type](img=img, obs_size=img.shape[1:])
    solver = solvers[args.solver_method](problem, config, verbose=True)
    reconstruction, logs, obs = solver.solve()
    reconstruction, obs = reconstruction.squeeze(), obs.squeeze()

  visdom_environment = 'inverse_vision_' + os.environ['USER'] + '_' + str(config['device'])
  
  imshow(img, title='image', env=visdom_environment)
  imshow(obs, title='observation', env=visdom_environment)
  imshow(reconstruction, title='reconstruction', env=visdom_environment)

  if args.solver_method == 'img2img':
    # store intermediate iterations in logs, and plot them
    sgd_summary = tile_images([tonumpy(k)[0] for k in logs['sgd_per_iter']], border_pixels_x_y=(5, 5))
    img2img_summary = tile_images([tonumpy(k)[0] for k in logs['img2img_per_iter']], border_pixels_x_y=(5, 5))
    imshow(sgd_summary, title='sgd_iterations', env=visdom_environment)
    imshow(img2img_summary, title='img2img_iterations', env=visdom_environment)

  if os.path.exists(args.save_path):
    exp_num = len(os.listdir(args.save_path))
    save_path = os.path.join(args.save_path, f'experiment_{exp_num}')
    os.mkdir(save_path)
    shutil.copytree(os.path.join(os.getcwd(), 'solvers'), os.path.join(save_path, 'solvers'))
    shutil.copytree(os.path.join(os.getcwd(), 'problems'), os.path.join(save_path, 'problems'))
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
      json.dump(config, f)
    for img_content, img_name in zip([img, obs, reconstruction, img2img_summary, sgd_summary], ['img', 'obs', 'reconstruction', 'img2img_summary', 'sgd_summary']):
      img_content = tonumpy(img_content)
      img_content = img_content - np.min(img_content)
      img_content = img_content / np.max(img_content)
      imsave(os.path.join(save_path, f'{img_name}.jpeg'), img_content.transpose([1, 2, 0]))
