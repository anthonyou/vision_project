import os
# Manel: Hardcoded visdom environment. I have this on my path, but we can use the same.
# you need to launch the following command, if it's not already running:
# results are displayed in http://visiongpu09.csail.mit.edu:12890/, by selecting the corresponding visdom_environment
os.environ['VISDOM_HOST'] = 'visiongpu38'
visdom_environment = 'inverse_vision_' + os.environ['USER']

from my_python_utils.common_utils import *
from problems import problems
from solvers import solvers

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Main method')
  parser.add_argument("--img-file", type=str, default='img_examples/pug.png', help="img path to test")
  parser.add_argument('--img-size', type=int, default=512)
  parser.add_argument('--problem-type', type=str, default='random_projection', choices=problems.keys())
  parser.add_argument('--solver-method', type=str, default='img2img', choices=solvers.keys())
  parser.add_argument('--seed', default=1337, type=int)

  # params for img2img solver, ignored if other method is used
  parser.add_argument("--steps", type=int, default=250, help="number of ddim sampling steps")

  args = parser.parse_args()

  img = np.array(cv2_imread(args.img_file) / 255.0, dtype='float32')
  img = best_centercrop_image(img, args.img_size, args.img_size)

  problem = problems[args.problem_type](img=img, obs_size=img.shape[1:])
  obs = problem.forward()
  solver = solvers[args.solver_method](problem, verbose=True)

  reconstruction, logs = solver.solve()

  imshow(img, title='image', env=visdom_environment)
  imshow(obs, title='observation', env=visdom_environment)
  imshow(reconstruction, title='reconstruction', env=visdom_environment)

  if args.solver_method == 'img2img':
    # store intermediate iterations in logs, and plot them
    imshow(tile_images(logs['sgd_per_iter']), title='sgd_iterations', env=visdom_environment)
    imshow(tile_images(logs['img2img_per_iter']), title='img2img_iterations', env=visdom_environment)
