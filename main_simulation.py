import os
# Manel: Hardcoded visdom environment. I have this on my path, but we can use the same.
# you need to launch the following command, if it's not already running:
os.environ['VISDOM_HOST'] = 'visiongpu09:12890'

from my_python_utils.common_utils import *
from inverse_problems import *
from solvers import *

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Main method')
  parser.add_argument("--img-file", type=str, default='img_examples/pug.png', help="img path to test")
  parser.add_argument('--problem-type', type=str, default='random', choices=inverse_problems.keys())
  parser.add_argument('--solver-method', type=str, default='explicit', choices=['explicit', 'img2img', 'masking'])
  parser.add_argument('--seed', default=1337, type=int)

  # params for img2img solver, ignored if other method is used
  parser.add_argument("--steps", type=int, default=250, help="number of ddim sampling steps")

  args = parser.parse_args()

  img = np.array(cv2_imread(args.img_file) / 255.0, dtype='float32')
  problem = inverse_problems[args.problem_type](img=img)
  obs = problem.forward()
  solver = solvers[args.solver_method](problem, verbose=True)

  reconstruction, logs = solver.solve()

  imshow(img, title='gt')
  imshow(img, title='observation')
  imshow(reconstruction, title='reconstruction')

