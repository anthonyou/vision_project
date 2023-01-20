import os
# Manel: Hardcoded visdom environment. I have this on my path, but we can use the same.
# you need to launch the following command, if it's not already running:
os.environ['VISDOM_HOST'] = 'visiongpu09:12890'

from my_python_utils.common_utils import *

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Main method')
  parser.add_argument("--steps", type=int, default=250, help="number of ddim sampling steps")
  parser.add_argument("--img", type=str, default='img_examples/pug.png', help="img path to test")
  parser.add_argument('--method', type=str, default='explicit', choices=['explicit', 'img2img', 'masking'])
  parser.add_argument('--seed', default=1337, type=int)

