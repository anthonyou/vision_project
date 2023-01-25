#from my_python_utils.common_utils import *

class BaseSolver():
  def __init__(self, problem, verbose=False):
    self.problem = problem
    self.verbose = verbose
