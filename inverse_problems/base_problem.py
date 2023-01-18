
class BaseProblem():
  # class to be inherited by other problems
  def __init__(self, img=None, obs=None):
    assert (img is None) + (obs is None) == 1, "Either img (for simulation) or obs (for real data) should be provided!"
    self.img = img
    self.obs = obs

  def forward_process(self):
    raise Exception("Must be implemented by children class")

  def explicit_solve(self):
    raise Exception("Must be implemented by children class")
