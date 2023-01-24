from solvers.base_solver import BaseSolver

class ExplicitSolver(BaseSolver):
  def solve(self):
    logs = dict()
    return self.problem.explicit_solve(), logs