from base_solver import BaseSolver


class ExplicitSolver(BaseSolver):
  def solve(self):
    return self.problem.explicit_solve()