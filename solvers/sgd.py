import torch
import numpy as np
from numpy.random import default_rng
from scipy import sparse

from solvers.base_solver import BaseSolver

class StochasticGradDescSolver(BaseSolver):
    """
    Solver using gradient descent to bring image closer to observation
    """
    
    def __init__(self, problem, config, verbose=False):
        super().__init__(problem, config, verbose)
        if self.problem.tensor_type != 'torch':
            self.problem.init_torch()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.loss_cutoff = config['loss_cutoff']
        self.lr = config['learning_rate']
        self.max_iters = config['max_sgd_iters']

    def solve(self, img=None, obs=None):
        """
        img and obs are torch tensors and A is a torch sparse matrix
        """
        if img is None:
            assert self.problem.img is not None
            img = self.problem.img
        if obs is None:
            assert self.problem.obs is not None
            obs = self.problem.obs
        recovered_img = img.detach() # Need this line or else img will be overwritten
        recovered_img = recovered_img.to(self.device).requires_grad_(True)
        obs = obs.to(self.device).requires_grad_(False)

        loss = float('inf')
        i = 0
        optim = torch.optim.Adam(lr=self.lr, params=[recovered_img])
        while loss > self.loss_cutoff and i < self.max_iters:
            recovered_obs = self.problem.forward(recovered_img)
            loss = torch.mean((recovered_obs - obs)**2)
            if i % 10 == 0 and self.verbose:
                print('SGD loss it {} = {}'.format(i, loss))
            optim.zero_grad()
            loss.backward()
            optim.step()

            i += 1

        return recovered_img


if __name__ == "__main__":
    home_dir = '/data/vision/torralba/scratch/aou/vision_project'
    obs_filename = f'{home_dir}/classical_images/obs_with_extra_noise.npy'
    recovered_img = load_img(f'{home_dir}/med_dog.jpeg')[0]
    print(recovered_img.shape)
 
    noisy_obs = torch.from_numpy(np.load(obs_filename))

    N = noisy_obs.flatten().shape[0]
    rng = default_rng(123)
    A = sparse.random(N, N, density=0.0001, format='coo', dtype=None, random_state=rng, data_rvs=None)
    values, indices = torch.DoubleTensor(A.data), torch.LongTensor(np.vstack((A.row, A.col)))
    A = torch.sparse.DoubleTensor(indices, values, torch.Size(A.shape))

    solver = StochasticGradDescSolver(None)
    result = solver.solve(recovered_img, noisy_obs, A)
