import torch
import numpy as np
from numpy.random import default_rng
from scipy import sparse

from solvers.base_solver import BaseSolver

class StochasticGradDescSolver(BaseSolver):
    """
    Solver using gradient descent to bring image closer to observation
    """
    
    def __init__(self, problem, verbose=False):
        super().__init__(problem, verbose)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.loss_cutoff = 150000
        self.lr = 0.0005

        A = problem.A_mat
        values, indices = torch.DoubleTensor(A.data), torch.LongTensor(np.vstack((A.row, A.col)))
        self.A = torch.sparse.DoubleTensor(indices, values, torch.Size(A.shape)).float()


    def solve(self, img, obs):
        """
        img and obs are torch tensors and A is a torch sparse matrix
        """
        recovered_img = img.detach()
        recovered_img = img.to(self.device).requires_grad_(True)
        obs = obs.to(self.device).requires_grad_(False)
        A = self.A.to(self.device).requires_grad_(False)

        loss = float('inf')
        i = -1
        while loss > self.loss_cutoff:
            i += 1
            recovered_obs = torch.sparse.mm(A, recovered_img.flatten().unsqueeze(0).T)
            recovered_obs = recovered_obs.view(obs.shape)
            loss = torch.sum((recovered_obs - obs)**2)
            if i % 10 == 0:
                print(loss)
            loss.backward(retain_graph=True)
            recovered_img.data = recovered_img.data - self.lr * recovered_img.grad.data
            recovered_img.grad.zero_()
        return recovered_img


if __name__ == "__main__":
    home_dir = '/data/vision/torralba/scratch/aou/vision_project'
    obs_filename = f'{home_dir}/classical_images/obs_with_extra_noise.npy'
    # recovered_img = load_img(f'{home_dir}/med_dog.jpeg')[0]
    print(recovered_img.shape)
 
    noisy_obs = torch.from_numpy(np.load(obs_filename))

    N = noisy_obs.flatten().shape[0]
    rng = default_rng(123)
    A = sparse.random(N, N, density=0.0001, format='coo', dtype=None, random_state=rng, data_rvs=None)
    values, indices = torch.DoubleTensor(A.data), torch.LongTensor(np.vstack((A.row, A.col)))
    A = torch.sparse.DoubleTensor(indices, values, torch.Size(A.shape))

    solver = StochasticGradDescSolver(None)
    result = solver.solve(recovered_img, noisy_obs, A)
