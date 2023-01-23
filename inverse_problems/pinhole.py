from base_problem import BaseProblem
import numpy as np

class RandomProjectionWithGaussianNoise(BaseProblem):
    def __init__(self, obs=None, img=None):
        super().__init__(obs=obs, img=img)

        # assumes obs and img are the same dimensionality.

        # TODO: change to rasterization order
        N = img.flatten().shape[0]
        rng = default_rng(123)
        A = sparse.random(N, N, density=0.0001, format='coo', dtype=None, random_state=rng, data_rvs=None)
        self.A_mat = np.eye(obs.shape)

    def forward_process(self):
        assert not self.img is None
        return self.A_mat @ self.img.reshape(-1)

    def explicit_solve(self):
        assert not self.obs is None
        inv_mat = np.linalg.inv(self.A_mat)

        return inv_mat @ self.obs.reshape(-1)

    def implicit_solve(self, lr=0.0005, T=10000):
        img = torch.from_numpy(self.img).cuda().requires_grad_(True)
        obs = torch.from_numpy(self.obs).cuda().requires_grad_(False)

        values, indices = torch.DoubleTensor(self.A_mat.data), torch.LongTensor(np.vstack((A.row, A.col)))
        A_mat = torch.sparse.DoubleTensor(indices, values, torch.Size(self.A_mat.shape))
        A_mat = A_mat.cuda().requires_grad_(False)
        for i in range(T):
            recovered_obs = torch.sparse.mm(A_mat, recovered_img.flatten().unsqueeze(0).T.double())
            recovered_obs = recovered_obs.view(obs.shape)
            loss = torch.mean((recovered_obs - obs)**2)
            if i % 100 == 0:
                print(loss)
            loss.backward(retain_graph=True)
            recovered_img.data = recovered_img.data - lr * recovered_img.grad.data
            recovered_img.grad.zero_()
        return recovered_img.detach().cpu().numpy()
