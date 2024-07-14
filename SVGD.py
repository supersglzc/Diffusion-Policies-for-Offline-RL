# math
import numpy as np
import numpy.matlib as nm
from numpy import linalg
from scipy.stats import multivariate_normal, gaussian_kde
from scipy.spatial.distance import pdist, squareform
import torch
from utils import *
# from model_utils import *
import matplotlib.pyplot as plt
import wandb


class SVGD_algo():
    """class that will perform svgd via self.update"""
    def __init__(self):
        pass

    def svgd_kernel(self, theta, h=-1):
        """Gaussian RBF kernel function"""
        theta = theta.cuda()
        sq_dist = torch.cdist(theta, theta, p=2)
        pairwise_dists = sq_dist ** 2

        if h < 0:  # if h < 0, using median trick
            h = torch.median(pairwise_dists)
            h = torch.sqrt(0.5 * h / torch.log(torch.tensor(theta.shape[0] + 1.0, device=theta.device)))

        # Compute the RBF kernel
        Kxy = torch.exp(-pairwise_dists / (h ** 2) / 2)
        dxkxy = -torch.matmul(Kxy, theta)
        sumkxy = torch.sum(Kxy, axis=1)

        for i in range(theta.shape[1]):
            dxkxy[:, i] += theta[:, i] * sumkxy

        dxkxy = dxkxy / (h ** 2)
        return Kxy, dxkxy

    def update(self, state, particles, lnprob, num_points, n_iter=1000, stepsize=1e-3, h=0.01, alpha=0.9, debug=False, if_visualize=False, device='cuda'):
        """performs svgd
        Args:
            x0: np array - the initial set of particles
            lnprob: function - logarithmic gradient of the target density
            n_iter: int - number of iterations to run
            stepsize: float
            alpha: float - correction factor
            debug: boolean - verbose console output
        Returns:
            theta: np array - particles after svgd
        """

        fudge_factor = 1e-6
        historical_grad = 0

        for iter in range(n_iter):
            if debug and (iter + 1) % 1000 == 0:
                print('iter ' + str(iter + 1))
            
            action_optimizer = torch.optim.Adam([particles], lr=0.01, eps=1e-5)
            particles.requires_grad_(True)
            Q, _ = lnprob(state, particles)
            loss = -Q.mean()
            action_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            lnpgrad = -particles.grad
            particles.requires_grad_(False)

            kxy, dxkxy = self.svgd_kernel(particles, h=h)
            grad_theta = (torch.matmul(kxy, lnpgrad) + dxkxy) / particles.shape[0]

            if iter == 0:
                historical_grad = historical_grad + grad_theta ** 2
            else:
                historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta ** 2)

            adj_grad = grad_theta / (fudge_factor + torch.sqrt(historical_grad))
            particles = particles + stepsize * adj_grad
        
        if if_visualize:
            fig, ax = plt.subplots()
            ax.scatter(particles[:, 0].cpu().numpy(), particles[:, 1].cpu().numpy(), alpha=0.3)
            ax.set_xlim((-1, 1))
            ax.set_ylim((-1, 1))
            fig.canvas.draw()  # Draw the canvas, cache the renderer
            image_flat = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
            img = image_flat.reshape(*reversed(fig.canvas.get_width_height()), 3)
            plt.close()
            return particles, img
        else:
            return particles
    
    def only_gradient(self, x0, lnprob, n_iter = 1000, stepsize = 1e-3, bandwidth = -1, alpha = 0.9, debug = False):
        """performs only the gradiational descend part of svgd"""
        # Check input
        if x0 is None or lnprob is None:
            raise ValueError('x0 or lnprob cannot be None!')
        theta = np.copy(x0) 
        # adagrad with momentum
        fudge_factor = 1e-6
        historical_grad = 0
        # new add
        theta = torch.tensor(theta, dtype=torch.float32).reshape(num_points*num_points, 2)
        for iter in range(n_iter):
            if debug and (iter+1) % 1000 == 0:
                print ('iter ' + str(iter+1))  

            theta = torch.tensor(theta, dtype=torch.float32)
            action_optimizer = torch.optim.Adam([theta], lr=0.01, eps=1e-5)
            theta.requires_grad_(True)
            Q = lnprob(theta)
            loss = -Q.mean()
            action_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            lnpgrad = -theta.grad
            theta.requires_grad_(False)
            # lnpgrad = lnprob(theta)
            # calculating the kernel matrix
            kxy, dxkxy = self.svgd_kernel(theta, h = 0.0005)  
            grad_theta = (np.matmul(kxy, lnpgrad)) / x0.shape[0]  
            # adagrad 
            if iter == 0:
                historical_grad = historical_grad + grad_theta ** 2
            else:
                historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta ** 2)
            adj_grad = np.divide(grad_theta, fudge_factor+np.sqrt(historical_grad))
            theta = theta + stepsize * adj_grad

            if (iter+1) % 100 == 0:
                fig, ax = plt.subplots()
                ax.scatter(theta[:, 0].cpu().numpy(), theta[:, 1].cpu().numpy())
                ax.set_xlim((-1, 1))
                ax.set_ylim((-1, 1))
                fig.canvas.draw()  # Draw the canvas, cache the renderer
                image_flat = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
                img = image_flat.reshape(*reversed(fig.canvas.get_width_height()), 3)
                img = wandb.Image(img)
                wandb.log({'images/actions': img}, step=iter)
                plt.close()
        return theta
    
    def only_kernel(self, x0, lnprob, n_iter = 1000, stepsize = 1e-3, bandwidth = -1, alpha = 0.9, debug = False):
        """performs only the kernel part of svgd, the repulsive force"""
        # Check input
        if x0 is None or lnprob is None:
            raise ValueError('x0 or lnprob cannot be None!')
        theta = np.copy(x0) 
        # adagrad with momentum
        fudge_factor = 1e-6
        historical_grad = 0
        for iter in range(n_iter):
            if debug and (iter+1) % 1000 == 0:
                print ('iter ' + str(iter+1))  
            lnpgrad = lnprob(theta)
            # calculating the kernel matrix
            kxy, dxkxy = self.svgd_kernel(theta, h = -1)  
            grad_theta = (dxkxy) / x0.shape[0]  
            # adagrad 
            if iter == 0:
                historical_grad = historical_grad + grad_theta ** 2
            else:
                historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta ** 2)
            adj_grad = np.divide(grad_theta, fudge_factor+np.sqrt(historical_grad))
            theta = theta + stepsize * adj_grad 
        return theta
    
# wandb_run = wandb.init(project='SVGD', mode='online', name=f'test')
# Qnet = MLPNet(in_dim=2, out_dim=1, hidden_layers=[64, 32]).to('cuda')
# weight = torch.load('/home/supersglzc/code/Diffusion-Policies-for-Offline-RL/Qnet-z.pth')
# Qnet.load_state_dict(weight)

# n_iter = 1000
# num_points = 33
# x, y = np.meshgrid(np.linspace(-1, 1, num_points), np.linspace(-1, 1, num_points))
# x0 = np.stack([x, y], axis=-1)
# # compute 
# theta = SVGD_algo().update(x0, Qnet, num_points, n_iter=n_iter, stepsize=0.01)