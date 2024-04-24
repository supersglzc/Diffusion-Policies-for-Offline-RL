import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


# from six.moves import xrange

# import umap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import wandb
import umap
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
# from torchvision.utils import make_grid
from pyclustering.cluster import cluster_visualizer_multidim
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import SIMPLE_SAMPLES
from matplotlib.patches import Rectangle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = 0.99
        self._epsilon = 1e-5
        self.anchor = 'closest'

    def forward(self, inputs):
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        #distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
        #            + torch.sum(self._embedding.weight**2, dim=1)
        #            - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        
        normed_z_flattened = F.normalize(flat_input, dim=1).detach()
        normed_codebook = F.normalize(self._embedding.weight, dim=1)
        distances = -torch.einsum('bd,dn->bn', normed_z_flattened, normed_codebook.t())

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # # Use EMA to update the embedding vectors
        # if self.training:
        #     self._ema_cluster_size = self._ema_cluster_size * self._decay + \
        #                              (1 - self._decay) * torch.sum(encodings, 0)
            
        #     # Laplace smoothing of the cluster size
        #     n = torch.sum(self._ema_cluster_size.data)
        #     self._ema_cluster_size = (
        #         (self._ema_cluster_size + self._epsilon)
        #         / (n + self._num_embeddings * self._epsilon) * n)
            
        #     dw = torch.matmul(encodings.t(), flat_input)
        #     self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
        #     self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        # loss = self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

         # online clustered reinitialisation for unoptimized points
        if self.training:
            # calculate the average usage of code entries
            self._ema_cluster_size.mul_(self._decay).add_(avg_probs, alpha= 1 - self._decay)
            # running average updates
            if self.anchor in ['closest', 'random', 'probrandom']:
                # closest sampling
                if self.anchor == 'closest':
                    sort_distance, indices = distances.sort(dim=0)
                    random_feat = flat_input.detach()[indices[-1,:]]
                # # feature pool based random sampling
                # elif self.anchor == 'random':
                #     random_feat = self.pool.query(z_flattened.detach())
                # # probabilitical based random sampling
                # elif self.anchor == 'probrandom':
                #     norm_distance = F.softmax(d.t(), dim=1)
                #     prob = torch.multinomial(norm_distance, num_samples=1).view(-1)
                #     random_feat = z_flattened.detach()[prob]
                # decay parameter based on the average usage
                decay = torch.exp(-(self._ema_cluster_size*self._num_embeddings*10)/(1-self._decay)-1e-3).unsqueeze(1).repeat(1, self._embedding_dim)
                self._embedding.weight.data = self._embedding.weight.data * (1 - decay) + random_feat * decay
        
        return loss, quantized, perplexity, encodings
    

class Encoder(nn.Module):
    '''
    Encoder module P(z_e|tau). Take a trajectory with states and actions and output a latent embedding z_e 
    '''
    def __init__(self, state_dim, action_dim, output_dim, lstm_hidden=256):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(state_dim+action_dim, 512), nn.ReLU(),
                                    nn.Linear(512, lstm_hidden)
        )
        # nn.Linear(state_dim+action_dim, lstm_hidden)
        self.lstm = nn.LSTM(input_size=lstm_hidden, hidden_size=lstm_hidden, batch_first=True, num_layers=2, bidirectional=True)
        self.out = nn.Sequential(nn.Linear(lstm_hidden*2, 256), nn.ReLU(),
                                    nn.Linear(256, output_dim)
        )

    def forward(self, states, actions):
        '''
        Takes 
        Inputs: 
            states: batch_size x T x state_dim
        Outputs:
            z_e: batch_size x embedding_dim
        '''
        batch_size, sequence_size, state_dim = states.shape
        action_dim = actions.shape[-1]
        states = states.reshape(-1, state_dim)
        actions = actions.reshape(-1, action_dim)
        s_a = torch.cat([states, actions], dim=-1)
        # [batch_size x T, lstm_hidden]
        embed = self.embed(s_a)  
        # [batch_size, T, lstm_hidden]
        embed = embed.reshape(batch_size, sequence_size, -1)  
        # [batch_size, T, lstm_hidden]
        o, _ = self.lstm(embed)  
        # take last output cell: [batch_size, lstm_hidden]
        o = o[:, -1, :]
        # [batch, embedding_dim]
        o = self.out(o)
        return o
    

class Decoder(nn.Module):
    '''
    Decoder module P(s_{t+1}, ..., s_{t+H}|z_q, s_t). Take a codevector z_q and a state s_t and output a state chunk 
    '''
    def __init__(self, embedding_dim, H_dim, state_dim, lstm_hidden=256):
        super().__init__()
        self.H_dim = H_dim
        self.state_embed = nn.Linear(state_dim, 256)
        self.embed = nn.Linear(embedding_dim+256, lstm_hidden)
        self.lstm = nn.LSTM(input_size=lstm_hidden, hidden_size=lstm_hidden, batch_first=True, num_layers=2, bidirectional=True)
        self.out = nn.Sequential(nn.Linear(lstm_hidden*2, 256), nn.ReLU(),
                                    nn.Linear(256, state_dim)
        )

    def forward(self, z, state):
        '''
        Inputs: 
            z_q: batch_size x embedding_dim
            state s_t: batch_size x state_dim
        Outputs:
            states: batch_size x H x state_dim
        '''
        batch_size, state_dim = state.shape
        embed_state = self.state_embed(state)
        # [batch_size, embedding_dim + state_dim]
        z_s = torch.cat([z, embed_state], dim=-1)
        # [batch_size, lstm_hidden]
        embed = self.embed(z_s)
        # [batch_size, H_dim, lstm_hidden]
        embed = embed.repeat(1, self.H_dim).reshape(batch_size, self.H_dim, -1)
        # [batch, H_dim, lstm_hidden]
        o, _ = self.lstm(embed)
        # [batch x H_dim, lstm_hidden]
        o = o.reshape(batch_size*self.H_dim, -1)
        # [batch x H_dim, state_dim]
        o = self.out(o)  
        o = o.reshape(batch_size, self.H_dim, state_dim)
        return o


class Prior(nn.Module):
    '''
    Prior module P(z_q|s_{t}, ..., s_{t+H}). Take a states chunk and output corresponding codevector z_q 
    '''
    def __init__(self, input_dim, output_dim, lstm_hidden=512):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU(),
                                    nn.Linear(512, lstm_hidden)
        )
        self.lstm = nn.LSTM(input_size=lstm_hidden, hidden_size=lstm_hidden, batch_first=True, num_layers=2, bidirectional=True)
        self.out = nn.Sequential(nn.Linear(lstm_hidden*2, 256), nn.ReLU(),
                                    nn.Linear(256, output_dim)
        )

    def forward(self, states):
        '''
        Takes 
        Inputs: 
            states: batch_size x T x state_dim
        Outputs:
            z_e: batch_size x embedding_dim
        '''
        batch_size, sequence_size, state_dim = states.shape
        states = states.reshape(-1, state_dim)
        # [batch_size x T, lstm_hidden]
        embed = self.embed(states)  
        # [batch_size, T, lstm_hidden]
        embed = embed.reshape(batch_size, sequence_size, -1)  
        # [batch_size, T, lstm_hidden]
        o, _ = self.lstm(embed)  
        # take last output cell: [batch_size, lstm_hidden]
        o = o[:, -1, :]
        # [batch, embedding_dim]
        o = self.out(o)
        return o
    
    
class Model(nn.Module):
    def __init__(self, state_dim, action_dim, H_dim, num_embeddings, embedding_dim, commitment_cost):
        super(Model, self).__init__()
        self._encoder = Encoder(state_dim=state_dim, action_dim=action_dim, output_dim=embedding_dim)
        self._vq_vae = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=commitment_cost)
        self._decoder = Decoder(embedding_dim=embedding_dim, H_dim=H_dim, state_dim=state_dim)
        self.action_prior = Prior(input_dim=state_dim, output_dim=num_embeddings)

    def forward(self, tau, tau_actions, state_chunk):
        z = self._encoder(tau, tau_actions)
        vq_loss, quantized, perplexity, encodings = self._vq_vae(z)
        x_reconstruction = self._decoder(quantized, state_chunk[:, 0])

        # loss
        reconstruction_loss = F.mse_loss(x_reconstruction, state_chunk[:, 1:])
        # similarity of codevectors
        codebook = self._vq_vae._embedding.weight
        differences = codebook.unsqueeze(1) - codebook.unsqueeze(0)
        squared_distances = torch.sum(differences ** 2, dim=2)
        exp_neg_distances = torch.exp(-squared_distances)
        distance_loss = torch.sum(torch.triu(exp_neg_distances, diagonal=1))
        # punish the scale of codevector
        l_infinity_norms = torch.max(torch.abs(codebook), dim=1).values
        adjusted_max_norm = torch.max(l_infinity_norms**2 - 1, torch.zeros_like(l_infinity_norms))
        max_loss = torch.max(adjusted_max_norm)
        loss = vq_loss + reconstruction_loss + 0.0 * distance_loss + 0.0 * max_loss
        
        result_dict = dict(
            reconstruction=x_reconstruction,
            reconstruction_loss=reconstruction_loss,
            distance_loss=distance_loss,
            max_loss=max_loss,
            encodings=encodings,
            perplexity=perplexity,
            quantized=quantized
        )
        return loss, result_dict
    
    @torch.no_grad()
    def encode(self, tau, tau_actions):
        z = self._encoder(tau, tau_actions)
        _, _, _, encodings = self._vq_vae(z)
        return encodings
    

def get_clusters(projected_embed):
    # Prepare initial centers - amount of initial centers defines amount of clusters from which X-Means will
    # start analysis.
    amount_initial_centers = 2
    initial_centers = kmeans_plusplus_initializer(projected_embed, amount_initial_centers).initialize()
    
    # Create instance of X-Means algorithm. The algorithm will start analysis from 2 clusters, the maximum
    # number of clusters that can be allocated is 20.
    xmeans_instance = xmeans(projected_embed, initial_centers, 20)
    xmeans_instance.process()
    
    # Extract clustering results: clusters and their centers
    clusters = xmeans_instance.get_clusters()
    centers = xmeans_instance.get_centers()
    return clusters, centers


def get_projection(embeddings, n_neighbors=10, min_dist=0.0):
    proj = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric='cosine').fit_transform(embeddings)
    fig, ax = plt.subplots()
    ax.scatter(proj[:,0], proj[:,1], alpha=0.3)
    fig.canvas.draw()  # Draw the canvas, cache the renderer
    image_flat = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
    embedding_img = image_flat.reshape(*reversed(fig.canvas.get_width_height()), 3)
    return proj, embedding_img, ax.get_xlim(), ax.get_ylim()


def plot_cluster(kwargs, traj, clusters,c=None):
    maze_map = kwargs['maze_map']
    maze_size = kwargs['maze_size_scaling']

    start = None
    goals = []
    blocks = []
    # find start and goal positions
    for i in range(len(maze_map)):
        for j in range(len(maze_map[0])):
            if maze_map[i][j] == 'r':
                start = (len(maze_map) - i - 1, j)
            elif maze_map[i][j] == 'g':
                goals.append((len(maze_map) - i - 1, j))
            elif maze_map[i][j] == 1:
                blocks.append((len(maze_map) - i - 1, j))

    fig, ax = plt.subplots()
    # compute limit
    x_lim = (-(start[1] + 0.5) * maze_size, (len(maze_map[0]) - 0.5 - start[1]) * maze_size)
    y_lim = (-(start[0] + 0.5) * maze_size, (len(maze_map) - 0.5 - start[0]) * maze_size)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    # draw blocks
    for block in blocks:
        x, y = x_lim[0] + maze_size * block[1], y_lim[0] + maze_size * block[0]  # maze_size * block[0] - y_lim[1]
        ax.add_patch(Rectangle((x, y), maze_size, maze_size, linewidth=0, rasterized=True, color='#C0C0C0'))
    
    # draw clusters
    colors = ['#808080', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', 
            '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', 
            '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', 
            '#000075', '#ffffff', '#000000']
    # colors = [
    #         "#B84C9B", "#50F6E3", "#9397F9", "#8CED1C", "#777221", "#9BDE77", "#4103EC", "#E844F0",
    #         "#63A9BC", "#52E7A7", "#D5DF52", "#E785F9", "#71C407", "#F9EAC9", "#173581", "#CD11DD",
    #         "#22BA04", "#3BE745", "#BBCE06", "#38CBF1", "#48FF69", "#DA598D", "#24D34F", "#65208C",
    #         "#4769A5", "#57C547", "#75D515", "#855A8B", "#69C942", "#36143F", "#403E9B", "#AE1B43",
    #         "#98EF80", "#010992", "#0BF415", "#87AA1A", "#27D89F", "#58F9CC", "#CDBC70", "#950157",
    #         "#57C7BD", "#9C9B3E", "#72EF74", "#5E27B9", "#4BC7E7", "#9AE681", "#55C8B5", "#C89F93",
    #         "#F6D1B0", "#6C5F9D", "#7D085D", "#131C2D", "#418640", "#51591E", "#37E0D6", "#904389",
    #         "#BEB828", "#809716", "#52E4CF", "#0BD870", "#8347AB", "#2CB2C1", "#BE4BC1", "#2E8F71",
    #         "#F1059F", "#FF143B", "#A7738D", "#1A766E", "#4D46AF", "#2C8AC9", "#514335", "#5402C0",
    #         "#AB91B9", "#818C09", "#CA9DB1", "#998BB9", "#99C5EA", "#F225CC", "#1499BA", "#C3664C",
    #         "#6E1375", "#43E408", "#601FBC", "#52EBAB", "#7FFA0D", "#1EC789", "#37812F", "#D6C6B3",
    #         "#23D567", "#059FC6", "#FE6181", "#025770", "#503DA7", "#2B28B6", "#ABF6D3", "#36EC99",
    #         "#D4F8FD", "#32388D", "#3212E6", "#AF13D9"
    #         ]

    for i in range(len(clusters)):
        points = []
        for j in range(len(clusters[i])):
            points.append(traj[clusters[i][j]])
        if len(points) != 0:
            points = np.concatenate(points)
            points[:, 1] = -points[:, 1]
            ax.scatter(points[:, 0], points[:, 1], s=0.01, color=colors[i] if c is None else c)

    # draw start and goal positions
    ax.plot(0, 0, 'ro')
    ax.annotate('start', (0, 0.25))
    for goal in goals:
        x = (goal[1] - start[1]) * maze_size
        y = (goal[0] - start[0]) * maze_size
        ax.plot(x, y, 'bo')
        ax.annotate('goal', (x, y + 0.25))
    # plt.show()
    # plt.savefig(f'dist_density/{name}.png')

    fig.canvas.draw()  # Draw the canvas, cache the renderer
    image_flat = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
    # reversed converts (W, H) from get_width_height to (H, W)
    image = image_flat.reshape(*reversed(fig.canvas.get_width_height()), 3)
    plt.close()
    return image
