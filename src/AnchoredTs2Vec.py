import numpy as np
import random
import sys
import torch
import torch.nn as nn
from skorch.net import NeuralNet
import torch.nn.functional as F


BETA_1 = .01
BETA_2 = .1


# ----- ----- LOSS FUNCTION ----- ----- #
# ----- ----- ------------- ----- ----- #
class ContextualCoherency():
    def score(self, model_out):
        e_actv, e_ap, e_an = model_out 
        ap_dist = F.relu(torch.norm((e_actv - e_ap), p=2, dim=1) - BETA_1)
        an_dist = F.relu(BETA_2 - torch.norm((e_actv - e_an), p=2, dim=1))
        return torch.mean(ap_dist + an_dist)

    def __call__(self, p1, p2=None):
        if isinstance(p1, NeuralNet): # p1=Model, p2=input
            with torch.no_grad():
                mout = p1.forward(p2)
                return self.score(mout)
        else: # p1=model_output
            return self.score(p1) 
    

# ----- ----- TUPLE MINING ----- ----- #
# ----- ----- ------------ ----- ----- #
def fast_filter(distances, discriminator, semi_hard=False):
    """Set to infinitum embeddings with same discriminator
    """
    discr_matrix = torch.stack([discriminator]*len(discriminator))
    same_discr = (discr_matrix == discr_matrix.T)
    distances[same_discr] = sys.maxsize

    # semi-hard triplet mining
    if semi_hard:
        distances[distances < BETA_1] = sys.maxsize-1

    return distances.argmin(axis=1)


def find_neg_anchors(e_actv, e_ap, discriminator):
    """Find negative anchors within a batch. 
    Embeddings with same discriminator are removed
    """
    # Computing distance matrix
    n = len(e_actv)
    dm = torch.pdist(e_actv)
    # Converting tu full nxn matrix
    tri = torch.zeros((n, n))
    tri[np.triu_indices(n, 1)] = dm
    fmatrix = torch.tril(tri.T, 1) + tri
    # Removing diagonal
    fmatrix += sys.maxsize * (torch.eye(n, n))
    # Getting the minimum
    idxs = fast_filter(fmatrix, discriminator) 
    dn = e_actv[idxs]
    
    return dn


# ----- ----- MODEL DEFINITION ----- ----- #
# ----- ----- ---------------- ----- ----- #
class AnchorTs2Vec(torch.nn.Module):
    def __init__(self, sigma=.0):
        super(AnchorTs2Vec, self).__init__()
        self.sigma = sigma

    def toembedding(self, x):
        raise NotImplementedError()

    def context_anomaly(self, ctx):
        activity_len = int(ctx.shape[1] / 2)
        a1 = ctx[:, :activity_len]
        a2 = ctx[:, activity_len:]
        return self.activity_coherency(a1, a2)

    def activity_coherency(self, a1, a2):
        """ Returns 1 if incoherency is detected, 0 otherwise
        """
        e_a1 = self.toembedding(a1)
        e_a2 = self.toembedding(a2)

        dist = (torch.norm(e_a1 - e_a2, p=2, dim=1) - BETA_1) / BETA_2
        dist += self.sigma
        return torch.clamp(dist, 0., 1.)

    def forward(self, context=None, host=None):
        context_len = context.shape[1]
        activity_len = int(context_len / 2)
        r = random.randint(0, context_len - activity_len)
        actv = context[:, r:r + activity_len]
        ap = context
        e_ap = self.toembedding(ap)
        e_actv = self.toembedding(actv)

        with torch.no_grad():
            e_an = find_neg_anchors(e_actv, e_ap,  host)
        return (e_actv, e_ap, e_an)


class GruLinear(AnchorTs2Vec):
    def __init__(self, input_size=None, sigma=.0, rnn_size=64, rnn_layers=64, latent_size=64):
        super(GruLinear, self).__init__(sigma)
        self.rnn = nn.GRU(input_size=input_size, hidden_size=rnn_size, num_layers=rnn_layers, batch_first=True)
        self.embedder = nn.Sequential(
            nn.Linear(rnn_size, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, latent_size),
            nn.ReLU())

    def toembedding(self, x):
        rnn_out, _ = self.rnn(x)
        e = self.embedder(rnn_out[:, -1])
        return e
