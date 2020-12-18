import random
import torch
import torch.nn as nn
from AnomalyDetector import ContextCriterion


# ----- ----- LOSS FUNCTION ----- ----- #
# ----- ----- ------------- ----- ----- #
class ReconstructionError(ContextCriterion):
    def score(self, model_out, target):
        pointwise_mse = torch.sum(torch.square(model_out - target) , axis=2) # Error of each point prediction
        batch_mse = torch.sum(pointwise_mse) # Sum of error of each prediction
        return torch.mean(batch_mse)


# ----- ----- MODELS ----- ----- #
# ----- ----- ------ ----- ----- #
class Seq2Seq(torch.nn.Module):
    def __init__(self, input_size=None, latent_size=64, rnn_layers=64, teacher_forcing_ratio=1.):
        super(Seq2Seq, self).__init__()
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.hidden_size = latent_size

        self.encoder = nn.GRU(input_size=input_size, hidden_size=latent_size, num_layers=rnn_layers, batch_first=True)
        self.decoder = nn.GRU(input_size=input_size, hidden_size=latent_size, num_layers=rnn_layers, batch_first=True)
        self.out = nn.Linear(latent_size, input_size)

    def encoder_init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

    def toembedding(self, X):
        _ , h_n = self.rnn(X)
        return h_n

    def context_anomaly(self, ctx):
        return NotImplementedError

    def forward(self, context=None, **kwargs): 
        X = context
        _, h_n = self.encoder(X)

        decoder_input = torch.zeros_like(X[:, 0:1, :]) # batch_size, sequence_len, features
        decoder_hidden = h_n
        
        x_tilde = torch.Tensor([])
        use_teacher_forcing = (True if random.random() < self.teacher_forcing_ratio else False)
        for i in range(X.size(1)): # sequence_len
            if use_teacher_forcing:
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                decoder_input = X[:, i:i+1, :]
            else:
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                decoder_input = decoder_output.detach()
            xi_tilde = self.out(decoder_output.squeeze()).unsqueeze(1)
            x_tilde = torch.cat([x_tilde, xi_tilde], axis=1)
        return x_tilde
