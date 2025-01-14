import torch
import torch.nn as nn
import torch.nn.functional as F

from model.autoencoder import Decoder, GIN


class CVGAE(nn.Module):

    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes, n_cond, stats_mean, stats_std):
        super(CVGAE, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = GIN(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc).to(self.device)
        self.fc_mu = nn.Linear(hidden_dim_enc, latent_dim).to(self.device)
        self.fc_logvar = nn.Linear(hidden_dim_enc, latent_dim).to(self.device)
        self.decoder = Decoder(latent_dim + n_cond, hidden_dim_dec, n_layers_dec, n_max_nodes).to(self.device)

        self.latent_dim = latent_dim

        self.mean = stats_mean.to(self.device)
        self.std = stats_std.to(self.device)


    def forward(self, data):
        z, mu, logvar = self.encode(data)
        stats = (data.stats - self.mean) / self.std
        z_stats = torch.cat([z, stats], dim=1)
        adj = self.decoder(z_stats)
        return adj

    def encode(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            return mu

    def loss_function(self, data, beta=0.):
        z, mu, logvar = self.encode(data)
        stats = (data.stats - self.mean) / self.std
        z_stats = torch.cat([z, stats], dim=1)
        adj = self.decoder(z_stats)
        
        recon = F.l1_loss(adj, data.A, reduction='mean')
        kld = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1), dim = 0)
        thresh = torch.Tensor([.1]).to(self.device)
        loss = recon + beta * torch.maximum(kld, thresh)

        return loss, recon, kld
    
    def sample(self, stats, z = None):
        num_samples = stats.shape[0]
        if z is None:
            z = torch.randn(num_samples, self.latent_dim).to(self.device)
        stats = (stats - self.mean) / self.std
        z_stats = torch.cat([z, stats], dim=1)
        return self.decoder(z_stats)
