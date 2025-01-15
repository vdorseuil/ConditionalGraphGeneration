import torch
import torch.nn as nn
import torch.nn.functional as F
from model.autoencoder import GIN, Decoder


class StatsEncoder(nn.Module):
    # Encoder for the stats
    def __init__(self, hidden_dim, latent_dim, n_layers, n_cond, dropout=0.1):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stat_dim = n_cond
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers

        self.layers = nn.ModuleList(
            [nn.Linear(self.stat_dim, hidden_dim)]
            + [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 2)]
            + [nn.Linear(hidden_dim, latent_dim)]
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.normalize = nn.LayerNorm(normalized_shape=latent_dim)

    def forward(self, x):
        x = x.stats
        for layer in self.layers[:-1]:
            x = self.dropout(self.relu(layer(x)))
        x = self.layers[-1](x)
        return self.normalize(x)


class ContrastiveCVGAE(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim_enc,
        hidden_dim_stats_enc,
        hidden_dim_dec,
        latent_dim,
        n_layers_enc,
        n_layers_stats_enc,
        n_layers_dec,
        n_max_nodes,
        n_cond,
        stats_mean,
        stats_std,
    ):
        super(ContrastiveCVGAE, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = GIN(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc).to(
            self.device
        )
        self.stats_encoder = StatsEncoder(
            hidden_dim_stats_enc, latent_dim, n_layers_stats_enc, n_cond
        )
        self.fc_contrast = nn.Linear(hidden_dim_enc, latent_dim).to(self.device)
        self.fc_mu = nn.Linear(hidden_dim_enc, latent_dim).to(self.device)
        self.fc_logvar = nn.Linear(hidden_dim_enc, latent_dim).to(self.device)
        self.decoder = Decoder(
            latent_dim + n_cond, hidden_dim_dec, n_layers_dec, n_max_nodes
        ).to(self.device)

        self.latent_dim = latent_dim

        self.mean = stats_mean.to(self.device)
        self.std = stats_std.to(self.device)

    def forward(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        stats = (data.stats - self.mean) / self.std
        stats += torch.randn_like(stats) / 10
        x = torch.cat([x_g, stats], dim=1)
        adj = self.decoder(x)
        return adj

    def reparameterize(self, mu, logvar, eps_scale=1.0):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def encode(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        return x_g

    def encode_text(self, data):
        s = self.stats_encoder(data)
        return s

    def decode_mu(self, mu):
        adj = self.decoder(mu)
        return adj

    def contrastive_loss(self, z1, z2, temperature=0.5):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        sim_matrix = torch.matmul(z1, z2.T) / temperature
        labels = torch.arange(z1.size(0)).to(z1.device)
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

    def loss_function(self, data, beta=1e-3, gamma=1e-2):
        stats_enc = self.stats_encoder(data)
        x_g = self.encoder(data)
        contrast = self.contrastive_loss(self.fc_contrast(x_g), stats_enc)

        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)

        stats = (data.stats - self.mean) / self.std
        x = torch.cat([x_g, stats], dim=1)
        adj = self.decoder(x)

        recon = F.l1_loss(adj, data.A, reduction="mean")
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        loss = recon + beta * kld + gamma * contrast

        return loss, recon, kld, contrast

    def sample(self, stats):
        num_samples = stats.shape[0]
        z = ((torch.randn(num_samples, self.latent_dim)) / 100.0).to(self.device)
        stats = (stats - self.mean) / self.std
        x = torch.cat([z, stats], dim=1)
        return self.decode_mu(x)
