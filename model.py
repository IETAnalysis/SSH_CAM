import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalTemporalEncoder(nn.Module):
    """
    Non-linear temporal redistribution unit.
    Implements CAM rectification (Eq. 4-6) or standard Logarithmic scaling.
    """

    def __init__(self, mode: str, d_time: int, kappa: float, delta: float):
        super().__init__()
        self.mode = mode
        self.kappa = kappa
        self.delta = delta
        self.proj = nn.Linear(1, d_time)

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        if self.mode == 'cam':
            # Hyperbolic compression and sigmoidal reshaping
            z = 2.0 * (torch.tanh(dt) - 0.5)
            z_rect = torch.clamp(z, -1.0 + self.delta, 1.0 - self.delta)
            z_log = torch.atanh(z_rect)
            p_t = torch.sigmoid(self.kappa * z_log)
            return self.proj(p_t.unsqueeze(-1))
        elif self.mode == 'log':
            return self.proj(torch.log1p(dt).unsqueeze(-1))
        else:
            raise ValueError(f"Unsupported temporal encoding mode: {self.mode}")


class GatedFeatureFusion(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Sigmoid()
        )
        self.transform = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-wise gating (Eq. 9)
        gate_weights = self.gate_net(x)
        return self.transform(x * gate_weights)


class SSHCAMSystem(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Heterogeneous Attribute Projectors
        self.embed_l = nn.Embedding(args.vocab_size, args.d_model // 2, padding_idx=0)
        self.embed_d = nn.Embedding(3, args.d_model // 4, padding_idx=0)
        self.enc_t = MultiModalTemporalEncoder(args.time_enc, args.d_model // 4, args.kappa, args.delta)

        self.fusion_module = GatedFeatureFusion(args.d_model)

        # Configurable Sequence Model Backbone
        if args.backbone == 'transformer':
            if args.d_model % args.nhead != 0:
                raise ValueError("Incompatible shapes: d_model must be multiple of nhead.")

            enc_layer = nn.TransformerEncoderLayer(
                d_model=args.d_model, nhead=args.nhead,
                dim_feedforward=args.d_model * 4, dropout=args.dropout, batch_first=True
            )
            self.seq_model = nn.TransformerEncoder(enc_layer, num_layers=args.n_layers)
        elif args.backbone == 'gru':
            self.seq_model = nn.GRU(
                input_size=args.d_model, hidden_size=args.d_model // 2,
                num_layers=args.n_layers, batch_first=True, bidirectional=True, dropout=args.dropout
            )
        else:
            raise NotImplementedError(f"Backbone '{args.backbone}' not implemented.")

        self.pooler = nn.Linear(args.d_model, 1)

        # Output Manifold Strategy
        if args.loss_type == 'gmm':
            self.gmm_centroids = nn.Parameter(torch.randn(args.num_classes, args.d_model))
            nn.init.orthogonal_(self.gmm_centroids)
        else:
            self.linear_head = nn.Linear(args.d_model, args.num_classes)

    def forward(self, l, t, d, mask):
        # 1. Feature Space Projection
        e_l = self.embed_l(l)
        e_d = self.embed_d(d)
        e_t = self.enc_t(t)

        h_in = torch.cat([e_l, e_d, e_t], dim=-1)
        h_fused = self.fusion_module(h_in)

        # 2. Representation Learning
        if self.args.backbone == 'transformer':
            features = self.seq_model(h_fused, src_key_padding_mask=mask)
        else:
            features, _ = self.seq_model(h_fused)

        # 3. Attention-Based Aggregation
        scores = self.pooler(features).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e12)

        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        latent_z = torch.sum(weights * features, dim=1)

        if self.args.loss_type == 'softmax':
            return self.linear_head(latent_z), latent_z
        return latent_z