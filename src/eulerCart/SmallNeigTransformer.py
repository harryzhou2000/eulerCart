import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallMultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, only_1st_q=False):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv_proj = nn.Linear(dim, dim * 3)  # Project to Q, K, V
        self.out_proj = nn.Linear(dim, dim)  # Final projection

        self.only_1st_q = only_1st_q

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        qkv = self.qkv_proj(x)  # [B, S, 3*D]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # Each is [B, S, H, D_head]

        # Transpose for attention: [B, H, S, D_head]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if self.only_1st_q:
            q = q[:, :, 0:1, :]  # q: [B, H, 1, D_head]

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (
            self.head_dim**0.5
        )  # [B, H, S, S], if only_1st_q: [B, H, 1, S]
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(
            attn, v
        )  # [B, H, S, D_head], if only_1st_q: [B, H, 1, D_head]

        context = context.transpose(1, 2).reshape(
            batch_size, seq_len if not self.only_1st_q else 1, dim
        )  # [B, S, D], if only_1st_q: [B, 1, D]
        return self.out_proj(context)  # Final projection


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(F.tanh(self.fc1(x)))


class Input(nn.Module):
    def __init__(self, dim, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, dim)

    def forward(self, x):
        return self.fc1(x)


class Output(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, out_dim)

    def forward(self, x):
        return self.fc1(x)


class SmallGraphTransformerEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, ff_hidden_dim):
        super().__init__()
        self.self_attn = SmallMultiHeadSelfAttention(dim, num_heads, only_1st_q=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ff_hidden_dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):  # only_1st_q: from Seq=N_neighbor to Seq=1
        # Self-attention block
        attn_out = self.self_attn(x)
        x = x[:, -1:, :] # only preserve the first of local seq
        x = self.norm1(x + attn_out)

        # Feed-forward block
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


class SmallTransformerEncoder(nn.Module):
    def __init__(self, dim, num_heads, ff_hidden_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SmallGraphTransformerEncoderLayer(dim, num_heads, ff_hidden_dim)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x,
        F_MP,  # functor: message passing
    ):
        for iLayer, layer in enumerate(self.layers):
            x = layer(F_MP(x))
        return x


class SmallNeigTransformer(nn.Module):

    def __init__(
        self,
        dim_phy,
        dim,
        num_heads,
        ff_hidden_dim,
        num_encoder_layers,
    ):
        super().__init__()
        self.trans_encoder = SmallTransformerEncoder(
            dim,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
            num_layers=num_encoder_layers,
        )
        self.input_layer = Input(dim, dim_phy)
        self.output_layer = Output(dim, dim_phy)
        self.dim_phy = dim_phy
        self.dim = dim

    def forward(
        self,
        x: torch.Tensor,  # [xxx, dim_phy]
        F_MP,  # functor: message passing
    ):
        assert x.shape[-1] == self.dim_phy
        xshape = x.shape

        x_latent = self.input_layer(x)
        x_latentShape = x_latent.shape
        x_latent = x_latent.reshape((-1, 1, self.dim))
        x_latentShape_flat = x_latent.shape

        def inner_F_MP(x):
            MP_x = F_MP(x.reshape(x_latentShape))
            n_neigh = MP_x.shape[-2]
            return MP_x.reshape((-1, n_neigh, self.dim))

        x_latent = self.trans_encoder(x_latent, inner_F_MP)
        x = x + self.output_layer(x_latent).reshape(xshape)
        return x


if __name__ == "__main__":
    N_batch = 2
    N_x = 10
    N_y = 10
    dim_phy = 6
    dim_latent = 32
    dim_head = 32
    ff_hidden_dim = 64
    N_layers = 2

    x_in = torch.zeros((2, N_x, N_y, dim_phy))
    model = SmallNeigTransformer(
        dim_phy=dim_phy,
        dim=dim_latent,
        num_heads=dim_latent // dim_head,
        ff_hidden_dim=ff_hidden_dim,
        num_encoder_layers=N_layers,
    )

    def F_MP(x: torch.Tensor):
        assert x.ndim == 4
        xLe = x.roll(1, 1)
        xRi = x.roll(-1, 1)
        xLo = x.roll(1, 2)
        xUp = x.roll(-1, 2)
        xs = [x, xLe, xRi, xLo, xUp]
        xs = [t.unsqueeze(3) for t in xs]
        return torch.concat(xs, 3)

    x_out = model(x_in, F_MP)

    print(x_out.shape)
