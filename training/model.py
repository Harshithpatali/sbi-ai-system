import torch
import torch.nn as nn


# =========================================
# PATCH EMBEDDING
# =========================================

class PatchEmbedding(nn.Module):

    def __init__(

        self,

        input_dim,

        embed_dim
    ):

        super().__init__()

        self.projection = nn.Linear(
            input_dim,
            embed_dim
        )

    def forward(self, x):

        return self.projection(x)


# =========================================
# PATCHTST MODEL
# =========================================

class PatchTST(nn.Module):

    def __init__(

        self,

        input_dim,

        seq_len=60,

        embed_dim=64,

        num_heads=4,

        num_layers=2,

        dropout=0.2
    ):

        super().__init__()

        # =====================================
        # PATCH EMBEDDING
        # =====================================

        self.patch_embedding = PatchEmbedding(

            input_dim=input_dim,

            embed_dim=embed_dim
        )

        # =====================================
        # POSITIONAL EMBEDDING
        # =====================================

        self.position_embedding = nn.Parameter(

            torch.randn(
                1,
                seq_len,
                embed_dim
            )
        )

        # =====================================
        # TRANSFORMER ENCODER
        # =====================================

        encoder_layer = nn.TransformerEncoderLayer(

            d_model=embed_dim,

            nhead=num_heads,

            dropout=dropout,

            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(

            encoder_layer,

            num_layers=num_layers
        )

        # =====================================
        # DROPOUT
        # =====================================

        self.dropout = nn.Dropout(dropout)

        # =====================================
        # FORECAST HEAD
        # =====================================

        self.fc = nn.Sequential(

            nn.Linear(embed_dim, 128),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(128, 1)
        )

    def forward(self, x):

        # =====================================
        # PATCH EMBEDDING
        # =====================================

        x = self.patch_embedding(x)

        # =====================================
        # POSITIONAL EMBEDDING
        # =====================================

        x = x + self.position_embedding

        # =====================================
        # TRANSFORMER ENCODER
        # =====================================

        x = self.transformer_encoder(x)

        # =====================================
        # GLOBAL AVERAGE POOLING
        # =====================================

        x = torch.mean(x, dim=1)

        # =====================================
        # FORECAST
        # =====================================

        output = self.fc(x)

        return output.squeeze()


# =========================================
# MODEL TEST
# =========================================

if __name__ == "__main__":

    batch_size = 32

    sequence_length = 60

    num_features = 19

    dummy_input = torch.randn(

        batch_size,

        sequence_length,

        num_features
    )

    model = PatchTST(

        input_dim=num_features
    )

    output = model(dummy_input)

    print("\nModel Created Successfully\n")

    print("Input Shape:")

    print(dummy_input.shape)

    print("\nOutput Shape:")

    print(output.shape)