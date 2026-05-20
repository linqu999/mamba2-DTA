"""MambaTransDTA-style CNN protein encoder."""

from __future__ import annotations

from bimamba2_proteindta.models.torch_utils import NN, require_torch


def masked_max_pool(sequence, attention_mask):
    torch, _ = require_torch()
    mask = attention_mask.bool().unsqueeze(-1)
    fill_value = torch.finfo(sequence.dtype).min
    masked = sequence.masked_fill(~mask, fill_value)
    pooled = masked.max(dim=1).values
    all_padding = attention_mask.sum(dim=1).eq(0).unsqueeze(-1)
    return torch.where(all_padding, torch.zeros_like(pooled), pooled)


class ProteinCNNEncoder(NN.Module):
    """CNN protein encoder aligned with the MambaTransDTA-style baseline."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        num_filters: int = 32,
        kernel_sizes: tuple[int, int, int] = (4, 8, 12),
        output_dim: int = 96,
        padding_idx: int = 0,
        dropout: float = 0.1,
    ) -> None:
        _, nn = require_torch()
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList(
            nn.Conv1d(embed_dim, num_filters, kernel_size=kernel_size, padding=kernel_size // 2)
            for kernel_size in kernel_sizes
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(num_filters * len(kernel_sizes), output_dim)
        self.output_dim = output_dim

    def forward(self, input_ids, attention_mask):
        torch, _ = require_torch()
        embedded = self.embedding(input_ids).transpose(1, 2)
        conv_outputs = []
        for conv in self.convs:
            hidden = self.activation(conv(embedded)).transpose(1, 2)
            if hidden.size(1) != attention_mask.size(1):
                hidden = hidden[:, : attention_mask.size(1), :]
            conv_outputs.append(masked_max_pool(hidden, attention_mask))
        combined = torch.cat(conv_outputs, dim=-1)
        return self.projection(self.dropout(combined))
