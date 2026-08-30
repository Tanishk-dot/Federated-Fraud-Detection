"""
Temporal Transformer Encoder
Processes transaction sequences with multi-head self-attention
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Learned positional encoding for temporal sequences"""
    
    def __init__(self, d_model, max_len=50, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create learnable positional embeddings
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    """
    Temporal Transformer for transaction sequence encoding
    
    Architecture:
    - Input projection: input_dim → hidden_dim
    - Learnable CLS token (aggregates sequence context)
    - Positional encoding
    - 4 TransformerEncoder layers with 8 heads
    - GELU activation
    """
    
    def __init__(
        self,
        input_dim=10,
        hidden_dim=128,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
        max_seq_len=50
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Learnable CLS token (prepended to sequence)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim)
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim) - transaction sequences
        
        Returns:
            cls_output: (batch, hidden_dim) - CLS token representation
            sequence_output: (batch, seq_len+1, hidden_dim) - full sequence
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_proj(x)  # (batch, seq_len, hidden_dim)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, hidden_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq_len+1, hidden_dim)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer(x)  # (batch, seq_len+1, hidden_dim)
        
        # Normalize
        x = self.norm(x)
        
        # Extract CLS token and full sequence
        cls_output = x[:, 0, :]  # (batch, hidden_dim)
        sequence_output = x  # (batch, seq_len+1, hidden_dim)
        
        return cls_output, sequence_output


if __name__ == "__main__":
    # Test the model
    print("Testing Temporal Transformer...")
    
    model = TemporalTransformer(
        input_dim=10,
        hidden_dim=128,
        num_heads=8,
        num_layers=4
    )
    
    # Create dummy input
    batch_size = 4
    seq_len = 10
    input_dim = 10
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Forward pass
    cls_out, seq_out = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"CLS output shape: {cls_out.shape}")
    print(f"Sequence output shape: {seq_out.shape}")
    print("✅ Temporal Transformer test passed!")
