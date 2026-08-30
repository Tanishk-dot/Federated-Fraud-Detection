"""
Cross-Modal Fusion Module
Fuses graph and temporal representations using bidirectional cross-attention
"""

import torch
import torch.nn as nn


class CrossModalFusion(nn.Module):
    """
    Bidirectional Cross-Attention Fusion
    
    Allows graph and temporal branches to inform each other:
    1. Graph features attend to temporal sequence
    2. Temporal sequence attends to graph features
    3. Concatenate and fuse through MLP
    """
    
    def __init__(self, hidden_dim=128, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Graph → Temporal cross-attention
        self.graph_to_temporal = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Temporal → Graph cross-attention
        self.temporal_to_graph = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, graph_features, temporal_cls, temporal_sequence):
        """
        Args:
            graph_features: (batch, hidden_dim) - graph branch output
            temporal_cls: (batch, hidden_dim) - temporal CLS token
            temporal_sequence: (batch, seq_len, hidden_dim) - full temporal sequence
        
        Returns:
            fused: (batch, hidden_dim) - fused representation
        """
        batch_size = temporal_cls.size(0)
        
        # Ensure graph_features has batch dimension
        if graph_features.dim() == 1:
            graph_features = graph_features.unsqueeze(0).expand(batch_size, -1)
        elif graph_features.size(0) == 1 and batch_size > 1:
            graph_features = graph_features.expand(batch_size, -1)
        
        # Reshape for attention: (batch, 1, hidden_dim)
        graph_query = graph_features.unsqueeze(1)
        
        # 1. Graph attends to Temporal sequence
        # Query: graph, Key/Value: temporal sequence
        graph_enriched, _ = self.graph_to_temporal(
            query=graph_query,
            key=temporal_sequence,
            value=temporal_sequence
        )
        graph_enriched = graph_enriched.squeeze(1)  # (batch, hidden_dim)
        
        # 2. Temporal attends to Graph
        # Query: temporal CLS, Key/Value: graph
        temporal_query = temporal_cls.unsqueeze(1)  # (batch, 1, hidden_dim)
        graph_key_value = graph_features.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        temporal_enriched, _ = self.temporal_to_graph(
            query=temporal_query,
            key=graph_key_value,
            value=graph_key_value
        )
        temporal_enriched = temporal_enriched.squeeze(1)  # (batch, hidden_dim)
        
        # 3. Concatenate and fuse
        combined = torch.cat([graph_enriched, temporal_enriched], dim=-1)  # (batch, hidden_dim*2)
        fused = self.fusion_mlp(combined)  # (batch, hidden_dim)
        
        return fused


if __name__ == "__main__":
    print("Testing Cross-Modal Fusion...")
    
    model = CrossModalFusion(hidden_dim=128, num_heads=8)
    
    # Create dummy inputs
    batch_size = 4
    seq_len = 10
    hidden_dim = 128
    
    graph_features = torch.randn(batch_size, hidden_dim)
    temporal_cls = torch.randn(batch_size, hidden_dim)
    temporal_sequence = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Forward pass
    fused = model(graph_features, temporal_cls, temporal_sequence)
    
    print(f"Graph features shape: {graph_features.shape}")
    print(f"Temporal CLS shape: {temporal_cls.shape}")
    print(f"Temporal sequence shape: {temporal_sequence.shape}")
    print(f"Fused output shape: {fused.shape}")
    print("✅ Cross-Modal Fusion test passed!")
