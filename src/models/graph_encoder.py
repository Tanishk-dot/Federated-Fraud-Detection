"""
Heterogeneous Graph Attention Network (HGAT)
Encodes transaction graph relationships using HGTConv
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import HGTConv, Linear as PyGLinear
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class HeterogeneousGraphEncoder(nn.Module):
    """
    Heterogeneous Graph Transformer Encoder
    
    Uses HGTConv (Heterogeneous Graph Transformer Convolution) to encode
    relationships between users, merchants, cards, devices, and locations.
    
    Architecture:
    - 3 HGTConv layers
    - 8 attention heads per layer
    - Layer normalization
    - Residual connections
    """
    
    def __init__(
        self,
        hidden_dim=128,
        num_layers=3,
        num_heads=8,
        dropout=0.1,
        metadata=None
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Default metadata if not provided
        if metadata is None:
            self.node_types = ['user', 'merchant']
            self.edge_types = [('user', 'transacts_at', 'merchant')]
        else:
            self.node_types, self.edge_types = metadata
        
        # HGT Convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=(self.node_types, self.edge_types),
                heads=num_heads,
                group='sum'
            )
            self.convs.append(conv)
        
        # Layer normalization for each node type
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x_dict, edge_index_dict):
        """
        Args:
            x_dict: Dict of node features {node_type: (num_nodes, hidden_dim)}
            edge_index_dict: Dict of edge indices {edge_type: (2, num_edges)}
        
        Returns:
            x_dict: Updated node features after graph convolution
        """
        for i, conv in enumerate(self.convs):
            # Apply HGT convolution
            x_dict_new = conv(x_dict, edge_index_dict)
            
            # Apply layer norm and dropout to each node type
            for node_type in x_dict_new.keys():
                x_dict_new[node_type] = self.norms[i](x_dict_new[node_type])
                x_dict_new[node_type] = self.dropout(x_dict_new[node_type])
                
                # Residual connection
                if node_type in x_dict:
                    x_dict_new[node_type] = x_dict_new[node_type] + x_dict[node_type]
            
            x_dict = x_dict_new
        
        return x_dict
    
    def get_user_embedding(self, x_dict):
        """Extract user node embeddings (for fusion with temporal branch)"""
        if 'user' in x_dict:
            return x_dict['user']
        else:
            # If no user nodes, return mean of all node embeddings
            all_embeddings = torch.cat([x for x in x_dict.values()], dim=0)
            return all_embeddings.mean(dim=0, keepdim=True)


class SimpleGraphEncoder(nn.Module):
    """
    Simplified graph encoder for when PyG HGTConv is not available
    Uses simple message passing
    """
    
    def __init__(self, hidden_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Simple MLP layers for message passing
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
    
    def forward(self, x_dict, edge_index_dict=None):
        """
        Simple forward pass - processes each node type's features independently.

        NOTE: this is a per-node-type residual MLP, not real message passing —
        there's no adjacency here (see HeterogeneousGraphEncoder above for the
        real HGTConv version, which needs edge_index_dict / entity IDs we don't
        currently have from the preprocessed sequence data). It exists so the
        graph branch contributes a real, per-sample signal instead of being fed
        constant zeros; each node type keeps its own batch dimension intact,
        which the previous version broke (it concatenated all node types along
        dim=0 then split the *batch* back apart as if it were node types).
        """
        if not isinstance(x_dict, dict):
            x = x_dict
            for layer in self.layers:
                x = layer(x) + x
            return x

        out_dict = {}
        for node_type, x in x_dict.items():
            for layer in self.layers:
                x = layer(x) + x  # Residual
            out_dict[node_type] = x
        return out_dict


if __name__ == "__main__":
    print("Testing Graph Encoder...")
    
    # Test with simple encoder (no PyG dependency for testing)
    model = SimpleGraphEncoder(hidden_dim=128, num_layers=3)
    
    # Create dummy graph data
    x_dict = {
        'user': torch.randn(10, 128),
        'merchant': torch.randn(20, 128)
    }
    
    # Forward pass
    out_dict = model(x_dict)
    
    print(f"Input user nodes: {x_dict['user'].shape}")
    print(f"Input merchant nodes: {x_dict['merchant'].shape}")
    print(f"Output user nodes: {out_dict['user'].shape}")
    print(f"Output merchant nodes: {out_dict['merchant'].shape}")
    print("✅ Graph Encoder test passed!")
