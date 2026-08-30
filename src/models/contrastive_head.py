"""
Contrastive Learning Head
NT-Xent loss + KNN anomaly detection with memory bank
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLearningHead(nn.Module):
    """
    Contrastive Learning Head for anomaly detection
    
    Components:
    1. Projection MLP: hidden_dim → projection_dim
    2. NT-Xent (InfoNCE) loss for training
    3. Memory bank: stores normal transaction embeddings
    4. KNN anomaly score: distance to k-nearest neighbors in memory bank
    """
    
    def __init__(
        self,
        hidden_dim=128,
        projection_dim=64,
        temperature=0.07,
        memory_bank_size=10000,
        k_neighbors=10
    ):
        super().__init__()
        
        self.projection_dim = projection_dim
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.memory_bank_size = memory_bank_size
        self.k_neighbors = k_neighbors
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        # Memory bank (FIFO buffer of normal transaction embeddings)
        self.register_buffer('memory_bank', torch.randn(memory_bank_size, projection_dim))
        self.register_buffer('memory_labels', torch.zeros(memory_bank_size))
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
        
        # Normalize memory bank
        self.memory_bank = F.normalize(self.memory_bank, dim=1)
    
    def forward(self, embeddings, labels=None):
        """
        Args:
            embeddings: (batch, hidden_dim) - fused representations
            labels: (batch,) - fraud labels (0=legit, 1=fraud)
        
        Returns:
            projections: (batch, projection_dim) - L2-normalized projections
            contrastive_loss: scalar (if labels provided)
            anomaly_scores: (batch,) - KNN distance scores
        """
        # Project and normalize
        projections = self.projection(embeddings)  # (batch, projection_dim)
        projections = F.normalize(projections, dim=1)  # L2 normalize
        
        contrastive_loss = None
        if labels is not None and self.training:
            # Compute NT-Xent loss on legitimate samples
            contrastive_loss = self.nt_xent_loss(projections, labels)
            
            # Update memory bank with legitimate samples
            self.update_memory_bank(projections, labels)
        
        # Compute anomaly scores (KNN distance)
        anomaly_scores = self.compute_anomaly_score(projections)
        
        return projections, contrastive_loss, anomaly_scores
    
    def nt_xent_loss(self, projections, labels):
        """
        NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss
        Only computed on legitimate (non-fraud) samples
        """
        # Filter legitimate samples
        legit_mask = (labels == 0)
        if legit_mask.sum() < 2:
            return torch.tensor(0.0, device=projections.device)
        
        legit_proj = projections[legit_mask]  # (num_legit, projection_dim)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(legit_proj, legit_proj.t()) / self.temperature
        
        # Mask out diagonal (self-similarity)
        mask = torch.eye(sim_matrix.size(0), device=sim_matrix.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)  # Use large negative instead of -inf
        
        # Compute loss (pull similar samples together)
        loss = -torch.log_softmax(sim_matrix, dim=1).mean()
        
        # Clamp to prevent inf/nan propagation
        loss = torch.clamp(loss, max=100.0)
        
        return loss
    
    def update_memory_bank(self, projections, labels):
        """
        Update memory bank with legitimate transaction embeddings (FIFO)
        """
        legit_mask = (labels == 0)
        legit_proj = projections[legit_mask].detach()
        
        if legit_proj.size(0) == 0:
            return
        
        # Add to memory bank (FIFO)
        ptr = int(self.memory_ptr)
        batch_size = legit_proj.size(0)
        
        if ptr + batch_size <= self.memory_bank_size:
            self.memory_bank[ptr:ptr + batch_size] = legit_proj
            self.memory_labels[ptr:ptr + batch_size] = 0
            ptr = (ptr + batch_size) % self.memory_bank_size
        else:
            # Wrap around
            remaining = self.memory_bank_size - ptr
            self.memory_bank[ptr:] = legit_proj[:remaining]
            self.memory_bank[:batch_size - remaining] = legit_proj[remaining:]
            ptr = batch_size - remaining
        
        self.memory_ptr[0] = ptr
    
    def compute_anomaly_score(self, projections):
        """
        Compute anomaly score as average distance to k-nearest neighbors in memory bank
        Higher score = more anomalous
        """
        # Compute distances to all memory bank samples
        distances = torch.cdist(projections, self.memory_bank)  # (batch, memory_size)
        
        # Get k-nearest neighbor distances
        knn_distances, _ = torch.topk(distances, k=self.k_neighbors, largest=False, dim=1)
        
        # Anomaly score = mean of k-NN distances
        anomaly_scores = knn_distances.mean(dim=1)  # (batch,)
        
        return anomaly_scores


if __name__ == "__main__":
    print("Testing Contrastive Learning Head...")
    
    model = ContrastiveLearningHead(
        hidden_dim=128,
        projection_dim=64,
        memory_bank_size=1000
    )
    
    # Create dummy inputs
    batch_size = 32
    hidden_dim = 128
    
    embeddings = torch.randn(batch_size, hidden_dim)
    labels = torch.randint(0, 2, (batch_size,))  # 0=legit, 1=fraud
    
    # Forward pass
    model.train()
    projections, contrastive_loss, anomaly_scores = model(embeddings, labels)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Projections shape: {projections.shape}")
    print(f"Contrastive loss: {contrastive_loss.item() if contrastive_loss is not None else 'None'}")
    print(f"Anomaly scores shape: {anomaly_scores.shape}")
    print(f"Anomaly scores range: [{anomaly_scores.min():.3f}, {anomaly_scores.max():.3f}]")
    print("✅ Contrastive Learning Head test passed!")
