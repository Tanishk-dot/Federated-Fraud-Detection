"""
Proxy Server for Secure Aggregation
Implements intermediate aggregation layer between clients and global server
"""

import torch
from typing import List, Dict, Optional
from .homomorphic_encryption import PaillierEncryption, SecureAggregator
import time


class ProxyServer:
    """
    Proxy Server for Secure Aggregation
    
    Responsibilities:
    1. Collect encrypted gradients from assigned clients
    2. Perform homomorphic aggregation (sum encrypted gradients)
    3. Forward aggregated encrypted sum to global server
    4. Never decrypt individual client gradients
    """
    
    def __init__(
        self,
        proxy_id: int,
        he: PaillierEncryption,
        assigned_clients: List[int]
    ):
        """
        Initialize proxy server
        
        Args:
            proxy_id: Unique identifier for this proxy
            he: Homomorphic encryption instance (public key only)
            assigned_clients: List of client IDs assigned to this proxy
        """
        self.proxy_id = proxy_id
        self.he = he
        self.assigned_clients = assigned_clients
        self.aggregator = SecureAggregator(he)
        
        # Statistics
        self.round_number = 0
        self.total_clients_served = 0
        self.total_aggregations = 0
        self.aggregation_times = []
        
        print(f"Proxy Server {proxy_id} initialized")
        print(f"   Assigned clients: {assigned_clients}")
    
    def receive_encrypted_gradient(
        self,
        client_id: int,
        encrypted_gradient: List[int]
    ) -> bool:
        """
        Receive encrypted gradient from a client
        
        Args:
            client_id: ID of the client sending gradient
            encrypted_gradient: Encrypted gradient from client
            
        Returns:
            success: Whether gradient was accepted
        """
        # Verify client is assigned to this proxy
        if client_id not in self.assigned_clients:
            print(f"Proxy {self.proxy_id}: Client {client_id} not assigned to this proxy")
            return False
        
        # Add to aggregator
        self.aggregator.add_client_gradient(encrypted_gradient)
        self.total_clients_served += 1
        
        print(f"Proxy {self.proxy_id}: Received encrypted gradient from Client {client_id}")
        print(f"   Gradient size: {len(encrypted_gradient)} encrypted values")
        
        return True
    
    def aggregate_and_forward(self) -> Optional[List[int]]:
        """
        Aggregate all received encrypted gradients
        
        Returns:
            aggregated_encrypted: Encrypted sum of all client gradients
        """
        start_time = time.time()
        
        num_clients = self.aggregator.get_num_clients()
        if num_clients == 0:
            print(f"Proxy {self.proxy_id}: No gradients to aggregate")
            return None
        
        print(f"\nProxy {self.proxy_id}: Aggregating {num_clients} encrypted gradients...")
        
        # Perform homomorphic aggregation (sum without decryption)
        aggregated_encrypted = self.aggregator.aggregate()
        
        # Record statistics
        aggregation_time = time.time() - start_time
        self.aggregation_times.append(aggregation_time)
        self.total_aggregations += 1
        self.round_number += 1
        
        print(f"Proxy {self.proxy_id}: Aggregation complete")
        print(f"   Clients aggregated: {num_clients}")
        print(f"   Aggregation time: {aggregation_time:.3f}s")
        print(f"   Output size: {len(aggregated_encrypted)} encrypted values")
        
        return aggregated_encrypted
    
    def get_statistics(self) -> Dict:
        """
        Get proxy server statistics
        
        Returns:
            stats: Dictionary of statistics
        """
        avg_aggregation_time = (
            sum(self.aggregation_times) / len(self.aggregation_times)
            if self.aggregation_times else 0
        )
        
        return {
            'proxy_id': self.proxy_id,
            'assigned_clients': self.assigned_clients,
            'round_number': self.round_number,
            'total_clients_served': self.total_clients_served,
            'total_aggregations': self.total_aggregations,
            'avg_aggregation_time': avg_aggregation_time,
            'last_aggregation_time': self.aggregation_times[-1] if self.aggregation_times else 0
        }
    
    def reset_round(self):
        """Reset for new round"""
        self.aggregator = SecureAggregator(self.he)


class ProxyServerManager:
    """
    Manager for multiple proxy servers
    Handles load balancing and coordination
    """
    
    def __init__(
        self,
        num_proxies: int,
        num_clients: int,
        he: PaillierEncryption
    ):
        """
        Initialize proxy server manager
        
        Args:
            num_proxies: Number of proxy servers
            num_clients: Total number of clients
            he: Homomorphic encryption instance
        """
        self.num_proxies = num_proxies
        self.num_clients = num_clients
        self.he = he
        
        # Assign clients to proxies (round-robin)
        self.client_assignments = self._assign_clients_to_proxies()
        
        # Create proxy servers
        self.proxies = []
        for proxy_id in range(num_proxies):
            assigned_clients = self.client_assignments[proxy_id]
            proxy = ProxyServer(proxy_id, he, assigned_clients)
            self.proxies.append(proxy)
        
        print(f"\nProxy Server Manager initialized")
        print(f"   Number of proxies: {num_proxies}")
        print(f"   Number of clients: {num_clients}")
        print(f"   Client assignments: {self.client_assignments}")
    
    def _assign_clients_to_proxies(self) -> Dict[int, List[int]]:
        """
        Assign clients to proxies using round-robin
        
        Returns:
            assignments: Dict mapping proxy_id to list of client_ids
        """
        assignments = {i: [] for i in range(self.num_proxies)}
        
        for client_id in range(self.num_clients):
            proxy_id = client_id % self.num_proxies
            assignments[proxy_id].append(client_id)
        
        return assignments
    
    def get_proxy_for_client(self, client_id: int) -> ProxyServer:
        """
        Get the proxy server assigned to a client
        
        Args:
            client_id: Client ID
            
        Returns:
            proxy: Proxy server for this client
        """
        proxy_id = client_id % self.num_proxies
        return self.proxies[proxy_id]
    
    def aggregate_all_proxies(self) -> List[int]:
        """
        Aggregate encrypted sums from all proxy servers
        
        Returns:
            global_encrypted: Final encrypted sum from all proxies
        """
        print("\nGlobal Server: Aggregating from all proxy servers...")
        
        proxy_aggregates = []
        for proxy in self.proxies:
            aggregated = proxy.aggregate_and_forward()
            if aggregated is not None:
                proxy_aggregates.append(aggregated)
        
        if not proxy_aggregates:
            raise ValueError("No proxy aggregates available")
        
        # Final homomorphic aggregation
        print(f"\nGlobal Server: Final aggregation of {len(proxy_aggregates)} proxy sums...")
        global_encrypted = self.he.aggregate_encrypted(proxy_aggregates)
        
        print(f"Global Server: Final aggregation complete")
        
        return global_encrypted
    
    def get_all_statistics(self) -> List[Dict]:
        """
        Get statistics from all proxy servers
        
        Returns:
            stats_list: List of statistics from each proxy
        """
        return [proxy.get_statistics() for proxy in self.proxies]
    
    def reset_all_proxies(self):
        """Reset all proxies for new round"""
        for proxy in self.proxies:
            proxy.reset_round()


# Example usage and testing
if __name__ == "__main__":
    print("Testing Proxy Server Implementation...")
    
    # Initialize encryption
    from homomorphic_encryption import PaillierEncryption
    he = PaillierEncryption(key_size=512)
    print(f"Generated encryption keys")
    
    # Create proxy server manager
    num_proxies = 2
    num_clients = 6
    manager = ProxyServerManager(num_proxies, num_clients, he)
    
    # Simulate clients sending encrypted gradients
    print("\n" + "="*60)
    print("SIMULATING FEDERATED LEARNING ROUND")
    print("="*60)
    
    client_gradients = []
    for client_id in range(num_clients):
        # Generate random gradient
        gradient = torch.randn(10)  # Small gradient for demo
        client_gradients.append(gradient)
        
        # Encrypt gradient
        encrypted_gradient = he.encrypt_tensor(gradient)
        
        # Send to assigned proxy
        proxy = manager.get_proxy_for_client(client_id)
        proxy.receive_encrypted_gradient(client_id, encrypted_gradient)
    
    # Aggregate at proxy level
    print("\n" + "="*60)
    print("PROXY AGGREGATION")
    print("="*60)
    
    # Aggregate at global level
    print("\n" + "="*60)
    print("GLOBAL AGGREGATION")
    print("="*60)
    
    global_encrypted = manager.aggregate_all_proxies()
    
    # Decrypt final result
    print("\n" + "="*60)
    print("DECRYPTION & VERIFICATION")
    print("="*60)
    
    global_decrypted = he.decrypt_tensor(global_encrypted, (10,))
    expected_sum = sum(client_gradients)
    
    print(f"\nDecrypted global gradient:\n{global_decrypted}")
    print(f"\nExpected sum:\n{expected_sum}")
    print(f"\nDifference: {torch.abs(global_decrypted - expected_sum).max():.6f}")
    
    # Print statistics
    print("\n" + "="*60)
    print("PROXY STATISTICS")
    print("="*60)
    
    for stats in manager.get_all_statistics():
        print(f"\nProxy {stats['proxy_id']}:")
        print(f"  Assigned clients: {stats['assigned_clients']}")
        print(f"  Total clients served: {stats['total_clients_served']}")
        print(f"  Total aggregations: {stats['total_aggregations']}")
        print(f"  Avg aggregation time: {stats['avg_aggregation_time']:.3f}s")
    
    print("\nAll tests passed!")
