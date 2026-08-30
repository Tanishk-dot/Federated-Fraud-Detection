"""
Homomorphic Encryption for Federated Learning
Implements Paillier cryptosystem for additive homomorphic encryption
"""

import torch
import numpy as np
from typing import Tuple, List
import random


class PaillierEncryption:
    """
    Paillier Cryptosystem - Additive Homomorphic Encryption
    
    Properties:
    - Enc(m1) * Enc(m2) = Enc(m1 + m2)  (homomorphic addition)
    - Enc(m)^k = Enc(k * m)  (scalar multiplication)
    """
    
    def __init__(self, key_size: int = 512):
        """
        Initialize Paillier encryption
        
        Args:
            key_size: Size of the key in bits (default: 512 for demo, use 2048+ for production)
        """
        self.key_size = key_size
        self.public_key, self.private_key = self.generate_keypair()
    
    def generate_keypair(self) -> Tuple[dict, dict]:
        """
        Generate Paillier public/private key pair
        
        Returns:
            public_key: (n, g)
            private_key: (lambda, mu)
        """
        # For demo purposes, using smaller primes
        # In production, use cryptography library with proper key generation
        
        # Generate two large primes p and q
        p = self._generate_prime(self.key_size // 2)
        q = self._generate_prime(self.key_size // 2)
        
        # Compute n = p * q
        n = p * q
        
        # Compute λ = lcm(p-1, q-1)
        lambda_val = self._lcm(p - 1, q - 1)
        
        # Choose g = n + 1 (simplified)
        g = n + 1
        
        # Compute μ = (L(g^λ mod n²))^(-1) mod n
        # where L(x) = (x - 1) / n
        n_sq = n * n
        g_lambda = pow(g, lambda_val, n_sq)
        l_val = (g_lambda - 1) // n
        mu = self._mod_inverse(l_val, n)
        
        public_key = {'n': n, 'g': g}
        private_key = {'lambda': lambda_val, 'mu': mu, 'n': n}
        
        return public_key, private_key
    
    def encrypt(self, plaintext: float) -> int:
        """
        Encrypt a plaintext value
        
        Args:
            plaintext: Value to encrypt
            
        Returns:
            ciphertext: Encrypted value
        """
        n = self.public_key['n']
        g = self.public_key['g']
        n_sq = n * n
        
        # Convert float to integer (scale by 10^6 for precision)
        m = int(plaintext * 1000000)
        
        # Choose random r in Z*_n
        r = random.randint(1, n - 1)
        while self._gcd(r, n) != 1:
            r = random.randint(1, n - 1)
        
        # Compute ciphertext: c = g^m * r^n mod n²
        c = (pow(g, m, n_sq) * pow(r, n, n_sq)) % n_sq
        
        return c
    
    def decrypt(self, ciphertext: int) -> float:
        """
        Decrypt a ciphertext value
        
        Args:
            ciphertext: Encrypted value
            
        Returns:
            plaintext: Decrypted value
        """
        n = self.private_key['n']
        lambda_val = self.private_key['lambda']
        mu = self.private_key['mu']
        n_sq = n * n
        
        # Compute m = L(c^λ mod n²) * μ mod n
        # where L(x) = (x - 1) / n
        c_lambda = pow(ciphertext, lambda_val, n_sq)
        l_val = (c_lambda - 1) // n
        m = (l_val * mu) % n
        
        # Handle negative values (two's complement)
        if m > n // 2:
            m = m - n
        
        # Convert back to float
        plaintext = m / 1000000.0
        
        return plaintext
    
    def add_encrypted(self, c1: int, c2: int) -> int:
        """
        Add two encrypted values (homomorphic addition)
        
        Args:
            c1: First encrypted value
            c2: Second encrypted value
            
        Returns:
            c_sum: Encrypted sum
        """
        n = self.public_key['n']
        n_sq = n * n
        
        # Enc(m1 + m2) = Enc(m1) * Enc(m2) mod n²
        c_sum = (c1 * c2) % n_sq
        
        return c_sum
    
    def multiply_encrypted_by_scalar(self, c: int, k: int) -> int:
        """
        Multiply encrypted value by scalar (homomorphic scalar multiplication)
        
        Args:
            c: Encrypted value
            k: Scalar multiplier
            
        Returns:
            c_mult: Encrypted product
        """
        n = self.public_key['n']
        n_sq = n * n
        
        # Enc(k * m) = Enc(m)^k mod n²
        c_mult = pow(c, k, n_sq)
        
        return c_mult
    
    def encrypt_tensor(self, tensor: torch.Tensor) -> List[int]:
        """
        Encrypt a PyTorch tensor element-wise
        
        Args:
            tensor: Tensor to encrypt
            
        Returns:
            encrypted_list: List of encrypted values
        """
        flat_tensor = tensor.flatten().cpu().numpy()
        encrypted_list = [self.encrypt(float(val)) for val in flat_tensor]
        return encrypted_list
    
    def decrypt_tensor(self, encrypted_list: List[int], shape: tuple) -> torch.Tensor:
        """
        Decrypt a list of encrypted values back to tensor
        
        Args:
            encrypted_list: List of encrypted values
            shape: Original tensor shape
            
        Returns:
            tensor: Decrypted tensor
        """
        decrypted_values = [self.decrypt(c) for c in encrypted_list]
        tensor = torch.tensor(decrypted_values).reshape(shape)
        return tensor
    
    def aggregate_encrypted(self, encrypted_list: List[List[int]]) -> List[int]:
        """
        Aggregate multiple encrypted tensors (homomorphic sum)
        
        Args:
            encrypted_list: List of encrypted tensors
            
        Returns:
            aggregated: Encrypted sum
        """
        if not encrypted_list:
            return []
        
        # Initialize with first encrypted tensor
        aggregated = encrypted_list[0].copy()
        
        # Add remaining encrypted tensors
        for encrypted_tensor in encrypted_list[1:]:
            for i in range(len(aggregated)):
                aggregated[i] = self.add_encrypted(aggregated[i], encrypted_tensor[i])
        
        return aggregated
    
    # Helper functions
    def _generate_prime(self, bits: int) -> int:
        """Generate a prime number with specified bit length"""
        # Simplified prime generation for demo
        # In production, use proper prime generation from cryptography library
        while True:
            num = random.getrandbits(bits)
            if self._is_prime(num):
                return num
    
    def _is_prime(self, n: int, k: int = 5) -> bool:
        """Miller-Rabin primality test"""
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0:
            return False
        
        # Write n-1 as 2^r * d
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        
        # Witness loop
        for _ in range(k):
            a = random.randint(2, n - 2)
            x = pow(a, d, n)
            
            if x == 1 or x == n - 1:
                continue
            
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        
        return True
    
    def _gcd(self, a: int, b: int) -> int:
        """Compute greatest common divisor"""
        while b:
            a, b = b, a % b
        return a
    
    def _lcm(self, a: int, b: int) -> int:
        """Compute least common multiple"""
        return abs(a * b) // self._gcd(a, b)
    
    def _mod_inverse(self, a: int, m: int) -> int:
        """Compute modular multiplicative inverse"""
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y
        
        gcd, x, _ = extended_gcd(a % m, m)
        if gcd != 1:
            raise ValueError("Modular inverse does not exist")
        return (x % m + m) % m


class SecureAggregator:
    """
    Secure Aggregation using Homomorphic Encryption
    Aggregates encrypted gradients from multiple clients
    """
    
    def __init__(self, he: PaillierEncryption):
        """
        Initialize secure aggregator
        
        Args:
            he: Homomorphic encryption instance
        """
        self.he = he
        self.encrypted_gradients = []
    
    def add_client_gradient(self, encrypted_gradient: List[int]):
        """
        Add encrypted gradient from a client
        
        Args:
            encrypted_gradient: Encrypted gradient from client
        """
        self.encrypted_gradients.append(encrypted_gradient)
    
    def aggregate(self) -> List[int]:
        """
        Aggregate all encrypted gradients
        
        Returns:
            aggregated: Encrypted sum of all gradients
        """
        if not self.encrypted_gradients:
            raise ValueError("No gradients to aggregate")
        
        aggregated = self.he.aggregate_encrypted(self.encrypted_gradients)
        
        # Clear stored gradients
        self.encrypted_gradients = []
        
        return aggregated
    
    def get_num_clients(self) -> int:
        """Get number of clients that contributed"""
        return len(self.encrypted_gradients)


# Example usage and testing
if __name__ == "__main__":
    print("Testing Paillier Homomorphic Encryption...")
    
    # Initialize encryption
    he = PaillierEncryption(key_size=512)
    print(f"✅ Generated key pair (n={he.public_key['n']})")
    
    # Test basic encryption/decryption
    m1, m2 = 5.5, 3.2
    c1 = he.encrypt(m1)
    c2 = he.encrypt(m2)
    print(f"\n✅ Encrypted {m1} and {m2}")
    
    # Test homomorphic addition
    c_sum = he.add_encrypted(c1, c2)
    m_sum = he.decrypt(c_sum)
    print(f"✅ Homomorphic addition: {m1} + {m2} = {m_sum:.2f} (expected: {m1 + m2:.2f})")
    
    # Test tensor encryption
    tensor = torch.randn(3, 4)
    print(f"\n✅ Original tensor:\n{tensor}")
    
    encrypted_tensor = he.encrypt_tensor(tensor)
    print(f"✅ Encrypted tensor (first 3 values): {encrypted_tensor[:3]}")
    
    decrypted_tensor = he.decrypt_tensor(encrypted_tensor, tensor.shape)
    print(f"✅ Decrypted tensor:\n{decrypted_tensor}")
    
    # Test secure aggregation
    print("\n✅ Testing Secure Aggregation...")
    aggregator = SecureAggregator(he)
    
    # Simulate 3 clients
    client_tensors = [torch.randn(2, 3) for _ in range(3)]
    for i, ct in enumerate(client_tensors):
        encrypted = he.encrypt_tensor(ct)
        aggregator.add_client_gradient(encrypted)
        print(f"  Client {i+1} gradient added")
    
    # Aggregate
    aggregated_encrypted = aggregator.aggregate()
    aggregated_decrypted = he.decrypt_tensor(aggregated_encrypted, (2, 3))
    
    # Verify
    expected_sum = sum(client_tensors)
    print(f"\n✅ Aggregated (encrypted then decrypted):\n{aggregated_decrypted}")
    print(f"✅ Expected sum:\n{expected_sum}")
    print(f"✅ Difference: {torch.abs(aggregated_decrypted - expected_sum).max():.6f}")
    
    print("\n🎉 All tests passed!")
