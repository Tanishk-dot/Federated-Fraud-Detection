import ray
import os

os.makedirs("/tmp/ray_sockets", exist_ok=True)
os.makedirs("/Volumes/Untitled/ray_tmp", exist_ok=True)

try:
    context = ray.init(
        _temp_dir="/Volumes/Untitled/ray_tmp",
        _plasma_store_socket_name="/tmp/ray_sockets/plasma.sock",
        _raylet_socket_name="/tmp/ray_sockets/raylet.sock"
    )
    print("RAY INITIALIZED SUCESSFULLY WITH SPLIT STORAGE!")
    print("Sockets mapped to /tmp, but Temp files mapped to /Volumes/Untitled")
    ray.shutdown()
except Exception as e:
    print(f"FAILED: {e}")
