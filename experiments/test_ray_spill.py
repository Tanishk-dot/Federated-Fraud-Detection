import ray
import json
import os

os.makedirs("/Volumes/Untitled/ray_tmp/spill", exist_ok=True)

try:
    context = ray.init(
        _temp_dir="/tmp/ray_fl",
        _system_config={
            "object_spilling_config": json.dumps(
                {"type": "filesystem", "params": {"directory_path": ["/Volumes/Untitled/ray_tmp/spill"]}}
            )
        }
    )
    print("RAY INITIALIZED SUCESSFULLY WITH SPLIT STORAGE!")
    ray.shutdown()
except Exception as e:
    print(f"FAILED: {e}")
