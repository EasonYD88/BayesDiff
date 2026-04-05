import torch
print("torch:", torch.__version__)
try:
    import torch_geometric
    print("PyG: OK")
except ImportError as e:
    print(f"PyG: MISSING - {e}")
try:
    import rdkit
    print("rdkit: OK")
except ImportError as e:
    print(f"rdkit: MISSING - {e}")
try:
    import matplotlib
    print("matplotlib: OK")
except ImportError as e:
    print(f"matplotlib: MISSING - {e}")
try:
    import easydict
    print("easydict: OK")
except ImportError as e:
    print(f"easydict: MISSING - {e}")
try:
    import lmdb
    print("lmdb: OK")
except ImportError as e:
    print(f"lmdb: MISSING - {e}")
try:
    import yaml
    print("yaml: OK")
except ImportError as e:
    print(f"yaml: MISSING - {e}")
