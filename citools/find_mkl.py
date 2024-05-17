from importlib import metadata
from ctypes import CDLL
mkl_rt=[p for p in metadata.files('mkl') if 'mkl_rt' in str(p)][0] 
resl=CDLL(mkl_rt.locate())
print(resl )
