from numpy.lib.npyio import load
from my_utils import save_var_to_pkl, load_var_from_pkl
import numpy as np

a= load_var_from_pkl("ax")
b = load_var_from_pkl("bx")

print(np.allclose(a,b))