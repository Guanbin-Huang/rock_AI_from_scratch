from numpy.lib.npyio import load
from my_utils import save_var_to_pkl, load_var_from_pkl
import numpy as np

a = load_var_from_pkl("a")
b = load_var_from_pkl("b")

print(np.max(a - b))
