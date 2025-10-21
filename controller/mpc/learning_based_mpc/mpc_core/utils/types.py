# types.py
from typing import Callable
import numpy as np
Array = np.ndarray
DynFun = Callable[[float, Array], Array]
