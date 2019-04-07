
import numpy as np

def array_to_list(array, level):
    if level == 0:
        return array

    if isinstance(array, np.ndarray):
        return array_to_list(array.tolist(), level - 1)
    elif isinstance(array, list):
        return [array_to_list(item, level - 1) for item in array]
    elif isinstance(array, tuple):
        return tuple(array_to_list(item, level - 1) for item in array)
    else:
        return array