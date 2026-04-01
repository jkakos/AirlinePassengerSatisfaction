import os
import sys


def modify_path():
    """
    Modify the path of the Jupyter notebook to include the higher level
    directories.

    """
    module_path = os.path.abspath(os.path.join('..'))
    if module_path not in sys.path:
        sys.path.append(module_path)
