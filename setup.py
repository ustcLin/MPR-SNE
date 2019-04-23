from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("random_walk\\random_walk_cython.pyx")
)


setup(
    ext_modules=cythonize("graph\\src\\get_nodes_table.pyx")
)
