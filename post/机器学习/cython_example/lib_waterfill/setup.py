from __future__ import absolute_import, print_function
import os
import sys

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = cythonize([
    Extension("waterfill",
              sources=["waterfill.pyx"],
              include_dirs=[os.getcwd(), np.get_include()],  # path to .h file(s), np.get_include(): avoid fatal error: numpy/arrayobject.h: No such file or directory”
              library_dirs=[os.getcwd()] # path to .a or .so file(s)
              ) 
],annotate=True, compiler_directives={'language_level' : "3"})

setup(
    name='waterfill',
    ext_modules=ext_modules,
)