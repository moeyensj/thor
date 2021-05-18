import numpy as np
from setuptools import setup
from Cython.Build import cythonize

setup(
    use_scm_version={
        "write_to": "thor/version.py",
        "write_to_template": "__version__ = '{version}'",
    },
    ext_modules=cythonize("thor/clusters.pyx", gdb_debug=True),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
