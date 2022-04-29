import os
import sys
from setuptools import setup


package_basename = 'densitysplit'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), package_basename))
import _version
version = _version.__version__


setup(name=package_basename,
      version=version,
      author='Mathilde Pinon',
      author_email='',
      description='package for density splits',
      license='BSD3',
      url='http://github.com/mathildepinon/densitysplit',
      install_requires=['numpy', 'scipy', 'pypower'],
      extras_require={},
      package_data={},
      packages=[package_basename])
