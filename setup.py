from setuptools import setup
import pmlayer

DESCRIPTION = 'pmlayer: partially monotone layer'
NAME = 'pmlayer'
AUTHOR = 'Hiroki Yanagisawa'
AUTHOR_EMAIL = 'yanagis@jp.ibm.com'
URL = 'https://github.com/IBM/pmlayer'
LICENSE = 'Apache 2.0'
DOWNLOAD_URL = 'https://github.com/IBM/pmlayer'
VERSION = '1.0.1'
PYTHON_REQUIRES = ">=3.6"

INSTALL_REQUIRES = [
    'torch>=1.5',
    'numpy>=0.1',
]

PACKAGES = [
    'pmlayer',
    'pmlayer.common',
    'pmlayer.torch',
    'pmlayer.torch.layers'
]

CLASSIFIERS = [
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: Apache Software License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3 :: Only',
]

with open('README.md', 'r') as fp:
    readme = fp.read()
long_description = readme

setup(name=NAME,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      maintainer=AUTHOR,
      maintainer_email=AUTHOR_EMAIL,
      description=DESCRIPTION,
      long_description=long_description,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      python_requires=PYTHON_REQUIRES,
      install_requires=INSTALL_REQUIRES,
      extras_require={},
      packages=PACKAGES,
      classifiers=CLASSIFIERS
    )
