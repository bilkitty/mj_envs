import os
import sys
from setuptools import setup, find_packages

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))
    
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='mj_envs_vision',
    version='1.0.0',
    packages=find_packages(),
    description='environments simulated in MuJoCo',
    long_description=read('README.md'),
    url='https://github.com/bilkitty/mj_envs.git',
    install_requires=[
        'click', 'gym==0.26.2', 'mujoco-py<2.2,>=2.1.2', 'termcolor',
    ],
)