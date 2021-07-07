# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

print(find_packages())    
#quit()
    
setup(
    name='EMPeaks',
    version='1.0.0',
    description='high-throughput spectrum peak modeling tools by using Spectrum adapted EM algorithms',
    long_description=readme,
    author='Yasunobu ANDO',
    author_email='yasunobu.ando@aist.go.jp',
    license=license,
    install_requires=['numpy', 'scipy', 'matplotlib'],
    packages=find_packages(exclude=('tests', 'docs'))
)

