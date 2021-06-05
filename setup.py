from setuptools import setup, find_namespace_packages

setup(name='music-generation-toolbox',
      version='1.0.0',
      description='Toolbox for generating music',
      author='Vincent Bons',
      url='https://github.com/wingedsheep/music-generation-toolbox',
      download_url='https://github.com/wingedsheep/music-generation-toolbox',
      license='MIT',
      install_requires=[],
      packages=find_namespace_packages(),
      package_data={"": ["*.mid", "*.midi"]})
