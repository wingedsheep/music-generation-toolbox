from setuptools import setup, find_namespace_packages

setup(name='music-generation-toolbox',
      version='0.1.0',
      description='Toolbox for generating music',
      author='Vincent Bons',
      url='https://github.com/wingedsheep/music-generation-toolbox',
      download_url='https://github.com/wingedsheep/music-generation-toolbox',
      license='MIT',
      install_requires=['pretty_midi>=0.2.9', 'miditoolkit>=0.1.14', 'scipy>=1.6.3',
                        'pylab-sdk>=1.3.2', 'requests>=2.25.1', 'matplotlib>=3.4.2',
                        'reformer-pytorch>=1.4.2', 'x-transformers>=0.12.1', 'torch~=1.8.1'
                        'numpy~=1.20.3', 'routing_transformer>=1.5.2'],
      packages=find_namespace_packages(),
      package_data={"": ["*.mid", "*.midi"]})
