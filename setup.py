from setuptools import setup, find_namespace_packages

setup(name='music-generation-toolbox',
      version='0.7.1',
      description='Toolbox for generating music',
      author='Vincent Bons',
      url='https://github.com/wingedsheep/music-generation-toolbox',
      download_url='https://github.com/wingedsheep/music-generation-toolbox',
      license='MIT',
      install_requires=['pretty_midi>=0.2.10', 'miditoolkit>=1.0.1', 'scipy>=1.15.3',
                        'pylab-sdk>=1.7.2', 'requests>=2.32.3', 'matplotlib>=3.10.3',
                        'reformer-pytorch>=1.4.4', 'x-transformers>=2.3.12', 'torch>=2.7.1',
                        'numpy>=2.2.6', 'routing_transformer>=1.6.1', 'perceiver-ar-pytorch>=0.0.10',
                        'recurrent_memory_transformer_pytorch>=0.7.0'],
      packages=find_namespace_packages(),
      package_data={"": ["*.mid", "*.midi"]})
