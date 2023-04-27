from setuptools import setup, find_namespace_packages

setup(name='music-generation-toolbox',
      version='0.6.0',
      description='Toolbox for generating music',
      author='Vincent Bons',
      url='https://github.com/wingedsheep/music-generation-toolbox',
      download_url='https://github.com/wingedsheep/music-generation-toolbox',
      license='MIT',
      install_requires=['pretty_midi>=0.2.10', 'miditoolkit>=0.1.16', 'scipy>=1.10.1',
                        'pylab-sdk>=1.3.2', 'requests>=2.29.0', 'matplotlib>=3.7.1',
                        'reformer-pytorch>=1.4.4', 'x-transformers>=1.10.0', 'torch>=2.0.0',
                        'numpy~=1.24.3', 'routing_transformer>=1.6.1', 'perceiver-ar-pytorch>=0.0.10',
                        'recurrent_memory_transformer_pytorch>=0.2.2'],
      packages=find_namespace_packages(),
      package_data={"": ["*.mid", "*.midi"]})
