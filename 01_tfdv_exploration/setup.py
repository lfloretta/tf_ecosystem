import setuptools

setuptools.setup(
    name='tfdv_exploration',
    version='0.1',
    install_requires=['apache-beam[gcp]==2.14',
                      'tensorflow==2.5.3',
                      'tensorflow-data-validation==0.14.1',
                      'tensorflow-transform==0.14.0',
                      'pyarrow==0.14.0'],
    packages=setuptools.find_packages(),
)
