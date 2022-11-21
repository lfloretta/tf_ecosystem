import setuptools

setuptools.setup(
    name='tft_train_test',
    version='0.1',
    install_requires=['apache-beam[gcp]==2.14',
                      'tensorflow==2.9.3',
                      'tensorflow-transform==0.14.0'],
    packages=[]
)
