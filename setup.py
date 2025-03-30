from setuptools import setup, find_packages

setup(
    name='trustquant',
    version='1.0.0',
    packages=find_packages(),
    install_requires=['numpy', 'scikit-learn', 'matplotlib'],
    author='Erim_Yanik',
    description='Trustworthiness metrics for Softmax predictive models',
    url='https://github.com/yaniker/trust-quantification-metrics-for-deep-neural-networks-python',
    license='MIT',
)
