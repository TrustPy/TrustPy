from setuptools import setup, find_packages

setup(
    name='nettrustscore',
    version='2.0.2',
    author='Erim_Yanik',
    author_email='erimyanik@gmail.com',
    description='Trustworthiness metrics for Softmax predictive models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yaniker/nettrustscore-python',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20',
        'scikit-learn>=1.0',
        'matplotlib>=3.0'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
