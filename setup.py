from setuptools import setup, find_packages

setup(
    name='trustpy-tools',
    version='2.0.5',
    author='Erim_Yanik',
    author_email='erimyanik@gmail.com',
    description='Trustworthiness metrics and calibration tools predictive models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yaniker/trustpy',
    packages=find_packages(include=['trustpy', 'trustpy.*']),
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
