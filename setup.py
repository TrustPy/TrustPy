from setuptools import setup, find_packages

setup(
    name='trustpy_tools',
    version='2.0.12',
    author='Erim_Yanik',
    author_email='erimyanik@gmail.com',
    description='Trustworthiness metrics and calibration tools for predictive models',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/TrustPy/TrustPy',
    packages=find_packages(include=['trustpy', 'trustpy.*']),
    include_package_data=True,
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
    entry_points={
            'console_scripts': [
                'trustpy=trustpy.__main__:main',
            ],
    },
)
