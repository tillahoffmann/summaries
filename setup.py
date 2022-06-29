from setuptools import find_packages, setup


setup(
    name='summaries',
    packages=find_packages(),
    version='0.1.0',
    install_requires=[
        'cmdstanpy',
        'doit',
        'matplotlib',
        'numpy',
        'pandas',
        'scikit-learn',
        'scipy',
        'torch',
        'tqdm',
    ],
    extras_require={
        'tests': [
            'flake8',
            'pytest',
            'pytest-bootstrap',
            'pytest-cov',
        ],
        'docs': [
            'sphinx',
        ],
    }
)
