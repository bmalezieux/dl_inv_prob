import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dl_inv_prob",
    version="0.0.1",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'scikit-learn',
        'torch',
        'pathlib',
        'tqdm',
        'joblib',
    ],
    python_requires='>=3.8',
)
