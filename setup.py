from setuptools import setup, find_packages

setup(
    name = 'preturnie',
    version = 0.1,
    url = 'https://github.com/lusterck',
    author_email = 'lucas.sterckx@gmail.com',
    description = 'A SpaCy pipeline and models for medical notes.',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        'Intended Audience :: Science/Research',
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    packages = find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    license="Apache",
    install_requires=[
        "spacy>=2.1.3",
        "awscli",
        "conllu",
        "numpy",
        "scispacy",
        "joblib",
        "nmslib>=1.7.3.6",
        "scikit-learn>=0.20.3"
        ],
    tests_require=[
        "pytest",
        "pytest-cov",
        "pylint"
        ],
    python_requires='>=3.6.0',
)