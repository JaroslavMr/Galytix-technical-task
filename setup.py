from setuptools import setup, find_packages

setup(
    name="text_matcher",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "gensim",
        "scipy"
    ],
    entry_points={
        "console_scripts": [
            "text-matcher = main:main"
        ]
    },
)
