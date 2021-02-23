import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="image-registration",
    version="1.0.0",
    description="Python utility for registering multiband images.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SenteraLLC/image-registration",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "simpleitk",
        "matplotlib",
        ""
    ],
    extras_require={
        'dev': [
            'pytest',
            'sphinx_rtd_theme',
            'pylint',
            'm2r',
            'sphinx',
            'pipenv'
        ]
    },
)
