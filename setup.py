from tools import __version__
print(f'Installing tools@{__version__}')

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tools",
    version=__version__,
    author="JaeSung Yoo",
    author_email="jsyoo61@unc.edu",
    description="python syntax tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jsyoo61/tools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    # install_requires=['numpy','pandas','matplotlib']
)
