import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tools-jsyoo61", # Replace with your own username
    version="2020.12.4",
    author="JaeSung Yoo",
    author_email="jsyoo61@korea.ac.kr",
    description="personal syntax tool",
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
    install_requires=['numpy','pandas','matplotlib']
)
