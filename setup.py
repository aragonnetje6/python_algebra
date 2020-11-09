import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="python_algebra",  # Replace with your own username
    version="0.0.1",
    author="Twan Stok",
    author_email="twanstok@gmail.com",
    description="Symbolic math package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aragonnetje6/python_algebra",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: win",
    ],
    python_requires='>=3.8',
)
