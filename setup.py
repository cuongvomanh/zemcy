import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="zemcy",
    version="0.0.9",
    author="Cuong Vo Manh",
    author_email="cuong.vomanh195@gmail.com",
    description="zemcy library for computer vision",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cuongvomanh/zemcy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
