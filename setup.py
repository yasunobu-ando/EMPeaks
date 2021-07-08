import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="EMPeaks",
    version="1.0.0",
    author="Yasunobu Ando",
    author_email="yasunobu.ando@aist.go.jp",
    description='high-throughput spectrum peak modeling tools by using Spectrum adapted EM algorithms',
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/pypa/sampleproject",
    # project_urls={
    #    "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    # },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD-3-clause License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "EMPeaks"},
    packages=setuptools.find_packages(where="EMpeaks"),
    python_requires=">=3.6",
)
