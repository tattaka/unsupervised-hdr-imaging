from setuptools import find_packages, setup


def _requires_from_file(filename):
    return open(filename).read().splitlines()


setup(
    name="unsupervised_hdr",
    version="0.1.0",
    license="MIT License",
    description="A Library for unsupervised exposure changes from LDR video",
    author="tattaka",
    url="https://github.com/tattaka/unsupervised-hdr-imaging",
    packages=find_packages(),
    install_requires=_requires_from_file("requirements.txt"),
)
