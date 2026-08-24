from setuptools import setup, find_packages

setup(
    name="linear_algebra_and_stat",
    version="0.1.0",
    description="A mathematically rigorous exploration of linear algebra concepts required for ML.",
    author="Obssa Degefu",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
)
