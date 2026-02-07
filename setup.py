"""Setup script for RAGOps. Package metadata is in pyproject.toml."""
from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(
        name="ragops",
        version="1.3.0",
        packages=find_packages(where=".", include=("ragops*", "core*", "pipeline*", "storage*", "config*")),
        package_dir={"": "."},
        entry_points={"console_scripts": ["ragops = ragops.cli.main:main"]},
    )
