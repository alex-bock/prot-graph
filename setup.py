
import setuptools


with open("README.md", "r") as f:
    description = f.read()


if __name__ == "__main__":
    
    setuptools.setup(
        name="prot_graph",
        description="Library for working with protein structure graphs",
        long_description=description,
        long_description_content_type="text/markdown",
        author="Alexander Bock",
        version="0.0.0",
        packages=setuptools.find_packages(),
        install_requires=[
            "biopython~=1.81",
            "numpy~=1.26.2",
            "pandas~=2.1.4",
            "plotly~=5.18.0",
            "scikit-learn~=1.3.2",
            "scipy~=1.11.4",
            "wandb~=0.17.0",
            "torchdrug @ git+https://git@github.com/alex-bock/torchdrug@dev#egg=torchdrug"
        ]
    )