from setuptools import setup, find_packages
from distutils.cmd import Command
import glob
import os


version_path = os.path.join(
    os.path.dirname(__file__),
    "transcriptomic_clustering",
    "VERSION.txt"
)
with open(version_path, "r") as version_file:
    version = version_file.read().strip()


readme_path = os.path.join(
    os.path.dirname(__file__),
    "README.md"
)
with open(readme_path, "r") as readme_file:
    readme = readme_file.read()


requirements_path = os.path.join(
    os.path.dirname(__file__), 'requirements.txt'
)
with open(requirements_path) as f:
    requirements = f.read().splitlines()

class CheckVersionCommand(Command):
    description = (
        "Check that this package's version matches a user-supplied version"
    )
    user_options = [
        ('expected-version=', "e", 'Compare package version against this value')
    ]

    def initialize_options(self):
        self.package_version = version
        self.expected_version = None
    
    def finalize_options(self):
        assert self.expected_version is not None
        if self.expected_version[0] == "v":
            self.expected_version = self.expected_version[1:]

    def run(self):
        if self.expected_version != self.package_version:
            raise ValueError(
                f"expected version {self.expected_version}, but this package "
                f"has version {self.package_version}"
            )

setup(
    version=version,
    name="transcriptomic_clustering",
    author="Allen Institute for Brain Science",
    author_email="Marmot@AllenInstitute.onmicrosoft.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    description="clustering pipeline for transcriptomic data",
    long_description=readme,
    long_description_content_type='text/markdown',
    entry_points={
        "console_scripts": [
            "",
        ]
    },
    keywords=["neuroscience", "bioinformatics", "scientific"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License", # Allen Institute Software License
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
    ], 
    cmdclass={'check_version': CheckVersionCommand}
)
