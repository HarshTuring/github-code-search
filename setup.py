from setuptools import setup, find_packages

setup(
    name="github-codebase-analyzer",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.7",
    install_requires=[
        "gitpython>=3.1.0",
        "requests>=2.25.0",
    ],
    extras_require={
        "magic": [
            "python-magic>=0.4.24;platform_system!='Windows'",
            "python-magic-bin>=0.4.14;platform_system=='Windows'",
        ],
        "dev": [
            "pytest>=6.0.0",
            "black>=20.8b1",
            "isort>=5.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "github-analyzer=src.main:main",
            "fetch-repo=src.fetcher.cli:main",
            "parse-repo=src.parser.cli:main",
        ]
    },
    author="Harsh Master",
    author_email="harshmaster.h@turing.com",
    description="A tool for analyzing GitHub repositories and enabling natural language conversations about them",
    keywords="github, code, analysis, nlp",
    url="https://github.com/HarshTuring/github-code-search",
)