from setuptools import setup, find_packages

setup(
    name="lucidity-preprocessor",
    version="0.1.0",
    author="Chris von Csefalvay",
    author_email="chris@chrisvoncsefalvay.com",
    description="Modular video ML inference pipeline with self-discovering model plugins",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "click>=8.1.0",
        "pydantic>=2.0.0",
        "tqdm>=4.65.0",
    ],
    entry_points={
        "console_scripts": [
            "lucidity=lucidity.cli:cli",
        ],
        "lucidity.models": [
            # Model plugins will register here
        ],
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
