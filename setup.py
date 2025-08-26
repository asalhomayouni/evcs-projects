from setuptools import setup, find_packages

setup(
    name="evcs",
    version="0.1.0",
    description="EV charging station optimization utilities",
    author="Asal Homayouni",
    package_dir={"": "src"},                
    packages=find_packages(where="src"),   
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "pyomo",
        "highspy", 
    ],
    python_requires=">=3.9",
)
