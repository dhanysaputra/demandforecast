from setuptools import setup, find_packages

setup(
    name="demandforecast",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "scikit-learn",
        "xgboost",
        "statsmodels"
    ],
    author="Dhany Saputra",
    description="Hybrid demand forecasting and inventory optimization system.",
    license="MIT",
)
