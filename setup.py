from setuptools import setup, find_packages

# Define setup.
setup(
    name="absl_extra",
    version="0.0.1",
    description="A wrapper to run and monitor absl app.",
    install_requires=["absl_py", "ml_collections", "slack_sdk"],
    url="http://github.com/aaarrti/absl_extra",
    author="Artem Sereda",
    author_email="artem.sereda.tub@gmail.com",
    packages=find_packages(),
    zip_safe=False,
    python_requires=">=3.7",
    include_package_data=True,
)
