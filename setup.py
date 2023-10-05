from setuptools import setup

setup(
    name="dashboard_funnel",
    version="1.1",
    packages=["models"],
    install_requires=[
        "pydantic>=2.3",
        "pymc==5.8.1",  # I'm assuming you're using pymc3, if not, specify the correct pymc version.
        "typing-extensions; python_version>='3.11'",  # This is only needed if you're supporting Python versions <3.8
    ],
)