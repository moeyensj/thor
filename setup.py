from setuptools import setup

setup(
   name="thor",
   license="BSD 3-Clause License",
   author="Joachim Moeyens, Mario Juric",
   author_email="moeyensj@uw.edu",
   long_description=open("README.md").read(),
   long_description_content_type="text/markdown",
   url="https://github.com/moeyensj/thor",
   packages=["thor"],
   package_dir={"thor": "thor"},
   package_data={"thor": ["tests/*.txt"]},
   use_scm_version=True,
   setup_requires=["pytest-runner", "setuptools_scm"],
   tests_require=["pytest"],
)
