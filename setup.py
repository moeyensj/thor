from setuptools import setup

setup(
   name="rascals",
   version="0.2.1.dev0",
   description="Range, Shift, Cluster and Link Scheme",
   license="BSD 3-Clause License",
   author="Joachim Moeyens, Mario Juric",
   author_email="moeyensj@uw.edu",
   url="https://github.com/moeyensj/RaSCaLS",
   packages=["rascals"],
   package_dir={"rascals": "rascals"},
   package_data={"rascals": ["data/*.orb",
                             "tests/data/*"]},
   setup_requires=["pytest-runner"],
   tests_require=["pytest"],
)
