from setuptools import setup

setup(
   name="thor",
   version="0.0.1.dev0",
   description="Tracklet-less Heliocentric Orbit Recovery",
   license="BSD 3-Clause License",
   author="Joachim Moeyens, Mario Juric",
   author_email="moeyensj@uw.edu",
   url="https://github.com/moeyensj/thor",
   packages=["thor"],
   package_dir={"thor": "thor"},
   package_data={"thor": ["data/*.orb",
                             "tests/data/*"]},
   setup_requires=["pytest-runner"],
   tests_require=["pytest"],
)
