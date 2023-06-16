Note: This guide is meant for systems that already have a Nvidia GPU together with CUDA 11.7 installed.
If this does not match your system, simply remove the torch dependency from poetry to install the CPU version by default.
Before using Poetry you should install the following packages separately, because they are not available via pip:
- poetry for dependency management
  - pip install poetry
- install all dependencies
  - poetry install
