Before using Poetry you should install the following packages separately, because they are not available via pip:
- pytables
  - conda install pytables
- torch for Cuda GPUs: https://pytorch.org/get-started/locally/
  - pip install torch --index-url https://download.pytorch.org/whl/cu117
  - pip install pytorch also works, but only for CPU (slower)