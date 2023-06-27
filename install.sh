#!/bin/bash

# install Python dependencies in a virtual env
virtualenv -p python3.8 venv
. venv/bin/activate
pip install numpy scipy gym dotmap matplotlib tqdm opencv-python pyyaml omegaconf hydra-core
pip install numpy==1.19.5
pip install rl-games==1.1.4
pip install torch==1.13.1
pip install torchvision==0.14.1