# IIFL: Implicit Interactive Fleet Learning

## Installation

Installation instructions are similar to the [IFL Benchmark](https://github.com/BerkeleyAutomation/ifl_benchmark) on Github. First create a Python 3.8 virtual environment and install dependencies by running `. install.sh`.

To run the IFL Benchmark you will need to install Isaac Gym. Download Isaac Gym 1.0rc3 from https://developer.nvidia.com/isaac-gym (you may need to send a request but it should be quickly approved) and read the installation instructions in the docs to pip install into the virtual environment. You will need NVIDIA driver version >= 470.

Then clone NVIDIA IsaacGymEnvs from https://github.com/NVIDIA-Omniverse/IsaacGymEnvs and pip install it into the virtual environment. **Note**: make sure to run `git checkout 347cfbfaeeb708e7e94bc3bd8e7f2ef069e24fde` for the correct version of IsaacGymEnvs (1.3.0), since IsaacGymEnvs is actively under development.

## Reproducing Results

Simply run 

```. scripts/run_[env].sh```

where `env` is one of `{ant, anymal, ball_balance, franka_cube}`. This will run with default expert checkpoints and offline datasets, which you can re-generate if you wish.

## Acknowledgement
IFL implementation is based on the [IFL Benchmark](https://github.com/BerkeleyAutomation/ifl_benchmark). 
IBC implementation is adapted from [Kevin Zakka's PyTorch implementation](https://github.com/kevinzakka/ibc).
