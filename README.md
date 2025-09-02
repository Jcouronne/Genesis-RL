# Genesis Reinforcement Learning (RL) Framework

Genesis engine: https://genesis-world.readthedocs.io/en/latest/index.html <br>
Original repo: https://github.com/RochelleNi/GenesisEnvs

## Overview

This repository implements Proximal Policy Optimization (PPO) reinforcement learning in the Genesis physics engine. <br>
Only the scenario "PickPlaceRandomBlock" has been fine tuned and trained.
![](https://github.com/Jcouronne/Genesis-RL/blob/main/graphs/task_video.gif)

## Installation

### Prerequisites

Genesis officially supports Windows, Mac, and Linux. Since this repository was created using Ubuntu 22.04.5, the following installation guide should be easier to follow if you are on Ubuntu. Otherwise, follow the instructions on the Genesis website linked above.

Creating a Python virtual environment is highly recommended to avoid version mismatches in modules.  
Tutorial for creating virtual environments: https://www.youtube.com/watch?v=hrnN2BRfIXE

### Installation Steps
1. **Install Python**:
   '''bash
   sudo apt install python3.10
   '''
   *Genesis supports Python: >=3.10,<3.14

3. **Install PyTorch**:
   Follow the official guide: https://pytorch.org/get-started/locally/ (copy the command you need)
   
   Check your CUDA version:
   ```bash
   nvidia-smi
   ```

4. **Install additional dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

Run the following to start training with enhanced visualization:
```bash
python run_ppo.py -n 30
```
*Use -n int to choose the number of environments running in parallel*

Specify a task with `-t taskname`:
```bash
python run_ppo.py -n 30 -t PickPlaceRandomBlock
```
*Default task: PickPlaceRandomBlock*

Load a pre-trained model with `-l directory`:
```bash
python run_ppo.py -n 30 -l
```
*Default directory: logs folder* <br>
*Note: Files must be marked with "_released" (e.g., PickPlaceRandomBlock_ppo_checkpoint_released.pth)*

### Evaluation

Run evaluation mode:
```bash
python run_ppo_test.py -n 30
```
*Uses the _released checkpoint in logs directory*

## Requirements

See `requirements.txt` for dependencies
