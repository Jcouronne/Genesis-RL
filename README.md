# Genesis Reinforcement Learning (RL) Framework

Genesis engine: https://genesis-world.readthedocs.io/en/latest/index.html <br>
Original repo: https://github.com/RochelleNi/GenesisEnvs

## Overview

This repository implements Proximal Policy Optimization (PPO) reinforcement learning in the Genesis physics engine. <br>
Only the scenario "PickPlaceRandomBlock" has been fine tuned and trained.
![](https://github.com/Jcouronne/Genesis-RL/blob/main/graphs/task_video.gif)

## Installation

### Prerequisites

Genesis officially supports Windows, Mac, and Linux. Since this repository was created using Ubuntu 22.04.5, the following installation guide should be easier to follow if you are on Ubuntu. Otherwise, follow the instructions on the Genesis website : https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html

Creating a Python virtual environment is highly recommended to avoid version mismatches in modules.  
Tutorial for creating virtual environments: https://www.youtube.com/watch?v=hrnN2BRfIXE

### Installation Steps
1. **Install Python**:
   ```bash
   sudo apt install python3.10
   ```
   *Genesis supports Python: >=3.10,<3.14*

2. **Install PyTorch**:
   Follow the official guide: https://pytorch.org/get-started/locally/ (copy the command you need)
   
   Check your CUDA version:
   ```bash
   nvidia-smi
   ```

3. **Install additional dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

Run the following to start training:
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

## Architecture Overview

```
run_ppo.py / run_ppo_test.py
│
├──> env/
│     ├── env/__init__.py
│     ├── grasp_fixed_block.py, pick_place_fixed_block.py, ...
│     └── Each defines a class for a specific environment (robot task)
│           - Handles 3D scene setup, resets, state/action/reward logic
│
├──> algo/ppo_agent.py
│     └── PPOAgent
│           - Handles agent initialization, loading/saving, action selection, training
│           - Uses network/ppo.py for neural network policy
│
└──> network/ppo.py
      └── PPO (nn.Module)
            - Defines the neural network architecture used by PPOAgent
```

---

## File-by-File Function Summary

### `run_ppo.py`
- **create_environment(task_name):** Returns the appropriate environment class for a given string.
- **train_ppo(args, lr, gamma, clip_epsilon, num_layers, hidden_dim):** Sets up environment and agent, loads models, handles training loop.
- **run(env, agent, num_episodes):** Main RL loop: resets env, collects experience, calls `agent.select_action`, interacts with env, and trains agent.

### `run_ppo_test.py`
- **Same structure as `run_ppo.py`**, but for evaluation/testing.  
- Loads a trained model and runs it in the environment for visualization or metrics.
- **Difference from standard PPO:** PPO test does not update the policy network during testing; it strictly evaluates the agent's performance using a fixed policy.

### `algo/ppo_agent.py`
- **PPOAgent class**
  - `__init__`: Initializes agent, neural network, optimizer, loads checkpoint if requested.  
    Inputs: state/action dimensions, hyperparameters, device, checkpoint path.
  - `save_checkpoint`: Saves model and optimizer state.
  - `load_checkpoint`: Loads model and optimizer state from file.
  - `select_action(state)`: Returns an action for a given state using the current policy.
  - `train(states, actions, rewards, dones)`: Updates policy network based on experience.

### `network/ppo.py`
- **PPO(nn.Module):**
  - Neural network used by PPOAgent.  
  - Sequence of Linear + LayerNorm + Swish layers, configurable depth/width.
  - `forward(x)`: Returns policy logits.

### `env/`
- **Each file (e.g., grasp_fixed_block.py, pick_place_fixed_block.py, shadow_hand.py, etc.)**
  - Defines an environment class inheriting from a base structure.
    - `__init__(vis, device, num_envs)`: Sets up scene/entities.
    - `build_env()`: Detailed scene and actuator setup.
    - `reset()`: Resets scene for new episode.
    - `step(action)`: For example, in `pick_place_fixed_block.py`, this function advances the environment using the given action, updates the robot and objects, computes the reward, and determines if the episode is done (e.g., checks if the block is placed correctly).
- **env/util.py**
  - `euler_to_quaternion(roll, pitch, yaw)`: Helper for coordinate conversions.

---

## Inter-file Call Graph and Dataflow

- `run_ppo.py` / `run_ppo_test.py`
  - Imports environment classes from `env/`
  - Instantiates `PPOAgent` from `algo/ppo_agent.py` (passes env state/action dims, hyperparameters, device)
  - Passes environment and agent to local `run()` function (training/testing loop)
    - Each episode:
      - Calls `env.reset()` (from env class)
      - For each step:
        - Calls `agent.select_action(state)` (returns action)
        - Calls `env.step(action)` (returns next state, reward, done)
      - After episode:
        - Calls `agent.train(states, actions, rewards, dones)` (training only (run_ppo.py); not called in PPO test/evaluation (run_ppo_test.py))

- `algo/ppo_agent.py`
  - Imports neural network from `network/ppo.py` (class PPO)
  - Uses `PPO` as policy network

- `env/`
  - Each environment class uses Genesis SDK, numpy, torch, and sometimes utility functions from `env/util.py`

---

## Example Dataflow

1. **Training:**
    - `run_ppo.py` parses args, sets hyperparams.
    - Chooses environment class from `env/`.
    - Creates environment (`env = GraspFixedBlockEnv(...)`) and agent (`agent = PPOAgent(...)`).
    - Training loop calls `agent.select_action(state)` → returns action.
    - Calls `env.step(action)` → returns next state, reward, done.
    - After batch/episode, calls `agent.train()`.

2. **Testing:**
    - `run_ppo_test.py` loads model checkpoint, runs agent in env for evaluation.
    - **Key difference:** The agent's policy is fixed; no training or policy update occurs during testing.
