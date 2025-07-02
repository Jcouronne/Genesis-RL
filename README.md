# Genesis Environment
This repository contains **parallelized** RL environment for Genesis general-purpose physics platform.

## 🔥 News

- [2025-01-18] Added a new `WaterFranka` environment to play with liquid! 
- [2025-01-10] Added successfully trained [checkpoints](https://github.com/RochelleNi/GenesisEnvs/tree/master/logs) for `GraspFixedBlock`, `GraspFixedRod` and `GraspRandomBlock`! 
- [2025-01-08] Supported [**MacOS**](#MacOS-Usage) training and visualization! 🎉
  
## Requirements
All necessary dependencies have been listed in `requirements.txt`.
You can create a conda environment by:

```bash
conda create --name genesis_env --file requirements.txt
```

## Command-line Arguments

- `-v` or `--vis` enables visualization.
- `-l` or `--load_path` specifies the loading path of a previously saved model checkpoint. Do **not** include this argument if you intend to train your model from scratch. If only `-l default` is provided, the default loading path will be: `logs/{task}_{algo}_checkpoint_released.pth`.
- `-n` or `--num_envs` specifies the number of parallel environments. If none is provided, the default is `1`.
- `-b` or `--batch_size` defines the batch size used for training. If none is provided, the default is `64 * num_envs`.
- `-r` or `--replay_size` defines the size of replay buffer for DQN. If none is provided, the default is `10 * batch_size`.
- `-hd` or `--hidden_dim` sets the hidden dimension for the network. If none is provided, the default is `64`.
- `-t` or `--task` specifies the task to train on. If none is provided, the default is `GraspFixedBlock`. Available tasks include:
  - `GraspFixedBlock`: Environment for grasping a fixed block.
  - `GraspFixedRod`: Environment for grasping a fixed rod.
  - `GraspRandomBlock`: Environment for grasping a randomly placed block.
  - `GraspRandomRod`: Environment for grasping a randomly placed rod.
  - `ShadowHandBase`: Environment for controlling a ShadowHand.
  - `WaterFranka`: Environment for interacting with liquid.



## Usage

- Training

You can run different learning algorithms with the following command structure. Here is an example of running training with 10 envs:
```bash
python run_{algo}.py -n 10
```
where `algo` can be `dqn`, `ppo` or `heuristic`.

<img  src="figs/train.gif" width="300">

- Evaluation

To test the trained policy, you can load a pretrained model from the checkpoint and visualize the rollout, by executing the script with the following command-line arguments:
```bash
python run_{algo}.py -l -v -n 1 -t GraspFixedBlock
```
Similarly, you can specify `algo` as you like.

<img  src="figs/eval.gif" width="300">

- Liquid

Apart from rigid bodies, you can also play with the `WaterFranka` environment to interact with liquid.
```bash
python run_dqn.py -v -n 1 -t WaterFranka
```

<img src="figs/water.gif" width="300">

## Saving and Loading Checkpoints

The agent periodically saves the model's weights and the target network state for later resumption. 

```python
def save_checkpoint(self, file_path):
    checkpoint = {
        'model_state_dict': self.model.state_dict(),
        'target_model_state_dict': self.target_model.state_dict()
    }
    torch.save(checkpoint, file_path)
```
You can load a checkpoint by setting the `--load` flag. We've provided a successfully trained checkpoint `logs/{task}_{algo}_checkpoint_released.pth` for a Franka robot to grasp a block, which you can use for evaluation.
```python
def load_checkpoint(self, file_path):
    checkpoint = torch.load(file_path)
    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
    self.model.eval()
    self.target_model.eval()
```

## MacOS Usage
- Training

You can add `-d mps` to train:
```bash
python run_dqn.py -n 10 -d mps
```

- Evaluation

You can add `-d mps` to eval and visualization:
```bash
python run_dqn.py -l -v -n 1 -t GraspFixedBlock -d mps
```