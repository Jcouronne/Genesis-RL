Original repo : https://github.com/RochelleNi/GenesisEnvs

## Usage ##

- Training :

#Run the following to start training :
```bash
python run_ppo.py -n 10
```

#To specify a task add "-t taskname", exemple :
```bash
python run_ppo.py -n 10 -t PickPlaceRandomBlock
```
If no task is specified PickPlaceRandomBlock is used.

To load a file, it must be marked with "_released", exemple : PickPlaceRandomBlock_ppo_checkpoint_released.pth

#To load an already trained model add -l :
```bash
python run_ppo.py -n 10 -l
```
If no path is specified the "logs" file is used.

- Evaluation :

#To run an evaluation use :
```bash
python run_ppo_test.py -n 10
```
The file marked as _released in logs will be used
