import argparse
import genesis as gs
import torch
from algo.dqn_agent import DQNAgent
from env import *
import os

gs.init(backend=gs.gpu, precision="32")

task_to_class = {
    'GraspFixedBlock': GraspFixedBlockEnv,
    'GraspFixedRod': GraspFixedRodEnv,
    'GraspRandomBlock': GraspRandomBlockEnv,
    'GraspRandomRod': GraspRandomRodEnv,
    'WaterFranka': WaterFrankaEnv,
    'ShadowHandBase': ShadowHandBaseEnv,
}

def create_environment(task_name):
    if task_name in task_to_class:
        return task_to_class[task_name]  
    else:
        raise ValueError(f"Task '{task_name}' is not recognized.")


def train_dqn(args):
    if args.load_path == "default":
        load = True
        checkpoint_path = f"logs/{args.task}_dqn_checkpoint_released.pth"
    elif args.load_path: 
        load = True
        checkpoint_path = args.load_path
    else:
        load = False
        checkpoint_path = f"logs/{args.task}_dqn_checkpoint.pth"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    env = create_environment(args.task)(vis=args.vis, device=args.device, num_envs=args.num_envs)
    print(f"Created environment: {env}")
    
    batch_size = args.batch_size if args.batch_size else 64 * args.num_envs
    replay_size = args.replay_size if args.replay_size else max(100000, 10 * batch_size) 
    agent = DQNAgent(input_dim=env.state_dim, output_dim=env.action_space, lr=1e-3, gamma=0.99, epsilon=0.5, epsilon_decay=0.995, epsilon_min=0.01, \
                     device=args.device, load=load, num_envs=args.num_envs, hidden_dim=args.hidden_dim, \
                        checkpoint_path=checkpoint_path, batch_size=batch_size, replay_size=replay_size)
    if args.device == "mps":
        gs.tools.run_in_another_thread(fn=run, args=(env, agent))
        env.scene.viewer.start()
    else:
        run(env, agent)

def run(env, agent):
    num_episodes = 500
    target_update_interval = 10
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = torch.zeros(env.num_envs).to(args.device)
        done_array = torch.tensor([False] * env.num_envs).to(args.device)
        for step in range(50):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.memory.add(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            total_reward += reward
            done_array = torch.logical_or(done_array, done)
            if done_array.all():
                break

        if episode % target_update_interval == 0:
            agent.update_target_network()
            agent.save_checkpoint()
        print(f"Episode {episode}, Total Reward: {total_reward}")

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False, help="Enable visualization") 
    parser.add_argument("-l", "--load_path", type=str, nargs='?', default=None, help="Path for loading model from checkpoint") 
    parser.add_argument("-n", "--num_envs", type=int, default=1, help="Number of environments to create") 
    parser.add_argument("-b", "--batch_size", type=int, default=None, help="Batch size for training")
    parser.add_argument("-r", "--replay_size", type=int, default=None, help="Size of replay buffer for DQN")
    parser.add_argument("-hd", "--hidden_dim", type=int, default=64, help="Hidden dimension for the network")
    parser.add_argument("-t", "--task", type=str, default="GraspFixedBlock", help="Task to train on")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="device: cpu or cuda:x or mps for macos")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parser()
    train_dqn(args)
