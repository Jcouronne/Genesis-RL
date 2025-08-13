import argparse
import genesis as gs
import torch
from algo.ppo_agent import PPOAgent
from env import *
import os
import matplotlib.pyplot as plt
from datetime import datetime
import time
import sys

num_episodes = 200
lr=1e-3
gamma=0.85
clip_epsilon=0.2
num_layers = 12
hidden_dim = 32
gs.init(backend=gs.gpu, precision="64")
task_to_class = {
    'GraspFixedBlock': GraspFixedBlockEnv,
    'PickPlaceFixedBlock': PickPlaceFixedBlockEnv,
    'PickPlaceRandomBlock': PickPlaceRandomBlockEnv,
    'ComplexPickPlaceRandomBlock': ComplexPickPlaceRandomBlockEnv,
    'GraspFixedRod': GraspFixedRodEnv,
    'GraspRandomBlock': GraspRandomBlockEnv,
    'GraspRandomRod': GraspRandomRodEnv,
    'ShadowHandBase': ShadowHandBaseEnv
}

def create_environment(task_name):
    if task_name in task_to_class:
        return task_to_class[task_name]  
    else:
        raise ValueError(f"Task '{task_name}' is not recognized.")

def train_ppo(args, lr, gamma, clip_epsilon, num_layers, hidden_dim):
    # Load file
    if args.load_path is not None:
        load = True
        checkpoint_path = f"logs/{args.task}_ppo_checkpoint_released.pth"
        print(f"Loading default checkpoint from {checkpoint_path}")
    else:
        load = False
        checkpoint_path = f"logs/{args.task}_ppo_checkpoint.pth"
        print(f"Creating new checkpoint at {checkpoint_path}")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    env = create_environment(args.task)(vis=args.vis, device=args.device, num_envs=args.num_envs)
    print(f"Created environment: {env}")

    agent = PPOAgent(input_dim=env.state_dim, output_dim=env.action_space, lr=lr, gamma=gamma, clip_epsilon=clip_epsilon, num_layers=num_layers, hidden_dim=hidden_dim, device=args.device, load=load, checkpoint_path=checkpoint_path)
    if args.device == "mps":
        gs.tools.run_in_another_thread(fn=run, args=(env, agent, num_episodes))
        env.scene.viewer.start()
    else:
        run(env, agent, num_episodes)

def run(env, agent, num_episodes):
    batch_size = args.batch_size if args.batch_size else 64 * args.num_envs
    rewards_stats, dones_stats, episode_stats = [], [], []
    all_states, all_actions, all_rewards, all_dones = [], [], [], []

    # Setup interactive plotting
    plt.ion()  # Turn on interactive mode
    figure, axis = plt.subplots(1, 2, figsize=(12, 5))
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = torch.zeros(env.num_envs).to(args.device)
        total_done = torch.zeros(env.num_envs).to(args.device) 
        done_array = torch.tensor([False] * env.num_envs).to(args.device)
        states, actions, rewards, dones = [], [], [], []
    
        for step in range(10): # Number of actions per episode
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            
            state = next_state
            total_reward += reward
            done_array = torch.logical_or(done_array, done)
            
            if done_array.all():
                break
            
            if step==5 and all(torch.equal(action, actions[0]) for action in actions) :
                print("Network collapsed")
                sys.exit(1)
                
        # Collect multiple episodes before training
        all_states.extend(states)
        all_actions.extend(actions)
        all_rewards.extend(rewards)
        all_dones.extend(dones)
        
        # Train every 3 episodes
        if (episode + 1) % 3 == 0 and len(all_states) > 0:
            agent.train(all_states, all_actions, all_rewards, all_dones)
            all_states, all_actions, all_rewards, all_dones = [], [], [], []

        # Store episode statistics
        episode_stats.append(episode)
        rewards_stats.append(torch.sum(total_reward.cpu())/(env.num_envs*5))
        done = torch.sum(done_array.cpu())/env.num_envs*100
        dones_stats.append(done)
        print(f"Episode {episode}, Total Reward: {total_reward}")
        
        # Update graph every 5 episodes
        if episode % 5 == 0 and episode > 0:
            # Clear previous plots
            axis[0].clear()
            axis[1].clear()
            
            # Plot rewards
            axis[0].plot(episode_stats, rewards_stats, color='b', label='Reward')
            axis[0].set_title('Rewards')
            axis[0].set_xlabel('Episode')
            axis[0].set_ylabel('Reward')
            axis[0].legend()
            axis[0].grid(True, alpha=0.3)
            
            # Plot dones
            axis[1].plot(episode_stats, dones_stats, color='r', label='Done %')
            axis[1].set_title('Dones (%)')
            axis[1].set_xlabel('Episode')
            axis[1].set_ylabel('Done')
            axis[1].legend()
            axis[1].grid(True, alpha=0.3)
            
            # Set a common title for the entire figure
            plt.suptitle(f"Episode {episode} - LR: {lr}, Gamma: {gamma}, Clip Epsilon: {clip_epsilon}, Layers: {num_layers}, Hidden Dim: {hidden_dim}")
            
            # Update display
            plt.tight_layout()
            plt.pause(0.01)  # Small pause to update display
            
            # Save checkpoint
            agent.save_checkpoint()

    # Turn off interactive mode and save final plot
    plt.ioff()
    axis[0].clear()
    axis[1].clear()
    
    axis[0].plot(episode_stats, rewards_stats, color='b', label='Reward')
    axis[0].set_title('Final Rewards')
    axis[0].set_xlabel('Episode')
    axis[0].set_ylabel('Reward')
    axis[0].legend()
    axis[0].grid(True, alpha=0.3)
    
    # Plot Dones
    axis[1].plot(episode_stats, dones_stats, color='r', label='Done %')
    axis[1].set_title('Final Dones (%)')
    axis[1].set_xlabel('Episode')
    axis[1].set_ylabel('Done')
    axis[1].legend()
    axis[1].grid(True, alpha=0.3)
    
    plt.suptitle(f"Final Results - LR: {lr}, Gamma: {gamma}, Clip Epsilon: {clip_epsilon}, Layers: {num_layers}, Hidden Dim: {hidden_dim}")

    # Save final plot
    ts = time.time()
    timestamp = datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
    plt.savefig("/home/devtex/Documents/Genesis/graphs/" + timestamp + ".png", dpi=300, bbox_inches='tight')
    plt.show()

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False, help="Enable visualization") 
    parser.add_argument("-l", "--load_path", action="store_const", const="default", default=None, help="Load model from default checkpoint path") 
    parser.add_argument("-n", "--num_envs", type=int, default=1, help="Number of environments to create") 
    parser.add_argument("-b", "--batch_size", type=int, default=None, help="Batch size for training")
    parser.add_argument("-t", "--task", type=str, default="GraspFixedBlock", help="Task to train on")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="device: cpu or cuda:x or mps for macos")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parser()
    train_ppo(args, lr, gamma, clip_epsilon, num_layers, hidden_dim)
