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
import numpy as np

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
    # Add start time tracking
    start_time = time.time()
    
    batch_size = args.batch_size if args.batch_size else 64 * args.num_envs
    rewards_stats, dones_stats, episode_stats = [], [], []
    all_states, all_actions, all_rewards, all_dones = [], [], [], []

    # Add variance tracking
    window_size = 10  # Rolling window for variance calculation

    # Setup interactive plotting
    plt.ion()  # Turn on interactive mode
    figure, axis = plt.subplots(1, 2, figsize=(12, 5))
    
    # Save hyperparameters once at the beginning
    os.makedirs("graphs", exist_ok=True)
    ts_start = time.time()
    timestamp_start = datetime.fromtimestamp(ts_start).strftime('%Y-%m-%d_%H:%M:%S')
    hyperparams_filename = f"graphs/hyperparameters_{timestamp_start}.txt"
    
    # Create the main data file for all episodes
    data_filename = f"graphs/raw_episode_data_{timestamp_start}.txt"
    
    with open(hyperparams_filename, 'w') as f:
        f.write("=== PPO Training Hyperparameters ===\n")
        f.write(f"Training Started: {timestamp_start}\n\n")
        f.write(f"Learning Rate: {lr}\n")
        f.write(f"Gamma: {gamma}\n")
        f.write(f"Clip Epsilon: {clip_epsilon}\n")
        f.write(f"Number of Layers: {num_layers}\n")
        f.write(f"Hidden Dimension: {hidden_dim}\n")
        f.write(f"Number of Episodes: {num_episodes}\n")
        f.write(f"Number of Environments: {args.num_envs}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Task: {args.task}\n")
        f.write(f"Device: {args.device}\n")
    
    # Initialize the data file with header
    with open(data_filename, 'w') as f:
        f.write("=== PPO Raw Episode Data ===\n")
        f.write(f"Training Started: {timestamp_start}\n")
        f.write(f"Number of Environments: {args.num_envs}\n\n")
        
        # Write header with format: Rewards env 1, Dones env 1, Rewards env 2, Dones env 2...
        header_parts = []
        for env_idx in range(args.num_envs):
            header_parts.append(f"Rewards env {env_idx+1}")
            header_parts.append(f"Dones env {env_idx+1}")
        f.write(",".join(header_parts) + "\n")
    
    print(f"Hyperparameters saved to: {hyperparams_filename}")
    print(f"Episode data will be saved to: {data_filename}")
    
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
        reward_val = torch.sum(total_reward.cpu())/(env.num_envs*5)
        done_val = torch.sum(done_array.cpu())/env.num_envs*100
        
        rewards_stats.append(reward_val)
        dones_stats.append(done_val)
        
        print(f"Episode {episode}, Total Reward: {total_reward}")
        
        # Append current episode data to the main data file
        with open(data_filename, 'a') as f:
            # Create row with format: Reward env1, Done env1, Reward env2, Done env2...
            row_data = []
            for env_idx in range(args.num_envs):
                # Sum rewards for this environment across all steps in this episode
                episode_reward = sum(reward_tensor[env_idx].item() if hasattr(reward_tensor[env_idx], 'item') else reward_tensor[env_idx] for reward_tensor in rewards)
                # Check if this environment completed the task (any step was done)
                episode_done = any(done_tensor[env_idx].item() if hasattr(done_tensor[env_idx], 'item') else done_tensor[env_idx] for done_tensor in dones)
                
                row_data.append(f"{episode_reward:.6f}")
                row_data.append(f"{int(episode_done)}")
            
            f.write(",".join(row_data) + "\n")
        
        print(f"Episode {episode} data appended to: {data_filename}")
        
        # Update graph every 5 episodes
        if episode % 5 == 0 and episode > 0:
            # Calculate variance/std for error bands
            rewards_std = []
            dones_std = []
            
            for i in range(len(episode_stats)):
                # Get window around current episode
                start_idx = max(0, i - window_size//2)
                end_idx = min(len(rewards_stats), i + window_size//2 + 1)
                
                if end_idx - start_idx > 1:  # Need at least 2 points for std
                    reward_window = rewards_stats[start_idx:end_idx]
                    done_window = dones_stats[start_idx:end_idx]
                    
                    rewards_std.append(torch.tensor(reward_window).std().item())
                    dones_std.append(torch.tensor(done_window).std().item())
                else:
                    rewards_std.append(0)
                    dones_std.append(0)
            
            # Convert to numpy for easier manipulation
            episodes_np = np.array(episode_stats)
            rewards_np = np.array(rewards_stats)
            rewards_std_np = np.array(rewards_std)
            dones_np = np.array(dones_stats)
            dones_std_np = np.array(dones_std)
            
            # Clear previous plots
            axis[0].clear()
            axis[1].clear()
            
            # Plot rewards with filled variance bands
            axis[0].plot(episodes_np, rewards_np, color='blue', linewidth=2, label='Rewards')
            axis[0].fill_between(episodes_np, 
                               rewards_np - rewards_std_np, 
                               rewards_np + rewards_std_np, 
                               color='blue', alpha=0.3, label='±Sigma')
            axis[0].set_title('Rewards with Variance Band')
            axis[0].set_xlabel('Episode')
            axis[0].set_ylabel('Reward')
            axis[0].legend()
            axis[0].grid(True, alpha=0.3)
            
            # Plot dones with filled variance bands
            axis[1].plot(episodes_np, dones_np, color='red', linewidth=2, label='Done %')
            axis[1].fill_between(episodes_np, 
                               dones_np - dones_std_np, 
                               dones_np + dones_std_np, 
                               color='red', alpha=0.3, label='±Sigma')
            axis[1].set_title('Dones (%) with Variance Band')
            axis[1].set_xlabel('Episode')
            axis[1].set_ylabel('Done %')
            axis[1].legend()
            axis[1].grid(True, alpha=0.3)
            
            # Set a common title for the entire figure with running time
            current_time = time.time()
            elapsed_time = current_time - start_time
            elapsed_minutes = int(elapsed_time // 60)
            elapsed_seconds = int(elapsed_time % 60)
            plt.suptitle(f"Episode {episode} - Runtime: {elapsed_minutes}m {elapsed_seconds}s - LR: {lr}, Gamma: {gamma}, Clip Epsilon: {clip_epsilon}, Layers: {num_layers}, Hidden Dim: {hidden_dim}")
            
            # Update display
            plt.tight_layout()
            plt.pause(0.01)  # Small pause to update display
            
            # Save checkpoint
            agent.save_checkpoint()

    # Turn off interactive mode and create final plot with variance bands
    plt.ioff()
    
    # Calculate final running time
    final_time = time.time()
    total_elapsed = final_time - start_time
    total_minutes = int(total_elapsed // 60)
    total_seconds = int(total_elapsed % 60)
    
    # Calculate final variance
    final_rewards_std = []
    final_dones_std = []
    
    for i in range(len(episode_stats)):
        start_idx = max(0, i - window_size//2)
        end_idx = min(len(rewards_stats), i + window_size//2 + 1)
        
        if end_idx - start_idx > 1:
            reward_window = rewards_stats[start_idx:end_idx]
            done_window = dones_stats[start_idx:end_idx]
            
            final_rewards_std.append(torch.tensor(reward_window).std().item())
            final_dones_std.append(torch.tensor(done_window).std().item())
        else:
            final_rewards_std.append(0)
            final_dones_std.append(0)
    
    # Convert to numpy for final plots
    episodes_np = np.array(episode_stats)
    rewards_np = np.array(rewards_stats)
    final_rewards_std_np = np.array(final_rewards_std)
    dones_np = np.array(dones_stats)
    final_dones_std_np = np.array(final_dones_std)
    
    axis[0].clear()
    axis[1].clear()
    
    # Final plots with variance bands
    axis[0].plot(episodes_np, rewards_np, color='blue', linewidth=2, label='Rewards')
    axis[0].fill_between(episodes_np, 
                       rewards_np - final_rewards_std_np, 
                       rewards_np + final_rewards_std_np, 
                       color='blue', alpha=0.3, label='±1 STD')
    axis[0].set_title('Final Rewards with Variance Band')
    axis[0].set_xlabel('Episode')
    axis[0].set_ylabel('Reward')
    axis[0].legend()
    axis[0].grid(True, alpha=0.3)
    
    # Plot Dones with variance bands
    axis[1].plot(episodes_np, dones_np, color='red', linewidth=2, label='Done %')
    axis[1].fill_between(episodes_np, 
                       dones_np - final_dones_std_np, 
                       dones_np + final_dones_std_np, 
                       color='red', alpha=0.3, label='±1 STD')
    axis[1].set_title('Final Dones (%) with Variance Band')
    axis[1].set_xlabel('Episode')
    axis[1].set_ylabel('Done %')
    axis[1].legend()
    axis[1].grid(True, alpha=0.3)
    
    # Final title with total running time
    plt.suptitle(f"Final Results - Total Runtime: {total_minutes}m {total_seconds}s - LR: {lr}, Gamma: {gamma}, Clip Epsilon: {clip_epsilon}, Layers: {num_layers}, Hidden Dim: {hidden_dim}")

    # Save final plot
    ts = time.time()
    timestamp = datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
    os.makedirs("graphs", exist_ok=True)
    plt.savefig(f"graphs/{timestamp}.png", dpi=300, bbox_inches='tight')
    
    # Save final comprehensive training data
    final_data_filename = f"graphs/final_raw_data_{timestamp}.txt"
    with open(final_data_filename, 'w') as f:
        f.write("=== FINAL PPO Raw Training Data ===\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total Episodes: {num_episodes}\n")
        f.write(f"Total Runtime: {total_minutes}m {total_seconds}s\n")
        f.write(f"Training Efficiency: {total_elapsed/num_episodes:.2f} seconds per episode\n\n")
        
        f.write("=== Complete Raw Episode Data ===\n")
        f.write("Episode,Reward,CompletionRate(%)\n")
        for i, ep in enumerate(episode_stats):
            f.write(f"{ep},{rewards_stats[i]:.4f},{dones_stats[i]:.2f}\n")
    
    print(f"Final raw training data saved to: {final_data_filename}")
    plt.show()

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False, help="Enable visualization") 
    parser.add_argument("-l", "--load_path", action="store_const", const="default", default=None, help="Load model from default checkpoint path") 
    parser.add_argument("-n", "--num_envs", type=int, default=1, help="Number of environments to create") 
    parser.add_argument("-b", "--batch_size", type=int, default=None, help="Batch size for training")
    parser.add_argument("-t", "--task", type=str, default="PickPlaceRandomBlock", help="Task to train on")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="device: cpu or cuda:x or mps for macos")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parser()
    train_ppo(args, lr, gamma, clip_epsilon, num_layers, hidden_dim)
