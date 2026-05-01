# Spring 2026, 535518 Deep Learning
# Lab5: Value-based RL
# Contributors: Kai-Siang Ma and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time

gym.register_envs(ale_py)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    """
        Design the architecture of your deep Q network
        - Input size is the same as the state dimension; the output size is the same as the number of actions
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
    """
    def __init__(self, num_actions, input_dim, task = 2):
        super(DQN, self).__init__()
        # An example: 
        #self.network = nn.Sequential(
        #    nn.Linear(input_dim, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, num_actions)
        #)       
        ########## YOUR CODE HERE (5~10 lines) ##########
        self.task = task
        if task == 2 or task == 3:
            self.network = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 512),
                nn.ReLU(),
                nn.Linear(512, num_actions)
            )
        else:
            self.network = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, num_actions)
            )
        ########## END OF YOUR CODE ##########

    def forward(self, x):
        if self.task == 2 or self.task == 3:
            x = x / 255.0  # Normalize pixel values for Pong
        return self.network(x)


class AtariPreprocessor:
    """
        Preprocesing the state input of DQN for Atari
    """    
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


class PrioritizedReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """ 
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, transition, error=None):
        ########## YOUR CODE HERE (for Task 3) ########## 
        max_p = np.max(self.priorities) if self.buffer else 1.0
        priority = (abs(error) + 1e-5) ** self.alpha if error is not None else max_p
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity
        ########## END OF YOUR CODE (for Task 3) ########## 
        return 
    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ########## 
        current_size = len(self.buffer)
        priorities = self.priorities[:current_size]
        
        probs = priorities / priorities.sum()
        
        indices = np.random.choice(current_size, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total = current_size
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        return samples, indices, np.array(weights, dtype=np.float32)
        ########## END OF YOUR CODE (for Task 3) ########## 
        return
    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ########## 
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha
        ########## END OF YOUR CODE (for Task 3) ########## 
        return
        

class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):
        self.task = args.task
        self.pong = (self.task in [2, 3])
        self.terminal_output = args.terminal_output
        
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        self.preprocessor = AtariPreprocessor()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        input_dim = 4
        self.q_net = DQN(self.num_actions, input_dim, task=self.task).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(self.num_actions, input_dim, task=self.task).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.env_count = 0
        self.train_count = 0
        self.best_reward = 0  # Initilized to 0 for CartPole and to -21 for Pong
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.n_step = args.n_step
        self.memory = PrioritizedReplayBuffer(capacity=args.memory_size, alpha=args.per_alpha, beta=args.per_beta)
        self.n_step_buffer = deque(maxlen=self.n_step)
        self.episodes = args.episodes
        
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.log_buffer = deque(maxlen=20) 
        self.reached_19 = False
        self.post_19_counter = -1
        self.student_id = "411410010"

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self):
        episodes = self.episodes
        for ep in range(episodes):
            obs, _ = self.env.reset()

            state = obs if self.task == 1 else self.preprocessor.reset(obs)
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                next_state = next_obs if self.task == 1 else self.preprocessor.step(next_obs)
                # self.memory.append((state, action, reward, next_state, done))

                self.n_step_buffer.append((state, action, reward, next_state, done))
                
                if len(self.n_step_buffer) == self.n_step:
                    n_step_reward = sum([t[2] * (self.gamma ** i) for i, t in enumerate(self.n_step_buffer)])
                    n_step_state, n_step_action = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
                    n_step_next_state, n_step_done = self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]

                    self.memory.add((n_step_state, n_step_action, n_step_reward, n_step_next_state, n_step_done))
                    
                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1
                
                milestones = [600000, 1000000, 1500000, 2000000, 2500000]
                if self.env_count in milestones:
                    path = os.path.join(self.save_dir, f"LAB5_{self.student_id}_task3_{self.env_count}.pt")
                    torch.save(self.q_net.state_dict(), path)
                    print(f"\n [Milestone] achieve {self.env_count} steps, Model saved to: {path}\n")

                if self.env_count % 1000 == 0:                 
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })
                    
            self.n_step_buffer.clear()
                    ########## YOUR CODE HERE  ##########
                    # Add additional wandb logs for debugging if needed 
                    
                    ########## END OF YOUR CODE ##########   
            if self.terminal_output:
                print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon
            })
            ########## YOUR CODE HERE  ##########
            # Add additional wandb logs for debugging if needed 
            
            ########## END OF YOUR CODE ##########  
            if ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                
                log_str = f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}"
                self.log_buffer.append(log_str)
                
                if eval_reward >= 19.0 and not self.reached_19:
                    self.reached_19 = True
                    hit_msg = f"\n reach score 19 (Env Steps: {self.env_count}) save at LAB5_{self.student_id}_task3_best.pt"
                    self.log_buffer.append(hit_msg)
                    print(hit_msg)
                    
                    torch.save(self.q_net.state_dict(), os.path.join(self.save_dir, f"LAB5_{self.student_id}_task3_best.pt"))
                    
                    self.post_19_counter = 2  

                # 倒數計時器結束，寫入 txt 檔案
                elif self.reached_19 and self.post_19_counter > 0:
                    self.post_19_counter -= 1
                    if self.post_19_counter == 0:
                        txt_path = os.path.join(self.save_dir, "score19.txt")
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write("\n".join(self.log_buffer))
                        print(f"score 19 log added to {txt_path}\n")
                
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })

    def evaluate(self):
        obs, _ = self.test_env.reset()
        state = obs if self.task == 1 else self.preprocessor.reset(obs)
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_obs if self.task == 1 else self.preprocessor.step(next_obs)

        return total_reward


    def train(self):

        # if len(self.memory) < self.replay_start_size:
        #     return         
        if len(self.memory.buffer) < self.replay_start_size:
            return
        
        # Decay function for epsilin-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1
       
        ########## YOUR CODE HERE (<5 lines) ##########
        # Sample a mini-batch of (s,a,r,s',done) from the replay buffer
        # batch = random.sample(self.memory, self.batch_size)
        batch, indices, weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
      
            
        ########## END OF YOUR CODE ##########

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        # NOTE: Enable this part after you finish the mini-batch sampling
        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        ########## YOUR CODE HERE (~10 lines) ##########
        # Implement the loss function of DQN and the gradient updates 
        with torch.no_grad():
            # max_next_q_values = self.target_net(next_states).max(1)[0]
            # target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)
            next_actions = self.q_net(next_states).argmax(1).unsqueeze(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            
            gamma_n = self.gamma ** self.n_step
            target_q_values = rewards + gamma_n * next_q_values * (1 - dones)
          
        td_errors = (target_q_values - q_values).abs().cpu().detach().numpy()
        self.memory.update_priorities(indices, td_errors)
            
        # loss = nn.MSELoss()(q_values, target_q_values)
        loss = (weights * nn.MSELoss(reduction='none')(q_values, target_q_values)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
      
        ########## END OF YOUR CODE ##########  

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # NOTE: Enable this part if "loss" is defined
        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")
            wandb.log({
                "Train/Loss": loss.item(),
                "Train/Q_mean": q_values.mean().item(),
                "Train/Q_std": q_values.std().item(),
                "Update Count": self.train_count
            })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=1, choices=[1, 2, 3], help="Choose Task (1: CartPole, 2: Pong, 3: Enhanced Pong)")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--terminal-output", action="store_true", help="Whether to print the logs in the terminal", default=True)
    parser.add_argument("--n-step", type=int, default=3, help="N-step return")
    parser.add_argument("--per-alpha", type=float, default=0.6, help="PER alpha parameter")
    parser.add_argument("--per-beta", type=float, default=0.4, help="PER beta parameter")
    
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str, default="cartpole-run")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.999999)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=50000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=1)
    args = parser.parse_args()

    wandb.login(key="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")
    env_name = "CartPole-v1" if args.task == 1 else "ALE/Pong-v5"
    wandb.init(project="NYCU_DL_lab5", name=args.wandb_run_name, group=f"task_{args.task}", save_code=True)
    agent = DQNAgent(env_name=env_name, args=args)
    agent.run()
    # python dqn.py --task 2 --wandb-run-name task2_v0 --episodes 3000 --memory-size 300000 --batch-size 64 --target-update-frequency 5000
    # python dqn.py --task 2 --wandb-run-name task2_orginparm --episodes 3000
    # python dqn.py --task 3 --wandb-run-name task3_v0_enhance111 --episodes 3000 --n-step 3 --per-alpha 0.6 --per-beta 0.4