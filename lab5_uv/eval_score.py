import torch
import torch.nn as nn
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import argparse
from collections import deque

# 1. 統一版本的 DQN (支援 Task 1 & 2)
class DQN(nn.Module):
    def __init__(self, input_dim, num_actions, is_cnn=False):
        super(DQN, self).__init__()
        self.is_cnn = is_cnn
        if self.is_cnn:
            self.network = nn.Sequential(
                nn.Conv2d(input_dim, 32, kernel_size=8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
                nn.Linear(512, num_actions)
            )
        else:
            self.network = nn.Sequential(
                nn.Linear(input_dim, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, num_actions)
            )

    def forward(self, x):
        if self.is_cnn:
            x = x / 255.0
        return self.network(x)

class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame.copy())
        return np.stack(self.frames, axis=0)

def evaluate_fast(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    is_cnn = (args.task in [2, 3])
    env_name = "CartPole-v1" if args.task == 1 else "ALE/Pong-v5"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 移除 render_mode，因為我們不需要畫圖，這樣跑最快！
    env = gym.make(env_name)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    preprocessor = AtariPreprocessor()
    num_actions = env.action_space.n

    model = DQN(4, num_actions, is_cnn=is_cnn).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    all_rewards = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        state = obs if args.task == 1 else preprocessor.reset(obs)
        
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_obs if args.task == 1 else preprocessor.step(next_obs)

        all_rewards.append(total_reward)
        # 完美對齊講義 Figure 4 的輸出格式
        print(f"environment steps: {args.env_steps}, seed: {ep}, eval reward: {total_reward}")

    print(f"Average Reward: {np.mean(all_rewards):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20) # 講義規定要測 20 局
    parser.add_argument("--env-steps", type=str, default="Best", help="純粹用來印在截圖上的步數文字")
    parser.add_argument("--seed", type=int, default=0) # 講義截圖的 Seed 是從 0 開始
    args = parser.parse_args()
    evaluate_fast(args)