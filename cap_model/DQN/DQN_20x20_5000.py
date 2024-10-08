import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse

import gymnasium as gym
import torch as th
import numpy as np
import random
import math
import wandb
from envs.GymMoreRedBalls import GymMoreRedBalls
from envs.wrapper import MaxStepsWrapper
from envs.wrapper import FullyCustom
# Hyperparameters
parser = argparse.ArgumentParser(description='DQN Training for GymMoreRedBalls')
#parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')

parser.add_argument('--batch-size', type=int, default=50, metavar='B', help='Batch size')
parser.add_argument('--learning-rate', type=float, default=0.0001, metavar='LR', help='Learning rate')
parser.add_argument('--epsilon-start', type=float, default=1, metavar='ES', help='Start of epsilon')
parser.add_argument('--epsilon-end', type=float, default=0.2, metavar='EE', help='End of epsilon')
parser.add_argument('--epsilon-decay', type=int, default=200, metavar='ED', help='Epsilon decay')
parser.add_argument('--replay-memory-size', type=int, default=10000, metavar='RMS', help='Replay memory size')
parser.add_argument('--gamma', type=float, default=0.9, metavar='G', help='Discount factor')
parser.add_argument('--target-update-iter', type=int, default=200, metavar='TUI', help='Target network update interval')
parser.add_argument('--max-steps', type=int, default=2000, metavar='MS', help='Maximum number of steps per episode')
parser.add_argument('--episodes', type=int, default=1000, metavar='E', help='Total number of episodes')
parser.add_argument('--env', type=str, default='GymMoreRedBalls-v0', help='Gym environment')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--render', action='store_true', default=True, help='Render environment')
parser.add_argument('--wandb-project', type=str, default='3ball_CAP', help='WandB project name')
parser.add_argument('--wandb-entity', type=str, default='hails', help='WandB entity name')
parser.add_argument('--run-name', type=str, default='DQN_20x20_5000', help='WandB run name')

args = parser.parse_args()


# Initialize wandb and log hyperparameters
wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.run_name,config={
    "batch_size": args.batch_size,
    "learning_rate": args.learning_rate,
    "epsilon_start": args.epsilon_start,
    "epsilon_end": args.epsilon_end,
    "epsilon_decay": args.epsilon_decay,
    "replay_memory_size": args.replay_memory_size,
    "gamma": args.gamma,
    "target_update_iter": args.target_update_iter,
    "max_steps": args.max_steps,
})

# Create and wrap the environment
#env = gym.make(args.env, render_mode='human' if args.render else None)
env = GymMoreRedBalls(room_size=20, render_mode="rgb_array")
env = FullyCustom(env, args.max_steps)
env = MaxStepsWrapper(env, args.max_steps)
device = th.device("cuda" if th.cuda.is_available() else "cpu")
n_action = 3

if isinstance(env.observation_space, gym.spaces.Dict):
    n_state = np.prod(env.observation_space['image'].shape)
else:
    n_state = np.prod(env.observation_space.shape)

hidden = 32

class Net(th.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = th.nn.Linear(n_state, hidden)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = th.nn.Linear(hidden, n_action)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = th.nn.functional.relu(x)
        out = self.out(x)
        return out

class ReplayMemory:
    def __init__(self):
        self.memory_size = args.replay_memory_size
        self.memory = []
        self.cur = 0

    def size(self):
        return len(self.memory)

    def store_transition(self, trans):
        if len(self.memory) < self.memory_size:
            self.memory.append(trans)
        else:
            self.memory[self.cur] = trans
            self.cur = (self.cur + 1) % self.memory_size

    def sample(self):
        if len(self.memory) < args.batch_size:
            return -1
        sam = np.random.choice(len(self.memory), args.batch_size)
        batch = [self.memory[i] for i in sam]
        return np.array(batch, dtype=object)

class DQN:
    def __init__(self):
        self.eval_q_net, self.target_q_net = Net().to(device), Net().to(device)
        self.replay_mem = ReplayMemory()
        self.iter_num = 0
        self.optimizer = th.optim.Adam(self.eval_q_net.parameters(), lr=args.learning_rate)
        self.loss_fn = th.nn.MSELoss().to(device)
        self.loss_history = []

    def select_action(self, state):
        global steps_done
        sample = random.random()
        eps_threshold = args.epsilon_end + (args.epsilon_start - args.epsilon_end) * math.exp(-1. * steps_done / args.epsilon_decay)
        steps_done += 1
        if sample > eps_threshold:
            with th.no_grad():
                state = state.unsqueeze(0)  # 2차원 텐서로 변환
                return self.eval_q_net(state.to(device)).max(1)[1].view(1, 1)
        else:
            return th.tensor([[random.randrange(n_action)]], device=device, dtype=th.long)

    def learn(self):
        if self.iter_num % args.target_update_iter == 0:
            self.target_q_net.load_state_dict(self.eval_q_net.state_dict())
        self.iter_num += 1

        batch = self.replay_mem.sample()
        if len(batch) == 0:
            return

        b_s = th.FloatTensor(np.vstack(batch[:, 0])).to(device)
        b_a = th.LongTensor(batch[:, 1].astype(int).tolist()).to(device)
        b_r = th.FloatTensor(np.vstack(batch[:, 2])).to(device)
        b_s_ = th.FloatTensor(np.vstack(batch[:, 3])).to(device)
        b_d = th.FloatTensor(np.vstack(batch[:, 4])).to(device)

        q_target = th.zeros((args.batch_size, 1)).to(device)
        q_eval = self.eval_q_net(b_s)
        q_eval = th.gather(q_eval, dim=1, index=th.unsqueeze(b_a, 1))
        q_next = self.target_q_net(b_s_).detach()
        for i in range(b_d.shape[0]):
            if int(b_d[i].tolist()[0]) == 0:
                q_target[i] = b_r[i] + args.gamma * th.unsqueeze(th.max(q_next[i], 0)[0], 0)
            else:
                q_target[i] = b_r[i]
        td_error = self.loss_fn(q_eval, q_target)

        self.optimizer.zero_grad()
        td_error.backward()
        self.optimizer.step()

        self.loss_history.append(td_error.item())

dqn = DQN()
steps_done = 0

#def preprocess_state(obs):
#    if isinstance(obs, dict):
#        return obs['image'].flatten()
#    else:
#        return obs.flatten()

for episode in range(args.episodes):
    obs = env.reset()
    #s = preprocess_state(obs[0])  # Initial state
    s = obs
    step = 0
    r = 0.0
    total_reward = 0.0
    episode_loss = 0
    episode_value = 0
    done = False
    while not done:
        step += 1
        a = dqn.select_action(th.FloatTensor(s).to(device))
        obs, r, terminated, truncated, info = env.step(a.item())  # 액션을 넘겨줄 때 item() 메서드 사용
        done = terminated or truncated
        #s_ = preprocess_state(obs)
        s_=obs
        transition = [s.tolist(), a.item(), [r], s_.tolist(), [done]]
        dqn.replay_mem.store_transition(transition)
        total_reward += r
        s = s_
        print("step:", step, "reward:", r)
        print("episode : ", episode, "reward : ", r)
        if dqn.replay_mem.size() > args.batch_size:
            dqn.learn()

        if done:
            #wandb.log({"steps": step})
            break

    episode_loss = np.mean(dqn.loss_history[-step:]) if step > 0 else 0
    avg_q_value = th.mean(dqn.eval_q_net(th.FloatTensor([s]).to(device))).item()

    wandb.log({
         "episode": episode,
         "reward": total_reward,
         "average_loss": episode_loss,
        "avg_q_value": avg_q_value,
        "steps": step
     },step=episode)

th.save(dqn.eval_q_net.state_dict(), "dqn_eval_q_net_min.pth")
th.save(dqn.target_q_net.state_dict(), "dqn_target_q_net_min.pth")

env.close()
#wandb.finish()
