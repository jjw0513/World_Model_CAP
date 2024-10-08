import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

# from env import CONTROL_SUITE_ENVS, GYM_ENVS, Env, EnvBatcher
from memory import ExperienceReplay_nosym as ExperienceReplay
from models import ActorModel, Encoder, ObservationModel, RewardModel, TransitionModel, ValueModel, bottle
from planner import MPCPlanner
from Dreamer.utils import FreezeParameters, imagine_ahead, lambda_return, lineplot, write_video
from envs.GymMoreRedBalls import GymMoreRedBalls
from envs.wrapper import MaxStepsWrapper
from envs.wrapper import FullyCustom
from envs.wrapper import EnvBatcher
from envs.wrapper import ActionSpaceWrapper

import wandb
# Hyperparameters
parser = argparse.ArgumentParser(description='PlaNet or Dreamer')
parser.add_argument('--algo', type=str, default='dreamer', help='planet or dreamer')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument(
    '--env',
    type=str,
    default='GymMoreRedBalls-v0',
    # choices=GYM_ENVS + CONTROL_SUITE_ENVS,
    help='Gym/Control Suite environment',
)
parser.add_argument('--wandb-project', type=str, default='3ball_CAP', help='WandB project name')
parser.add_argument('--wandb-entity', type=str, default='hails', help='WandB entity name')
parser.add_argument('--run-name', type=str, default='Dreamer_20x20_5000', help='WandB run name')

parser.add_argument('--symbolic-env', action='store_true', help='Symbolic features')

parser.add_argument('--max-episode-length', type=int, default=1000, metavar='T', help='Max episode length')
#parser.add_argument('--max-episode-length', type=int, default=5, metavar='T', help='Max episode length')

parser.add_argument(
    '--experience-size', type=int, default=1000000, metavar='D', help='Experience replay size'
)  # Original implementation has an unlimited buffer size, but 1 million is the max experience collected anyway
parser.add_argument(
    '--cnn-activation-function',
    type=str,
    default='relu',
    choices=dir(F),
    help='Model activation function for a convolution layer',
)
parser.add_argument(
    '--dense-activation-function',
    type=str,
    default='elu',
    choices=dir(F),
    help='Model activation function a dense layer',
)
parser.add_argument(
    '--embedding-size', type=int, default=1024, metavar='E', help='Observation embedding size'
)  # Note that the default encoder for visual observations outputs a 1024D vector; for other embedding sizes an additional fully-connected layer is used
parser.add_argument('--hidden-size', type=int, default=200, metavar='H', help='Hidden size')
parser.add_argument('--belief-size', type=int, default=200, metavar='H', help='Belief/hidden size')
parser.add_argument('--state-size', type=int, default=30, metavar='Z', help='State/latent size')
parser.add_argument('--action-repeat', type=int, default=1, metavar='R', help='Action repeat')
parser.add_argument('--action-noise', type=float, default=0.3, metavar='ε', help='Action noise')

parser.add_argument('--episodes', type=int, default=1000, metavar='E', help='Total number of episodes')


parser.add_argument('--seed-episodes', type=int, default=5, metavar='S', help='Seed episodes')
parser.add_argument('--collect-interval', type=int, default=100, metavar='C', help='Collect interval')
parser.add_argument('--batch-size', type=int, default=50, metavar='B', help='Batch size')
parser.add_argument('--chunk-size', type=int, default=50, metavar='L', help='Chunk size')
parser.add_argument('--max-steps', type=int, default=5000, metavar='MS', help='Maximum number of steps per episode')
parser.add_argument(
    '--worldmodel-LogProbLoss',
    action='store_true',
    help='use LogProb loss for observation_model and reward_model training',
)
parser.add_argument(
    '--overshooting-distance',
    type=int,
    default=50,
    metavar='D',
    help='Latent overshooting distance/latent overshooting weight for t = 1',
)
parser.add_argument(
    '--overshooting-kl-beta',
    type=float,
    default=0,
    metavar='β>1',
    help='Latent overshooting KL weight for t > 1 (0 to disable)',
)
parser.add_argument(
    '--overshooting-reward-scale',
    type=float,
    default=0,
    metavar='R>1',
    help='Latent overshooting reward prediction weight for t > 1 (0 to disable)',
)
parser.add_argument('--global-kl-beta', type=float, default=0, metavar='βg', help='Global KL weight (0 to disable)')
parser.add_argument('--free-nats', type=float, default=3, metavar='F', help='Free nats')
parser.add_argument('--bit-depth', type=int, default=5, metavar='B', help='Image bit depth (quantisation)')
parser.add_argument('--model_learning-rate', type=float, default=1e-3, metavar='α', help='Learning rate')
parser.add_argument('--actor_learning-rate', type=float, default=8e-5, metavar='α', help='Learning rate')
parser.add_argument('--value_learning-rate', type=float, default=8e-5, metavar='α', help='Learning rate')
parser.add_argument(
    '--learning-rate-schedule',
    type=int,
    default=0,
    metavar='αS',
    help='Linear learning rate schedule (optimisation steps from 0 to final learning rate; 0 to disable)',
)
parser.add_argument('--adam-epsilon', type=float, default=1e-7, metavar='ε', help='Adam optimizer epsilon value')
# Note that original has a linear learning rate decay, but it seems unlikely that this makes a significant difference
parser.add_argument('--grad-clip-norm', type=float, default=100.0, metavar='C', help='Gradient clipping norm')
parser.add_argument('--planning-horizon', type=int, default=15, metavar='H', help='Planning horizon distance')
parser.add_argument('--discount', type=float, default=0.99, metavar='H', help='Planning horizon distance')
parser.add_argument('--disclam', type=float, default=0.95, metavar='H', help='discount rate to compute return')
parser.add_argument('--optimisation-iters', type=int, default=10, metavar='I', help='Planning optimisation iterations')
parser.add_argument('--candidates', type=int, default=1000, metavar='J', help='Candidate samples per iteration')
parser.add_argument('--top-candidates', type=int, default=100, metavar='K', help='Number of top candidates to fit')
parser.add_argument('--test', action='store_true', help='Test only')


parser.add_argument('--test-interval', type=int, default=25, metavar='I', help='Test interval (episodes)')
#parser.add_argument('--test-interval', type=int, default=10, metavar='I', help='Test interval (episodes)')

parser.add_argument('--test-episodes', type=int, default=5, metavar='E', help='Number of test episodes')

parser.add_argument('--checkpoint-interval', type=int, default=50, metavar='I', help='Checkpoint interval (episodes)')
parser.add_argument('--checkpoint-experience', action='store_true', help='Checkpoint experience replay')
parser.add_argument('--models', type=str, default='', metavar='M', help='Load model checkpoint')
parser.add_argument('--experience-replay', type=str, default='', metavar='ER', help='Load experience replay')
parser.add_argument('--render', action='store_true', help='Render environment')
args = parser.parse_args()
args.overshooting_distance = min(
    args.chunk_size, args.overshooting_distance
)  # Overshooting distance cannot be greater than chunk size
print(' ' * 26 + 'Options')
for k, v in vars(args).items():
    print(' ' * 26 + k + ': ' + str(v))


# Setup
results_dir = os.path.join('results', '{}_{}'.format(args.env, args.id))
os.makedirs(results_dir, exist_ok=True)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available() and not args.disable_cuda:
    print("using CUDA")
    args.device = torch.device('cuda')
    torch.cuda.manual_seed(args.seed)
else:
    print("using CPU")
    args.device = torch.device('cpu')
metrics = {
    'steps': [],
    'episodes': [],
    'train_rewards': [],
    'test_episodes': [],
    'test_rewards': [],
    'observation_loss': [],
    'reward_loss': [],
    'kl_loss': [],
    'actor_loss': [],
    'value_loss': [],
}

summary_name = results_dir + "/{}_{}_log"
writer = SummaryWriter(summary_name.format(args.env, args.id))
print("writer is ready")




wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.run_name, config={
    "batch_size": args.batch_size,
    "overshooting_distance" : args.overshooting_distance,
    "episodes" : args.episodes,
    "chunk_size" : args.chunk_size,
    "planning_horizon" : args.planning_horizon,
    "planning_discount" : args.discount,
    "max_steps": args.max_steps,
})


#env = gym.make(args.env, render_mode='human' if args.render else None)
env = GymMoreRedBalls(room_size=20)
env = ActionSpaceWrapper(env, args.max_steps,new_action_space=3)
env = FullyCustom(env, args.max_steps)

env = MaxStepsWrapper(env, args.max_steps, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth, new_action_space=3)


if 'GymMoreRedBalls' in args.env:
    action_space_n_valid = 3


print("environment is loaded") #env.env.env.env.max_steps : 5000
if args.experience_replay != '' and os.path.exists(args.experience_replay):
    D = torch.load(args.experience_replay)
    metrics['steps'], metrics['episodes'] = [D.steps] * D.episodes, list(range(1, D.episodes + 1))
elif not args.test:
    D = ExperienceReplay(
        args.experience_size, env.observation_space['image'].shape, env.action_space.n, args.bit_depth, args.device
    )
    # Initialise dataset D with S random seed episodes
    for s in range(1, args.seed_episodes + 1):
        done, t = False, 0
        #observation, info = env.reset()
        observation = env.reset()
        #observation = observation.reshape(-1, np.prod(env.observation_space.shape)) # in case you need to do flatten
        while not done:
            action = env.action_space.sample() # TODO this is also changed to use the method of our env not their
            next_observation, reward, done, truncated, _ = env.step(action)

            if done or truncated is True :
                print("break is better")

            if done or truncated:
                done = True

            D.append(observation, action, reward, done)
            observation = next_observation
            t += 1
        metrics['steps'].append(t * args.action_repeat + (0 if len(metrics['steps']) == 0 else metrics['steps'][-1]))
        metrics['episodes'].append(s)
print("experience replay buffer is ready")


# Initialise model parameters randomly
transition_model = TransitionModel(
    args.belief_size,
    args.state_size,
    #env.action_size,
    int(env.action_space.n),
    args.hidden_size,
    args.embedding_size,
    args.dense_activation_function,
).to(device=args.device)
observation_model = ObservationModel(
    args.symbolic_env,
    #env.observation_size,
    env.observation_space['image'].shape,

    args.belief_size,
    args.state_size,
    args.embedding_size,
    args.cnn_activation_function,
).to(device=args.device)
reward_model = RewardModel(args.belief_size, args.state_size, args.hidden_size, args.dense_activation_function).to(
    device=args.device
)
encoder = Encoder(args.symbolic_env, env.observation_space['image'].shape, args.embedding_size, args.cnn_activation_function).to(
    device=args.device
)
actor_model = ActorModel(
    args.belief_size, args.state_size, args.hidden_size, int(env.action_space.n), args.dense_activation_function
).to(device=args.device)
value_model = ValueModel(args.belief_size, args.state_size, args.hidden_size, args.dense_activation_function).to(
    device=args.device
)
param_list = (
    list(transition_model.parameters())
    + list(observation_model.parameters())
    + list(reward_model.parameters())
    + list(encoder.parameters())
)
value_actor_param_list = list(value_model.parameters()) + list(actor_model.parameters())
params_list = param_list + value_actor_param_list
print("transition, observation, reward, encoder, actor, value models are ready")
model_optimizer = optim.Adam(
    param_list, lr=0 if args.learning_rate_schedule != 0 else args.model_learning_rate, eps=args.adam_epsilon
)
actor_optimizer = optim.Adam(
    actor_model.parameters(),
    lr=0 if args.learning_rate_schedule != 0 else args.actor_learning_rate,
    eps=args.adam_epsilon,
)
value_optimizer = optim.Adam(
    value_model.parameters(),
    lr=0 if args.learning_rate_schedule != 0 else args.value_learning_rate,
    eps=args.adam_epsilon,
)
if args.models != '' and os.path.exists(args.models):
    print("loading pre-trained models")
    model_dicts = torch.load(args.models)
    transition_model.load_state_dict(model_dicts['transition_model'])
    observation_model.load_state_dict(model_dicts['observation_model'])
    reward_model.load_state_dict(model_dicts['reward_model'])
    encoder.load_state_dict(model_dicts['encoder'])
    actor_model.load_state_dict(model_dicts['actor_model'])
    value_model.load_state_dict(model_dicts['value_model'])
    model_optimizer.load_state_dict(model_dicts['model_optimizer'])
if args.algo == "dreamer":
    print("DREAMER")
    planner = actor_model
else:
    print("PLANET")
    planner = MPCPlanner(
        #env.action_size,
        int(env.action_space.n),
        args.planning_horizon,
        args.optimisation_iters,
        args.candidates,
        args.top_candidates,
        transition_model,
        reward_model,
    )
global_prior = Normal(
    torch.zeros(args.batch_size, args.state_size, device=args.device),
    torch.ones(args.batch_size, args.state_size, device=args.device),
)  # Global prior N(0, I)
free_nats = torch.full((1,), args.free_nats, device=args.device)  # Allowed deviation in KL divergence
print("models and planners are ready")


def update_belief_and_act( #agent에게 만들어진 신념과 transition을 기반으로 planning을 시키는 부분
    args, env, planner, transition_model, encoder, belief, posterior_state, action, observation, explore=False
):
    # Infer belief over current state q(s_t|o≤t,a<t) from the history
    # print("action size: ",action.size()) torch.Size([1, 6])

    #belif : ([1,1,200]), obseration : ([1,3,64,64])

    belief, _, _, _, posterior_state, _, _ = transition_model( #주어진 관찰과 이전 action을 사용하여 => 현재 상태에 대한 belief와 사후 상태를 업데이트
        posterior_state, action.unsqueeze(dim=0), belief, encoder(observation).unsqueeze(dim=0)
    )  # Action and observation need extra time dimension
    belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(
        dim=0
    )  # Remove time dimension from belief/state
    if args.algo == "dreamer":
        action = planner.get_action(belief, posterior_state, det=not (explore)) #결정적/탐험적 매개변수에 따라 action 취하는게 달라짐
    else: #그냥 planner에서 짜여진 action을 그대로 뽑음
        action = planner(belief, posterior_state)  # Get action from planner(q(s_t|o≤t,a<t), p)
    if explore: #탐험 모드 시, 가우시간 분포 추가하여 action 다양하게
        action = torch.clamp(
            Normal(action, args.action_noise).rsample(), -1, 1
        )  # Add gaussian exploration noise on top of the sampled action
        # action = action + args.action_noise * torch.randn_like(action)  # Add exploration noise ε ~ p(ε) to the action
    next_observation, reward, done,_,_ = env.step(
        action.cpu() if isinstance(env, EnvBatcher) else action[0].cpu()
    )  # Perform environment step (action repeats handled internally)
    if reward > 0.1 :
        pass
    return belief, posterior_state, action, next_observation, reward, done


# Testing only
if args.test:
    # Set models to eval mode
    transition_model.eval()
    reward_model.eval()
    encoder.eval()
    with torch.no_grad():
        total_reward = 0
        for _ in tqdm(range(args.test_episodes)):
            observation = env.reset()
            belief, posterior_state, action = (
                torch.zeros(1, args.belief_size, device=args.device),
                torch.zeros(1, args.state_size, device=args.device),
                torch.zeros(1, int(env.action_space.n), device=args.device),
            )
            #args.action_repeat로 나누어 반복 횟수를 줄인다.
            #pbar = tqdm(range(args.max_episode_length // args.action_repeat))
            pbar = tqdm(range(args.max_steps // args.action_repeat))
            for t in pbar:                      #update_belief_and_act함수를 호출하여 현재 상태를 업데이트하고,
                                                #다음 행동을 선택한다.
                belief, posterior_state, action, observation, reward, done = update_belief_and_act(
                    args,
                    env,
                    planner,
                    transition_model,
                    encoder,
                    belief,
                    posterior_state,
                    action,
                    observation.to(device=args.device),
                )
                total_reward += reward
                if args.render:
                    env.render()
                if done:
                    pbar.close()
                    break
    print('Average Reward:', total_reward / args.test_episodes)
    env.close()
    quit()

print("tqdm :", tqdm)

# Training (and testing)
for episode in tqdm(    #마지막으로 완료된 에피소드 요소에 +1하여 다음 에피소드부터 시작하도록/ 다음 에피소드부터 최동 에피소드까지
    range(metrics['episodes'][-1] + 1, args.episodes + 1), total=args.episodes, initial=metrics['episodes'][-1] + 1
):
    # Model fitting
    print("epdisode : ", episode)
    losses = []
    #학습할 모델 모듈들을 결합한다
    model_modules = transition_model.modules + encoder.modules + observation_model.modules + reward_model.modules

    print("training loop")
    for s in tqdm(range(args.collect_interval)):    #주어진 수집 간격(collct_interval)동안 반복하면서 데이터를 수집하고 모델을 업데이트
        #         # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
        observations, actions, rewards, nonterminals = D.sample(    #replay buffer에서 데이터 샘플을 추출한다.
            args.batch_size, args.chunk_size
        )  # Transitions start at time t = 0
        # Create initial belief and state for time t = 0
        #초기 신념과 초기 상태를 0으로 초기화한다.
        init_belief, init_state = torch.zeros(args.batch_size, args.belief_size, device=args.device), torch.zeros(
            args.batch_size, args.state_size, device=args.device
        )
        # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
        (
            beliefs,
            prior_states,
            prior_means,
            prior_std_devs,
            posterior_states,
            posterior_means,
            posterior_std_devs,
        ) = transition_model(   #초기 신념, 상태 업데이트=> 전이 모델
            init_state, actions[:-1], init_belief, bottle(encoder, (observations[1:],)), nonterminals[:-1]
        )
        # Calculate observation likelihood, reward likelihood and KL losses (for t = 0 only for latent overshooting); sum over final dims, average over batch and time (original implementation, though paper seems to miss 1/T scaling?)
        if args.worldmodel_LogProbLoss:
            observation_dist = Normal(bottle(observation_model, (beliefs, posterior_states)), 1)
            observation_loss = (
                -observation_dist.log_prob(observations[1:])
                .sum(dim=2 if args.symbolic_env else (2, 3, 4))
                .mean(dim=(0, 1))
            )
        else:
            observation_loss = (
                F.mse_loss(bottle(observation_model, (beliefs, posterior_states)), observations[1:], reduction='none')
                .sum(dim=2 if args.symbolic_env else (2, 3, 4))
                .mean(dim=(0, 1))
            )
        if args.worldmodel_LogProbLoss:
            reward_dist = Normal(bottle(reward_model, (beliefs, posterior_states)), 1)
            reward_loss = -reward_dist.log_prob(rewards[:-1]).mean(dim=(0, 1))
        else:
            reward_loss = F.mse_loss(
                bottle(reward_model, (beliefs, posterior_states)), rewards[:-1], reduction='none'
            ).mean(dim=(0, 1))

        # transition loss
        #사전, 사후 상태 분포 간 KL 손실을 계산
        # posterior 및 prior의 평균과 표준 편차를 사용하여 정규 분포를 생성
        div = kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(prior_means, prior_std_devs)).sum(dim=2)
        kl_loss = torch.max(div, free_nats).mean(
            dim=(0, 1)
        )  # Note that normalisation by overshooting distance and weighting by overshooting distance cancel out
        if args.global_kl_beta != 0: #글로벌 KL이 0이 아닌 경우, 전역 kl 손실을 추가로 계산하여 총 kl 손실에 더한다.
            kl_loss += args.global_kl_beta * kl_divergence(
                Normal(posterior_means, posterior_std_devs), global_prior
            ).sum(dim=2).mean(dim=(0, 1))
        # Calculate latent overshooting objective for t > 0
        if args.overshooting_kl_beta != 0:
            overshooting_vars = []  # Collect variables for overshooting to process in batch
            for t in range(1, args.chunk_size - 1):
                d = min(t + args.overshooting_distance, args.chunk_size - 1)  # Overshooting distance
                t_, d_ = t - 1, d - 1  # Use t_ and d_ to deal with different time indexing for latent states
                seq_pad = (
                    0,
                    0,
                    0,
                    0,
                    0,
                    t - d + args.overshooting_distance,
                )  # Calculate sequence padding so overshooting terms can be calculated in one batch
                # Store (0) actions, (1) nonterminals, (2) rewards, (3) beliefs, (4) prior states, (5) posterior means, (6) posterior standard deviations and (7) sequence masks
                overshooting_vars.append(
                    (
                        F.pad(actions[t:d], seq_pad),
                        F.pad(nonterminals[t:d], seq_pad),
                        F.pad(rewards[t:d], seq_pad[2:]),
                        beliefs[t_],
                        prior_states[t_],
                        F.pad(posterior_means[t_ + 1 : d_ + 1].detach(), seq_pad),
                        F.pad(posterior_std_devs[t_ + 1 : d_ + 1].detach(), seq_pad, value=1),
                        F.pad(torch.ones(d - t, args.batch_size, args.state_size, device=args.device), seq_pad),
                    )
                )  # Posterior standard deviations must be padded with > 0 to prevent infinite KL divergences
            overshooting_vars = tuple(zip(*overshooting_vars))
            # Update belief/state using prior from previous belief/state and previous action (over entire sequence at once)
            beliefs, prior_states, prior_means, prior_std_devs = transition_model(
                torch.cat(overshooting_vars[4], dim=0),
                torch.cat(overshooting_vars[0], dim=1),
                torch.cat(overshooting_vars[3], dim=0),
                None,
                torch.cat(overshooting_vars[1], dim=1),
            )
            seq_mask = torch.cat(overshooting_vars[7], dim=1)
            # Calculate overshooting KL loss with sequence mask
            kl_loss += (
                (1 / args.overshooting_distance)
                * args.overshooting_kl_beta
                * torch.max(
                    (
                        kl_divergence(
                            Normal(torch.cat(overshooting_vars[5], dim=1), torch.cat(overshooting_vars[6], dim=1)),
                            Normal(prior_means, prior_std_devs),
                        )
                        * seq_mask
                    ).sum(dim=2),
                    free_nats,
                ).mean(dim=(0, 1))
                * (args.chunk_size - 1)
            )  # Update KL loss (compensating for extra average over each overshooting/open loop sequence)
            # Calculate overshooting reward prediction loss with sequence mask
            if args.overshooting_reward_scale != 0:
                reward_loss += (
                    (1 / args.overshooting_distance)
                    * args.overshooting_reward_scale
                    * F.mse_loss(
                        bottle(reward_model, (beliefs, prior_states)) * seq_mask[:, :, 0],
                        torch.cat(overshooting_vars[2], dim=1),
                        reduction='none',
                    ).mean(dim=(0, 1))
                    * (args.chunk_size - 1)
                )  # Update reward loss (compensating for extra average over each overshooting/open loop sequence)




       # Apply linearly ramping learning rate schedule : 초기에 더 빠르게, 나중에 느리게 핚습률 조정
        if args.learning_rate_schedule != 0:
            for group in model_optimizer.param_groups:
                group['lr'] = min(
                    group['lr'] + args.model_learning_rate / args.model_learning_rate_schedule, args.model_learning_rate
                )
        model_loss = observation_loss + reward_loss + kl_loss #손실 저의
        # Update model parameters
        model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(param_list, args.grad_clip_norm, norm_type=2)
        model_optimizer.step()

        # Dreamer implementation: actor loss calculation and optimization
        with torch.no_grad():   #사후 state, belief를 그라디언트 계산하지 않고 actor 네트워크에 복사해줌

            actor_states = posterior_states.detach()
            actor_beliefs = beliefs.detach()
        with FreezeParameters(model_modules):
            imagination_traj = imagine_ahead(   #actor 네트워크의 state, belief, model과 trainsition 모델을 통해 상상된 경로를 생성한다.
                actor_states, actor_beliefs, actor_model, transition_model, args.planning_horizon
            )   #imagine_ahead를 통해 미래의 잠재 상태와 belief를 예측한다.
        imged_beliefs, imged_prior_states, imged_prior_means, imged_prior_std_devs = imagination_traj #궤적의 각 요소를 업데이트
        with FreezeParameters(model_modules + value_model.modules):
            imged_reward = bottle(reward_model, (imged_beliefs, imged_prior_states))
            value_pred = bottle(value_model, (imged_beliefs, imged_prior_states))
        returns = lambda_return( #lambda_return : 상상된 보상과 가치 예측을 사용해 미래의 반환값(returns)을 계산
            imged_reward, value_pred, bootstrap=value_pred[-1], discount=args.discount, lambda_=args.disclam
        )
        actor_loss = -torch.mean(returns)
        # Update model parameters
        actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor_model.parameters(), args.grad_clip_norm, norm_type=2)
        actor_optimizer.step()

        # Dreamer implementation: value loss calculation and optimization
        with torch.no_grad():
            value_beliefs = imged_beliefs.detach()
            value_prior_states = imged_prior_states.detach()
            target_return = returns.detach()
        value_dist = Normal(
            bottle(value_model, (value_beliefs, value_prior_states)), 1
        )  # detach the input tensor from the transition network.
        value_loss = -value_dist.log_prob(target_return).mean(dim=(0, 1))
        # Update model parameters
        value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(value_model.parameters(), args.grad_clip_norm, norm_type=2)
        value_optimizer.step()

        # # Store (0) observation loss (1) reward loss (2) KL loss (3) actor loss (4) value loss
        losses.append(
            [observation_loss.item(), reward_loss.item(), kl_loss.item(), actor_loss.item(), value_loss.item()]
        )
        #이쪽을 wandb로 찍기
    # Update and plot loss metrics
    losses = tuple(zip(*losses))
    metrics['observation_loss'].append(losses[0])
    metrics['reward_loss'].append(losses[1])
    metrics['kl_loss'].append(losses[2])
    metrics['actor_loss'].append(losses[3])
    metrics['value_loss'].append(losses[4])

    wandb.log({
        "observation_loss": sum(losses[0]) / len(losses[0]),
        "reward_loss": sum(losses[1]) / len(losses[1]),
        "kl_loss": sum(losses[2]) / len(losses[2]),
        "actor_loss": sum(losses[3]) / len(losses[3]),
        "value_loss": sum(losses[4]) / len(losses[4]),
    }, step=episode)


    lineplot(
        metrics['episodes'][-len(metrics['observation_loss']) :],
        metrics['observation_loss'],
        'observation_loss',
        results_dir,
    )
    lineplot(metrics['episodes'][-len(metrics['reward_loss']) :], metrics['reward_loss'], 'reward_loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['kl_loss']) :], metrics['kl_loss'], 'kl_loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['actor_loss']) :], metrics['actor_loss'], 'actor_loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['value_loss']) :], metrics['value_loss'], 'value_loss', results_dir)

    # Data collection
    print("Data collection") ##여기까지 완성
    with torch.no_grad():
        observation, total_reward = env.reset(), 0 #env.env.env.env.max_steps : 300
        belief, posterior_state, action = (
            torch.zeros(1, args.belief_size, device=args.device),
            torch.zeros(1, args.state_size, device=args.device),
            #torch.zeros(1, env.action_size, device=args.device),
            torch.zeros(1, int(env.action_space.n), device=args.device),
        )

        episode_steps = 0  # 에피소드 내 스텝 수 초기화
        episode_values = []  # 각 스텝에서의 value를 저장할 리스트

        #pbar = tqdm(range(args.max_episode_length // args.action_repeat))
        pbar = tqdm(range(args.max_steps // args.action_repeat))
        print("pbar : ", pbar)
        for t in pbar:  #총 에피소드 길이를 repeat수로 나눠 action 취


            belief, posterior_state, action, next_observation, reward, done = update_belief_and_act(
                args,
                env,
                planner,
                transition_model,
                encoder,
                belief,
                posterior_state,
                action,
                observation.to(device=args.device),
                explore=True,
            )
            D.append(observation, action.cpu(), reward, done)
            total_reward += reward
            episode_steps +=1
            if total_reward > 0.1 :
                pass #env.env.env.env.max_steps : 300
            observation = next_observation
            if args.render:
                env.render()
            if done:
                pbar.close()
                break



        # Update and plot train reward metrics
        metrics['steps'].append(t + metrics['steps'][-1])
        metrics['episodes'].append(episode)
        metrics['train_rewards'].append(total_reward)
        lineplot(
            metrics['episodes'][-len(metrics['train_rewards']) :],
            metrics['train_rewards'],
            'train_rewards',
            results_dir,
        )

        # wandb에 에피소드 성능 기록
        wandb.log({
            "episode": episode,
            "steps": episode_steps,
            "reward": total_reward,
            "mean_value": np.mean(episode_values),
        }, step=episode)

    # # Test model
    # print("Test model")
    # if episode % args.test_interval == 0:
    #     # Set models to eval mode
    #     transition_model.eval()
    #     observation_model.eval()
    #     reward_model.eval()
    #     encoder.eval()
    #     actor_model.eval()
    #     value_model.eval()
    #     # Initialise parallelised test environments
    #     # test_envs = EnvBatcher(
    #     #     env,
    #     #     (env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth, args.max_steps),
    #     #     {},
    #     #     args.test_episodes,
    #     # )
    #
    #     env = GymMoreRedBalls(room_size=10)
    #     env = ActionSpaceWrapper(env, args.max_steps, new_action_space=3)
    #     env = FullyCustom(env, args.max_steps)
    #     # env = MaxStepsWrapper(env, args.max_steps)
    #     test_envs = MaxStepsWrapper(env, args.max_steps, args.symbolic_env, args.seed, args.max_episode_length,
    #                           args.action_repeat, args.bit_depth, new_action_space=3)
    #
    #     with torch.no_grad():
    #         observation, total_rewards, video_frames = test_envs.reset(), np.zeros((args.test_episodes,)), []
    #         belief, posterior_state, action = (
    #             torch.zeros(args.test_episodes, args.belief_size, device=args.device),
    #             torch.zeros(args.test_episodes, args.state_size, device=args.device),
    #             #torch.zeros(args.test_episodes, env.action_size, device=args.device),
    #             torch.zeros(args.test_episodes, int(env.action_space.n), device=args.device),
    #         )
    #         pbar = tqdm(range(args.max_episode_length // args.action_repeat))
    #         for t in pbar:

    #             belief, posterior_state, action, next_observation, reward, done = update_belief_and_act(
    #                 args,
    #                 test_envs,
    #                 planner,
    #                 transition_model,
    #                 encoder,
    #                 belief,
    #                 posterior_state,
    #                 action,
    #                 observation.to(device=args.device), #tuple인데 .to가 있는게 문제
    #             )
    #             total_rewards += reward.numpy()
    #             if not args.symbolic_env:  # Collect real vs. predicted frames for video
    #                 video_frames.append(
    #                     make_grid(
    #                         torch.cat([observation, observation_model(belief, posterior_state).cpu()], dim=3) + 0.5,
    #                         nrow=5,
    #                     ).numpy()
    #                 )  # Decentre
    #             observation = next_observation
    #             if done.sum().item() == args.test_episodes:
    #                 pbar.close()
    #                 break
    #
    #         # wandb에 에피소드 성능 기록
    #         wandb.log({
    #                     "test_step": t,
    #                     #"episode_steps": episode_steps,  # 에피소드 내 스텝 수 기록
    #                     "test_reward ": reward.numpy(),
    #                     # "episode_reward": total_reward.item(),  # 에피소드에서의 총 리워드 기록
    #                     # "mean_value": np.mean(episode_values),  # 에피소드에서의 평균 value 기록
    #                     # "max_value": np.max(episode_values),  # 에피소드에서의 최대 value 기록
    #                     # "min_value": np.min(episode_values)  # 에피소드에서의 최소 value 기록
    #         }, step=metrics['steps'][-1])
    #
    #     # Update and plot reward metrics (and write video if applicable) and save metrics
    #     metrics['test_episodes'].append(episode)
    #     metrics['test_rewards'].append(total_rewards.tolist())
    #     lineplot(metrics['test_episodes'], metrics['test_rewards'], 'test_rewards', results_dir)
    #     lineplot(
    #         np.asarray(metrics['steps'])[np.asarray(metrics['test_episodes']) - 1],
    #         metrics['test_rewards'],
    #         'test_rewards_steps',
    #         results_dir,
    #         xaxis='step',
    #     )
    #     if not args.symbolic_env:
    #         episode_str = str(episode).zfill(len(str(args.episodes)))
    #         write_video(video_frames, 'test_episode_%s' % episode_str, results_dir)  # Lossy compression
    #         save_image(
    #             torch.as_tensor(video_frames[-1]), os.path.join(results_dir, 'test_episode_%s.png' % episode_str)
    #         )
    #     torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))
    #
    #     # Set models to train mode
    #     transition_model.train()
    #     observation_model.train()
    #     reward_model.train()
    #     encoder.train()
    #     actor_model.train()
    #     value_model.train()
    #
    #
    #     # Close test environments
    #     test_envs.close()

    writer.add_scalar("train_reward", metrics['train_rewards'][-1], metrics['steps'][-1])
    writer.add_scalar("train/episode_reward", metrics['train_rewards'][-1], metrics['steps'][-1] * args.action_repeat)
    writer.add_scalar("observation_loss", metrics['observation_loss'][0][-1], metrics['steps'][-1])
    writer.add_scalar("reward_loss", metrics['reward_loss'][0][-1], metrics['steps'][-1])
    writer.add_scalar("kl_loss", metrics['kl_loss'][0][-1], metrics['steps'][-1])
    writer.add_scalar("actor_loss", metrics['actor_loss'][0][-1], metrics['steps'][-1])
    writer.add_scalar("value_loss", metrics['value_loss'][0][-1], metrics['steps'][-1])
    print(
        "episodes: {}, total_steps: {}, train_reward: {} ".format(
            metrics['episodes'][-1], metrics['steps'][-1], metrics['train_rewards'][-1]
        )
    )

    # Checkpoint models
    if episode % args.checkpoint_interval == 0:
        torch.save(
            {
                'transition_model': transition_model.state_dict(),
                'observation_model': observation_model.state_dict(),
                'reward_model': reward_model.state_dict(),
                'encoder': encoder.state_dict(),
                'actor_model': actor_model.state_dict(),
                'value_model': value_model.state_dict(),
                'model_optimizer': model_optimizer.state_dict(),
                'actor_optimizer': actor_optimizer.state_dict(),
                'value_optimizer': value_optimizer.state_dict(),
            },
            os.path.join(results_dir, 'models_%d.pth' % episode),
        )
        if args.checkpoint_experience:
            torch.save(
                D, os.path.join(results_dir, 'experience.pth')
            )  # Warning: will fail with MemoryError with large memory sizes


# Close training environment
env.close()
