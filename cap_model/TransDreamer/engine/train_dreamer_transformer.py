import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils as vutils
from utils import Checkpointer
from solver import get_optimizer
from envs import make_env, count_steps
from data import EnvIterDataset
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import os
import numpy as np
from pprint import pprint
import pdb
import torch.autograd.profiler as profiler
from time import time
from collections import defaultdict
import wandb
def anneal_learning_rate(global_step, cfg):

  if (global_step - cfg.arch.prefill) < cfg.optimize.warmup_iter:
    # warmup
    lr = cfg.optimize.base_lr / cfg.optimize.warmup_iter * (global_step - cfg.arch.prefill)

  else:
    lr = cfg.optimize.base_lr

  # decay
  lr = lr * cfg.optimize.exp_rate ** ((global_step  - cfg.arch.prefill)/ cfg.optimize.decay_step)

  if (global_step - cfg.arch.prefill) > cfg.optimize.decay_step:
    lr = max(lr, cfg.optimize.end_lr)

  return lr

def anneal_temp(global_step, cfg):

  temp_start = cfg.arch.world_model.temp_start
  temp_end = cfg.arch.world_model.temp_end
  decay_steps = cfg.arch.world_model.temp_decay_steps
  temp = temp_start - (temp_start - temp_end) * (global_step - cfg.arch.prefill) / decay_steps

  temp = max(temp, temp_end)

  return temp

def simulate_test(model, test_env, cfg, global_step, device):

  model.eval()

  obs = test_env.reset()
  action_list = torch.zeros(1, 1, cfg.env.action_size).float()
  action_list[:, 0, 0] = 1. # B, T, C
  state = None
  done = False
  input_type = cfg.arch.world_model.input_type

  with torch.no_grad():
    while not done:
      next_obs, reward, done = test_env.step(action_list[0, -1].detach().cpu().numpy())
      prev_image = torch.tensor(obs[input_type])
      next_image = torch.tensor(next_obs[input_type])
      action_list, state = model.policy(prev_image.to(device), next_image.to(device), action_list.to(device), global_step, 0.1, state, training=False, context_len=cfg.train.batch_length)
      obs = next_obs

def train(model, cfg, device):
  wandb.init(project='3ball_CAP', entity='hails', name='TransD_20x20_500',config={
    "batch_size": cfg.train.batch_size,
    # "overshooting_distance": cfg.overshooting_distance,

    # "planning_discount": cfg.discount,
    "total_episodes": cfg.total_steps,
    "max_steps": cfg.env.max_steps,
  })
  print("======== Settings ========")
  pprint(cfg)

  print("======== Model ========")
  pprint(model)

  model = model.to(device)

  optimizers = get_optimizer(cfg, model)
  checkpointer_path = os.path.join(cfg.checkpoint.checkpoint_dir, cfg.exp_name, cfg.env.name, cfg.run_id)
  checkpointer = Checkpointer(checkpointer_path, max_num=cfg.checkpoint.max_num)
  with open(checkpointer_path + '/config.yaml', 'w') as f:
    cfg.dump(stream=f, default_flow_style=False)
    print(f"config file saved to {checkpointer_path + '/config.yaml'}")

  if cfg.resume:
    checkpoint = checkpointer.load(cfg.resume_ckpt)

    if checkpoint:
      model.load_state_dict(checkpoint['model'])
      for k, v in optimizers.items():
        if v is not None:
          v.load_state_dict(checkpoint[k])
      env_step = checkpoint['env_step']
      global_step = checkpoint['global_step']

    else:
      env_step = 0
      global_step = 0

  else:
    env_step = 0
    global_step = 0

  writer = SummaryWriter(log_dir=os.path.join(cfg.logdir, cfg.exp_name, cfg.env.name, cfg.run_id), flush_secs=30)

  datadir = os.path.join(cfg.data.datadir, cfg.exp_name, cfg.env.name, cfg.run_id, 'train_episodes')
  test_datadir = os.path.join(cfg.data.datadir, cfg.exp_name, cfg.env.name, cfg.run_id, 'test_episodes')
  train_env = make_env(cfg, writer, 'train', datadir, store=True)
  test_env = make_env(cfg, writer, 'test', test_datadir, store=True)

  # fill in length of 5000 frames # 그냥 랜덤행동으로 5000개 채워서 일단 기본적인 행동들 exploration 하는 부분
  train_env.reset()
  steps = count_steps(datadir, cfg)
  length = 0
  print("Collecting prefill-experience...")
  while steps < cfg.arch.prefill: # prefill 이라고 되어있는 것에서 알수 있듯이 그냥 채우는거(지정한 step 수를 넘기면 breakk)
    action = train_env.sample_random_action() # 진짜 말 그대로 random action 을 뽑겠다는 뜻
    next_obs, reward, done = train_env.step(action[0])

    #action = train_env.action_space.sample()
    #next_obs, reward, done,_,_ = train_env.step(action)

    # print("next_obs:",next_obs)
    # print("reward :",reward)
    length += 1
    steps += done * length
    length = length * (1. - done)
    if done:
      train_env.reset()
  print("Stop Collect...")
  steps = count_steps(datadir, cfg)
  print(f'collected {steps} steps. Start training...') #여기서 steps은 random action을 취해 리턴을 받은 총 steps을 의미(에피소드1,2,...의 steps 모든 합)
  train_ds = EnvIterDataset(datadir, cfg.train.train_steps, cfg.train.batch_length)
  train_dl = DataLoader(train_ds, batch_size=cfg.train.batch_size, num_workers=4)
  train_iter = iter(train_dl)
  global_step = max(global_step, steps)

  obs = train_env.reset()
  state = None
  action_list = torch.zeros(1, 1, cfg.env.action_size).float() # T, C
  action_list[0, 0, 0] = 1.
  input_type = cfg.arch.world_model.input_type
  temp = cfg.arch.world_model.temp_start
  episode_num = 0
  steps_in_episode = 0
  while global_step < cfg.total_steps:

    with torch.no_grad():
      model.eval()

      #next_obs, reward, done = train_env.step(action_list[0, -1].detach().cpu().numpy())
      next_obs, reward, done = train_env.step(action_list[0, -1].detach().cpu().numpy())

      #torch.Size([1, 1, 64, 64])
      prev_image = torch.tensor(obs[input_type])
      next_image = torch.tensor(next_obs[input_type])

      #prev_image = torch.tensor(obs)
      #next_image = torch.tensor(next_obs)
      #global step 고려하기

      action_list, state = model.policy(prev_image.to(device), next_image.to(device), action_list.to(device),
                                        global_step, 0.1, state, context_len=cfg.train.batch_length)

      obs = next_obs
      steps_in_episode += 1
      if done:  # 에피소드가 끝나면 로그를 기록하고 초기화
        # wandb에 로그 기록 (에피소드당 steps, reward)
        wandb.log({
          "episode": episode_num,
          "steps": steps_in_episode,
          "reward": reward
        }, step=episode_num)

        # 로그 출력
        print("episode:", episode_num)
        print("reward:", reward)
        print("steps_in_episode:", steps_in_episode)

        # 에피소드 카운트 증가 및 환경 초기화
        episode_num += 1
        train_env.reset()
        state = None
        action_list = torch.zeros(1, 1, cfg.env.action_size).float()  # T, C
        action_list[0, 0, 0] = 1.
        steps_in_episode = 0  # 에피소드가 끝났으므로 steps 초기화
    if global_step % cfg.train.train_every == 0:

      temp = anneal_temp(global_step, cfg)

      model.train()

      traj = next(train_iter)
      for k, v in traj.items(): # traj 는 지금 dict 로 불러져왔으니까 각각을 device 로 보내는 작업이 필요함
        traj[k] = v.to(device).float()

      logs = {}

      model_optimizer = optimizers['model_optimizer']
      model_optimizer.zero_grad()
      transformer_optimizer = optimizers['transformer_optimizer']
      if transformer_optimizer is not None:
        transformer_optimizer.zero_grad()
      model_loss, model_logs, prior_state, post_state = model.world_model_loss(global_step, traj, temp) #here is problem
      grad_norm_model = model.world_model.optimize_world_model(model_loss, model_optimizer, transformer_optimizer, writer, global_step)
      if cfg.arch.world_model.transformer.warm_up:
        lr = anneal_learning_rate(global_step, cfg)
        for param_group in transformer_optimizer.param_groups:
          param_group['lr'] = lr
      else:
        lr = cfg.optimize.model_lr

      actor_optimizer = optimizers['actor_optimizer']
      value_optimizer = optimizers['value_optimizer']
      actor_optimizer.zero_grad()
      value_optimizer.zero_grad()
      actor_loss, value_loss, actor_value_logs = model.actor_and_value_loss(global_step, post_state, traj, temp)
      grad_norm_actor = model.optimize_actor(actor_loss, actor_optimizer, writer, global_step)
      grad_norm_value = model.optimize_value(value_loss, value_optimizer, writer, global_step)



      if global_step % cfg.train.log_every_step == 0:

        logs.update(model_logs)
        logs.update(actor_value_logs)
        model.write_logs(logs, traj, global_step, writer)

        writer.add_scalar('train_hp/lr', lr, global_step)

        grad_norm = dict(
          grad_norm_model = grad_norm_model,
          grad_norm_actor = grad_norm_actor,
          grad_norm_value = grad_norm_value,
        )

        for k, v in grad_norm.items():
          writer.add_scalar('train_grad_norm/' + k, v, global_step=global_step)

    # wandb.log({
    #   "episode": episode_num,
    #   "steps": global_step,
    #   "reward": reward,
    #   # "model_loss": model_loss.item(),  # 모델 손실 기록
    #   #"actor_loss": actor_loss.item(),  # Actor 손실 기록
    #   #"value_loss": value_loss.item(),  # Value 손실 기록
    #   # "mean_value": np.mean(episode_values),  # 필요시 평균 값 기록
    # }, step=episode_num)
    # evaluate RL
    if global_step % cfg.train.eval_every_step == 0:
      simulate_test(model, test_env, cfg, global_step, device)

    if global_step % cfg.train.checkpoint_every_step == 0:
      env_step = count_steps(datadir, cfg)
      checkpointer.save('', model, optimizers, global_step, env_step)

    global_step += 1

  writer.close()
  wandb.close()
# def train(model, cfg, device):
#     wandb.init(project='3ball_CAP', entity='hails', config={
#         "batch_size": cfg.train.batch_size,
#         #"overshooting_distance": cfg.overshooting_distance,
#
#
#
#         #"planning_discount": cfg.discount,
#         "total_episodes" : cfg.total_steps,
#         "max_steps": cfg.env.max_steps,
#     })
#     print("======== Settings ========")
#     pprint(cfg)
#     print("======== Model ========")
#     pprint(model)
#
#     model = model.to(device)
#
#     # Optimizer 및 체크포인터 설정
#     optimizers = get_optimizer(cfg, model)
#     checkpointer_path = os.path.join(cfg.checkpoint.checkpoint_dir, cfg.exp_name, cfg.env.name, cfg.run_id)
#     checkpointer = Checkpointer(checkpointer_path, max_num=cfg.checkpoint.max_num)
#
#     # 설정 파일 저장
#     with open(checkpointer_path + '/config.yaml', 'w') as f:
#       cfg.dump(stream=f, default_flow_style=False)
#       print(f"config file saved to {checkpointer_path + '/config.yaml'}")
#
#     # 체크포인트 로드 (있을 경우)
#     if cfg.resume:
#       checkpoint = checkpointer.load(cfg.resume_ckpt)
#       if checkpoint:
#         model.load_state_dict(checkpoint['model'])
#         for k, v in optimizers.items():
#           if v is not None:
#             v.load_state_dict(checkpoint[k])
#         env_step = checkpoint['env_step']
#         global_step = checkpoint['global_step']
#       else:
#         env_step = 0
#         global_step = 0
#     else:
#       env_step = 0
#       global_step = 0
#
#     # TensorBoard 설정
#     writer = SummaryWriter(log_dir=os.path.join(cfg.logdir, cfg.exp_name, cfg.env.name, cfg.run_id), flush_secs=30)
#
#     # 환경 설정 및 초기화
#     datadir = os.path.join(cfg.data.datadir, cfg.exp_name, cfg.env.name, cfg.run_id, 'train_episodes')
#     test_datadir = os.path.join(cfg.data.datadir, cfg.exp_name, cfg.env.name, cfg.run_id, 'test_episodes')
#     train_env = make_env(cfg, writer, 'train', datadir, store=True)
#     test_env = make_env(cfg, writer, 'test', test_datadir, store=True)
#
#     # 초기 프리필 단계에서 랜덤 행동으로 데이터 수집
#     train_env.reset()
#     steps = count_steps(datadir, cfg)
#     length = 0
#     while steps < cfg.arch.prefill:
#       action = train_env.sample_random_action()
#       next_obs, reward, done = train_env.step(action[0])
#       length += 1
#       steps += done * length
#       length = length * (1. - done)
#       if done:
#         train_env.reset()
#
#     steps = count_steps(datadir, cfg)
#     print(f'collected {steps} steps. Start training...')
#     train_ds = EnvIterDataset(datadir, cfg.train.train_steps, cfg.train.batch_length)
#     train_dl = DataLoader(train_ds, batch_size=cfg.train.batch_size, num_workers=4)
#     train_iter = iter(train_dl)
#     global_step = max(global_step, steps)
#
#     obs = train_env.reset()
#     state = None
#     action_list = torch.zeros(1, 1, cfg.env.action_size).float()  # T, C
#     action_list[0, 0, 0] = 1.0
#     input_type = cfg.arch.world_model.input_type
#     temp = cfg.arch.world_model.temp_start
#     episode_num = 0  # 에피소드 카운터
#
#     train_steps = 0
#     total_train_steps = 0
#     # 에피소드 단위 학습 루프
#     while episode_num < cfg.total_steps:
#       with torch.no_grad():
#         model.eval()
#         total_reward = 0  # 현재 에피소드의 총 보상
#         done = False
#         print("episode is now :", episode_num)
#         while not done:
#           train_steps = 0
#           # 환경과 상호작용
#           next_obs, reward, done = train_env.step(action_list[0, -1].detach().cpu().numpy())
#           train_steps += 1
#           prev_image = torch.tensor(obs[input_type])
#           next_image = torch.tensor(next_obs[input_type])
#
#           # 모델의 정책에 따라 행동 선택
#           action_list, state = model.policy(
#             prev_image.to(device), next_image.to(device), action_list.to(device),
#             global_step, 0.1, state, context_len=cfg.train.batch_length
#           )
#           total_train_steps += train_steps
#           total_reward += reward
#           obs = next_obs
#           if done:
#             print("total_train_steps per episode : ", train_steps)
#             train_env.reset()
#             state = None
#             action_list = torch.zeros(1, 1, cfg.env.action_size).float()
#             action_list[0, 0, 0] = 1.0
#
#       # 에피소드 종료 후 학습 진행
#       model.train()
#       traj = next(train_iter)
#       for k, v in traj.items():
#         traj[k] = v.to(device).float()
#
#       logs = {}
#
#       # 모델 및 옵티마이저 업데이트
#       model_optimizer = optimizers['model_optimizer']
#       model_optimizer.zero_grad()
#       transformer_optimizer = optimizers['transformer_optimizer']
#       if transformer_optimizer is not None:
#         transformer_optimizer.zero_grad()
#
#       model_loss, model_logs, prior_state, post_state = model.world_model_loss(global_step, traj, temp)
#       grad_norm_model = model.world_model.optimize_world_model(model_loss, model_optimizer, transformer_optimizer,
#                                                                writer, global_step)
#
#       if cfg.arch.world_model.transformer.warm_up:
#         lr = anneal_learning_rate(global_step, cfg)
#         for param_group in transformer_optimizer.param_groups:
#           param_group['lr'] = lr
#       else:
#         lr = cfg.optimize.model_lr
#
#       actor_optimizer = optimizers['actor_optimizer']
#       value_optimizer = optimizers['value_optimizer']
#       actor_optimizer.zero_grad()
#       value_optimizer.zero_grad()
#
#       # Actor와 Value의 손실 계산 및 최적화
#       actor_loss, value_loss, actor_value_logs = model.actor_and_value_loss(global_step, post_state, traj, temp)
#       grad_norm_actor = model.optimize_actor(actor_loss, actor_optimizer, writer, global_step)
#       grad_norm_value = model.optimize_value(value_loss, value_optimizer, writer, global_step)
#
#       wandb.log({
#           "episode": episode_num,
#           "steps": train_steps,
#           "reward": total_reward,
#           "model_loss": model_loss.item(),  # 모델 손실 기록
#           "actor_loss": actor_loss.item(),  # Actor 손실 기록
#           "value_loss": value_loss.item(),  # Value 손실 기록
#           # "mean_value": np.mean(episode_values),  # 필요시 평균 값 기록
#       }, step=episode_num)
#
#       # 로그 기록
#       if global_step % cfg.train.log_every_step == 0:
#         logs.update(model_logs)
#         logs.update(actor_value_logs)
#         model.write_logs(logs, traj, global_step, writer)
#         writer.add_scalar('train_hp/lr', lr, global_step)
#
#         grad_norm = dict(
#           grad_norm_model=grad_norm_model,
#           grad_norm_actor=grad_norm_actor,
#           grad_norm_value=grad_norm_value,
#         )
#
#         for k, v in grad_norm.items():
#           writer.add_scalar('train_grad_norm/' + k, v, global_step=global_step)
#
#       # 주기적인 평가 및 체크포인트 저장
#       if global_step % cfg.train.eval_every_step == 0:
#         simulate_test(model, test_env, cfg, global_step, device)
#
#       if global_step % cfg.train.checkpoint_every_step == 0:
#         env_step = count_steps(datadir, cfg)
#         checkpointer.save('', model, optimizers, global_step, env_step)
#
#       # 에피소드 수 증가
#       episode_num += 1
#       global_step += 1
#
#     writer.close()
#     wandb.close()