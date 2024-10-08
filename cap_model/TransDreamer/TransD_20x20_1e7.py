import torch
from configs import cfg
from engine.train_dreamer_transformer import train
from model import get_model
from envs.atari_env import Atari
import os
import argparse
import pdb
import gym
import wandb
def get_config():
  parser = argparse.ArgumentParser(description='args for Seq_ROOTS project')

  parser.add_argument('--task', type=str, default='train',
                         help='which task to perfrom: train')
  parser.add_argument('--config-file', type=str, default='config_files/configs_atari.py',
                         help='config file')
  parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                      help='using command line to modify configs.')
  # parser.add_argument('--wandb-project', type=str, default='3ball_CAP', help='WandB project name')
  # parser.add_argument('--wandb-entity', type=str, default='hails', help='WandB entity name')

  args = parser.parse_args()


  if args.config_file:
    cfg.merge_from_file(args.config_file)

  if args.opts:
    cfg.merge_from_list(args.opts)

  if cfg.exp_name == '':
    if not args.config_file:
      raise ValueError('exp name can not be empty when config file is not provided')
    else:
      cfg.exp_name = os.path.splitext(os.path.basename(args.config_file))[0]

  task = {
    'train': train,
  }[args.task]

  return task, cfg

if __name__ == '__main__':

 # print(gym.envs.registry)
  parser = argparse.ArgumentParser(description='Evaluate for Transdreamer')
  parser.add_argument('--wandb-project', type=str, default='3ball_CAP', help='WandB project name')
  parser.add_argument('--wandb-entity', type=str, default='hails', help='WandB entity name')
  parser.add_argument('--run-name', type=str, default='TransD_20x20_1e7', help='WandB run name')
  args = parser.parse_args()
  task, cfg = get_config()



  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  model = get_model(cfg, device, cfg.seed)
  task(model, cfg, device)



