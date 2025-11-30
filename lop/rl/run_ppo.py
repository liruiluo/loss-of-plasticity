import os
import yaml
import pickle
import argparse
import subprocess
import numpy as np

import gym
import numpy as np
import torch
from torch.optim import Adam
from lop.algos.rl.buffer import Buffer
from lop.nets.policies import MLPPolicy
from lop.nets.valuefs import MLPVF
from lop.algos.rl.agent import Agent
from lop.algos.rl.ppo import PPO
from lop.utils.miscellaneous import compute_matrix_rank_summaries


def save_data(cfg, rets, termination_steps,
              pol_features_activity, stable_rank, mu, pol_weights, val_weights,
              action_probs=None, weight_change=None, friction=-1.0, num_updates=0,
              previous_change_time=0, successes=None):
    if weight_change is None:
        weight_change = []
    data_dict = {
        'rets': np.array(rets),
        'termination_steps': np.array(termination_steps),
        'pol_features_activity': pol_features_activity,
        'stable_rank': stable_rank,
        'action_output': mu,
        'pol_weights': pol_weights,
        'val_weights': val_weights,
        'action_probs': action_probs,
        'weight_change': torch.tensor(weight_change).numpy(),
        'friction': friction,
        'num_updates': num_updates,
        'previous_change_time': previous_change_time,
        'successes': None if successes is None else np.array(successes, dtype=np.float32),
    }
    with open(cfg['log_path'], 'wb') as f:
        pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)


def load_data(cfg):
    with open(cfg['log_path'], 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict


def save_checkpoint(cfg, step, learner):
    # Save step, model and optimizer states
    ckpt_dict = dict(
        step = step,
        actor = learner.pol.state_dict(),
        critic = learner.vf.state_dict(),
        opt = learner.opt.state_dict()
    )
    torch.save(ckpt_dict, cfg['ckpt_path'])
    print(f'Save checkpoint at step={step}')


def load_checkpoint(cfg, device, learner):
    # Load step, model and optimizer states
    step = 0
    ckpt_dict = torch.load(cfg['ckpt_path'], map_location=device)
    step = ckpt_dict['step']
    learner.pol.load_state_dict(ckpt_dict['actor'])
    learner.vf.load_state_dict(ckpt_dict['critic'])
    learner.opt.load_state_dict(ckpt_dict['opt'])
    print(f"Successfully restore from checkpoint: {cfg['ckpt_path']}.")
    return step, learner


def main():
    # Setup
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=False, type=str, default='./cfg/ant/std.yml')
    parser.add_argument('-s', '--seed', required=False, type=int, default="1")
    parser.add_argument('-d', '--device', required=False, default='cpu')

    args = parser.parse_args()
    if args.device: device = args.device
    else: device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = yaml.safe_load(open(args.config))
    cfg['seed'] = args.seed
    cfg['log_path'] = cfg['dir'] + str(args.seed) + '.log'
    cfg['ckpt_path'] = cfg['dir'] + str(args.seed) + '.pth'
    cfg['done_path'] = cfg['dir'] + str(args.seed) + '.done'

    bash_command = "mkdir -p " + cfg['dir']
    subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)

    # Set default values
    cfg.setdefault('wd', 0)
    cfg.setdefault('init', 'lecun')
    cfg.setdefault('to_log', [])
    cfg.setdefault('beta_1', 0.9)
    cfg.setdefault('beta_2', 0.999)
    cfg.setdefault('eps', 1e-8)
    cfg.setdefault('no_clipping', False)
    cfg.setdefault('loss_type', 'ppo')
    cfg.setdefault('frictions_file', 'cfg/frictions')
    cfg.setdefault('env_type', 'gym')
    cfg.setdefault('max_grad_norm', 1e9)
    cfg.setdefault('perturb_scale', 0)
    cfg['n_steps'] = int(float(cfg['n_steps']))
    cfg['perturb_scale'] = float(cfg['perturb_scale'])
    n_steps = cfg['n_steps']    

    # Set default values for CBP
    cfg.setdefault('mt', 10000)
    cfg.setdefault('rr', 0)
    cfg['rr'] = float(cfg['rr'])
    cfg.setdefault('decay_rate', 0.99)
    cfg.setdefault('redo', False)
    cfg.setdefault('threshold', 0.03)
    cfg.setdefault('reset_period', 1000)
    cfg.setdefault('util_type_val', 'contribution')
    cfg.setdefault('util_type_pol', 'contribution')
    cfg.setdefault('pgnt', (cfg['rr']>0) or cfg['redo'])
    cfg.setdefault('vgnt', (cfg['rr']>0) or cfg['redo'])
    cfg.setdefault('vf_coef', 1.0)

    # Initialize env
    seed = cfg['seed']
    friction = -1.0
    env_name = cfg['env_name']
    env_type = cfg.get('env_type', 'gym')

    if env_type == 'metaworld':
        # MetaWorld environments use the Gymnasium-style API. We wrap them to:
        # - keep terminated/truncated flags in info
        # - add terminal_observation on episode end
        # - enforce an optional time limit (default 500)
        # - clip actions and return old-Gym (obs, reward, done, info)
        from metaworld import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE

        if env_name not in ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE:
            raise ValueError(f"Unknown MetaWorld environment id: {env_name}")
        env_cls = ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]
        base_env = env_cls()
        # Ensure task randomization is not frozen
        if hasattr(base_env, "_freeze_rand_vec"):
            base_env._freeze_rand_vec = False

        max_episode_steps = int(cfg.get("max_episode_steps", 500))

        class GymCompatEnv:
            def __init__(self, wrapped_env, seed_value=None):
                self.env = wrapped_env
                self.observation_space = wrapped_env.observation_space
                self.action_space = wrapped_env.action_space
                self._seed = seed_value
                self._reset_called = False
                self._steps = 0

            def reset(self):
                # For reproducibility, only pass the seed on the first reset.
                if not self._reset_called and self._seed is not None:
                    obs, _ = self.env.reset(seed=self._seed)
                    self._reset_called = True
                else:
                    obs, _ = self.env.reset()
                self._steps = 0
                return obs

            def step(self, action):
                action = np.clip(action, self.action_space.low, self.action_space.high)
                obs, reward, terminated, truncated, info = self.env.step(action)
                self._steps += 1
                if self._steps >= max_episode_steps:
                    truncated = True
                done = bool(terminated or truncated)
                info = dict(info)
                info["terminated"] = bool(terminated)
                info["truncated"] = bool(truncated)
                if done:
                    info["terminal_observation"] = obs
                return obs, reward, done, info

            def close(self):
                return self.env.close()

        env = GymCompatEnv(base_env, seed_value=seed)
    elif env_name in ['SlipperyAnt-v2', 'SlipperyAnt-v3']:
        # Import only when needed to avoid unnecessary mujoco_py build
        import lop.envs  # noqa: F401
        xml_file = os.path.abspath(cfg['dir'] + f'slippery_ant_{seed}.xml')
        cfg.setdefault('friction', [0.02, 2])
        cfg.setdefault('change_time', int(2e6))

        with open(cfg['frictions_file'], 'rb+') as f:
            frictions = pickle.load(f)
        friction_number = 0
        new_friction = frictions[seed][friction_number]

        if friction < 0: # If no saved friction, use the default value 1.0
            friction = 1.0
        env = gym.make(env_name, friction=new_friction, xml_file=xml_file)
        print(f'Initial friction: {friction:.6f}')
    else:
        env = gym.make(env_name)
    env.name = None

    # Set random seeds
    np.random.seed(seed)
    random_state = np.random.get_state()
    torch_seed = np.random.randint(1, 2 ** 31 - 1)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    
    # Initialize algorithm
    opt = Adam
    num_layers = len(cfg['h_dim'])
    o_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    pol = MLPPolicy(o_dim, a_dim, act_type=cfg['act_type'], h_dim=cfg['h_dim'], device=device, init=cfg['init'])
    vf = MLPVF(o_dim, act_type=cfg['act_type'], h_dim=cfg['h_dim'], device=device, init=cfg['init'])
    np.random.set_state(random_state)
    buf = Buffer(o_dim, a_dim, cfg['bs'], device=device)

    learner = PPO(pol, buf, cfg['lr'], g=cfg['g'], vf=vf, lm=cfg['lm'], Opt=opt,
                  u_epi_up=cfg['u_epi_ups'], device=device, n_itrs=cfg['n_itrs'], n_slices=cfg['n_slices'],
                  u_adv_scl=cfg['u_adv_scl'], clip_eps=cfg['clip_eps'],
                  max_grad_norm=cfg['max_grad_norm'], vf_coef=cfg['vf_coef'], init=cfg['init'],
                  wd=float(cfg['wd']),
                  betas=(cfg['beta_1'], cfg['beta_2']), eps=float(cfg['eps']), no_clipping=cfg['no_clipping'],
                  loss_type=cfg['loss_type'], perturb_scale=cfg['perturb_scale'],
                  util_type_val=cfg['util_type_val'], replacement_rate=cfg['rr'], decay_rate=cfg['decay_rate'],
                  vgnt=cfg['vgnt'], pgnt=cfg['pgnt'], util_type_pol=cfg['util_type_pol'], mt=cfg['mt'],
                  redo=cfg['redo'], threshold=cfg['threshold'], reset_period=cfg['reset_period']
                  )

    to_log = cfg['to_log']
    agent = Agent(pol, learner, device=device, to_log_features=(len(to_log) > 0))

    # Load checkpoint
    if os.path.exists(cfg['ckpt_path']):
        start_step, agent.learner = load_checkpoint(cfg, device, agent.learner)
    else:
        start_step = 0
    
    # Initialize log
    if os.path.exists(cfg['log_path']):
        data_dict = load_data(cfg)
        num_updates = data_dict['num_updates']
        previous_change_time = data_dict['previous_change_time']
        for k, v in data_dict.items():
            try:
                data_dict[k] = list(v)
            except:
                pass
        rets = data_dict['rets']
        termination_steps = data_dict['termination_steps']
        pol_features_activity = data_dict['pol_features_activity']
        stable_rank = data_dict['stable_rank']
        if 'pol_features_activity' in to_log:
            short_term_feature_activity = torch.zeros(size=(1000, num_layers, cfg['h_dim'][0]))
            pol_features_activity = torch.stack(pol_features_activity)
        if 'stable_rank' in to_log:
            stable_rank = torch.stack(stable_rank)
        mu = data_dict['action_output']
        if 'mu' in to_log:
            mu = np.array(mu)
        pol_weights = data_dict['pol_weights']
        if 'pol_weights' in to_log:
            pol_weights = np.array(pol_weights)
        val_weights = data_dict['val_weights']
        if 'val_weights' in to_log:
            val_weights = np.array(val_weights)
        weight_change = data_dict['weight_change']
        successes = list(data_dict.get('successes', []))
    else:
        num_updates = 0
        previous_change_time = 0
        rets, termination_steps = [], []
        mu, weight_change, pol_features_activity, stable_rank, pol_weights, val_weights = [], [], [], [], [], []
        successes = []
        if 'mu' in to_log:
            mu = np.ones(size=(n_steps, a_dim))
        if 'pol_weights' in to_log:
            pol_weights = np.zeros(shape=(n_steps//1000 + 2, (len(pol.mean_net)+1)//2))
        if 'val_weights' in to_log:
            val_weights = np.zeros(shape=(n_steps//1000 + 2, (len(pol.mean_net)+1)//2))
        if 'pol_features_activity' in to_log:
            short_term_feature_activity = torch.zeros(size=(1000, num_layers, cfg['h_dim'][0]))
            pol_features_activity = torch.zeros(size=(n_steps//1000 + 2, num_layers, cfg['h_dim'][0]))
        if 'stable_rank' in to_log:
            stable_rank = torch.zeros(size=(n_steps//10000 + 2,))

    ret = 0
    epi_steps = 0
    o = env.reset()
    print('start_step:', start_step)
    # Interaction loop
    for step in range(start_step, n_steps):
        a, logp, dist, new_features = agent.get_action(o)
        op, r, done, infos = env.step(a)
        if env_type == 'metaworld':
            terminated = bool(infos.get('terminated', False)) if isinstance(infos, dict) else False
            truncated = bool(infos.get('truncated', False)) if isinstance(infos, dict) else False
            if truncated and not terminated:
                with torch.no_grad():
                    v_bootstrap = agent.learner.vf.value(
                        torch.tensor(op, dtype=torch.float32, device=device).unsqueeze(0)
                    ).item()
                r = float(r + cfg['g'] * v_bootstrap)
        epi_steps += 1
        op_ = op
        val_logs = agent.log_update(o, a, r, op_, logp, dist, done)
        # Logging
        with torch.no_grad():
            if 'weight_change' in to_log and 'weight_change' in val_logs.keys(): weight_change.append(val_logs['weight_change'])
            if 'mu' in to_log: mu[step] = a
            if step % 1000 == 0:
                if step % 10000 == 0 and 'stable_rank' in to_log:
                    _, _, _, stable_rank[step//10000] = compute_matrix_rank_summaries(m=short_term_feature_activity[:, -1, :], use_scipy=True)
                if 'pol_features_activity' in to_log:
                    pol_features_activity[step//1000] = (short_term_feature_activity>0).float().mean(dim=0)
                    short_term_feature_activity *= 0
                if 'pol_weights' in to_log:
                    for layer_idx in range((len(pol.mean_net) + 1) // 2):
                        pol_weights[step//1000, layer_idx] = pol.mean_net[2 * layer_idx].weight.data.abs().mean()
                if 'val_weights' in to_log:
                    for layer_idx in range((len(learner.vf.v_net) + 1) // 2):
                        val_weights[step//1000, layer_idx] = learner.vf.v_net[2 * layer_idx].weight.data.abs().mean()
            if 'pol_features_activity' in to_log:
                for i in range(num_layers):
                    short_term_feature_activity[step % 1000, i] = new_features[i]

        o = op
        ret += r
        if done:
            # print(step, "(", epi_steps, ") {0:.2f}".format(ret))
            rets.append(ret)
            termination_steps.append(step)
            # Record episode-level success when available in info dict
            if isinstance(infos, dict):
                successes.append(float(infos.get('success', 0.0)))
            else:
                successes.append(0.0)
            ret = 0
            epi_steps = 0
            if cfg['env_name'] in ['SlipperyAnt-v2', 'SlipperyAnt-v3'] and step - previous_change_time > cfg['change_time']:
                previous_change_time = step
                env.close()
                friction_number += 1
                new_friction = frictions[seed][friction_number]
                print(f'{step}: change friction to {new_friction:.6f}')
                env.close()
                env = gym.make(cfg['env_name'], friction=new_friction, xml_file=xml_file)
                env.name = None
                agent.env = env
            o = env.reset()

        if step % (n_steps//100) == 0 or step == n_steps-1:
            # Save checkpoint
            save_checkpoint(cfg, step, agent.learner)
            # Save data logs
            save_data(cfg=cfg, rets=rets, termination_steps=termination_steps,
                      pol_features_activity=pol_features_activity, stable_rank=stable_rank, mu=mu, pol_weights=pol_weights,
                      val_weights=val_weights, weight_change=weight_change, friction=friction,
                      num_updates=num_updates, previous_change_time=previous_change_time, successes=successes)
            # 打印当前已完成的 episode 成功率和平均回报
            ts = np.asarray(termination_steps, dtype=np.int64)
            succ = np.asarray(successes, dtype=np.float32) if len(successes) > 0 else None
            rets_arr = np.asarray(rets, dtype=np.float32) if len(rets) > 0 else None
            n = min(ts.shape[0], succ.shape[0]) if succ is not None else 0
            if n > 0 and rets_arr is not None and rets_arr.shape[0] >= n:
                mask = ts[:n] <= step
                if np.any(mask):
                    current_success = float(succ[:n][mask].mean())
                    current_return = float(rets_arr[:n][mask].mean())
                    print(f"Step {step}: success={current_success:.3f}, return={current_return:.1f}")

    with open(cfg['done_path'], 'w') as f:
        f.write('All done!')
        print('The experiment finished successfully!')


if __name__ == "__main__":
    main()
