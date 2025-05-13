import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from mappo_mpe import MAPPO_MPE
from mappo_mpe_adv import MAPPO_MPE_ADV
from pettingzoo.mpe import simple_adversary_v2, simple_spread_v3, simple_tag_v2
import time
from utils import index_to_one_hot

class Runner_MAPPO_MPE:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Create env
        self.env, dim_info = self.get_env()
        self.args.N = 3  # The number of agents
        self.args.obs_dim_n = [32 for i in range(6)]  # obs dimensions of N agents
        self.args.action_dim_n = [5 for i in range(6)]  # actions dimensions of N agents
        # Only for homogenous agents environments like Spread in MPE,all agents have the same dimension of observation space and action space
        self.args.obs_dim = self.args.obs_dim_n[0]  # The dimensions of an agent's observation space
        self.args.action_dim = self.args.action_dim_n[0]  # The dimensions of an agent's action space
        self.args.state_dim = 96  # fixed!!  The dimensions of global state space（Sum of the dimensions of the local observation space of all agents）
        print("observation_space=", self.env.observation_space)
        print("obs_dim_n={}".format(self.args.obs_dim_n))
        print("action_space=", self.env.action_space)
        print("action_dim_n={}".format(self.args.action_dim_n))

        # Create N agents
        self.agent_n= MAPPO_MPE(self.args)
        self.agent_n.load_model('simple_spread', 1, 0, 3000)

        # Create adv agents
        self.args.action_dim_n =  [36 for i in range(6)]
        self.args.action_dim = self.args.action_dim_n[0]
        self.args.obs_dim_n = [35 for i in range(6)]  # obs dimensions of N agents
        self.args.obs_dim = self.args.obs_dim_n[0]
        self.args.state_dim = 35 * 3
        args.rnn_hidden_dim = 128
        args.mlp_hidden_dim = 128
        self.agent_n_adv = MAPPO_MPE_ADV(self.args)



        self.replay_buffer = ReplayBuffer(self.args)

        # Create a tensorboard
        self.writer = SummaryWriter(log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}'.format(self.env_name, self.number, self.seed))

        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=self.args.N)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=self.args.N, gamma=self.args.gamma)

    def get_env(self, ):
        """create environment and get observation and action dimension of each agent in this environment"""
        new_env = simple_spread_v3.parallel_env(max_cycles=150)

        new_env.reset()
        _dim_info = {}
        for agent_id in new_env.agents:
            _dim_info[agent_id] = []  # [obs_dim, act_dim]
            _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0])
            _dim_info[agent_id].append(new_env.action_space(agent_id).n)

        return new_env, _dim_info

    def run(self, ):
        evaluate_num = -1  # Record the number of evaluations
        my_reward = 0
        i = 0
        while self.total_steps < self.args.max_train_steps:
            # if self.total_steps >= 250:
            #     self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
            #     evaluate_num += 1

            my_reward_local, episode_steps = self.run_episode_mpe(evaluate=False)  # Run an episode
            self.total_steps += episode_steps
            my_reward += my_reward_local
            if self.replay_buffer.episode_num == self.args.batch_size:
                self.agent_n_adv.train(self.replay_buffer, self.total_steps)  # Training
                self.replay_buffer.reset_buffer()
                i += 1
                if i == 32 or i % 5 == 0:
                    print(self.total_steps / 50, my_reward / (self.args.batch_size * 5))
                    my_reward = 0
                    i = 0
                    self.agent_n_adv.save_model(self.env_name, self.number, self.seed, self.total_steps)


        self.evaluate_policy()
        self.env.close()

    def evaluate_policy(self, ):
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            episode_reward, _ = self.run_episode_mpe(evaluate=True)
            evaluate_reward += episode_reward

        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)
        print("total_steps:{} \t evaluate_reward:{}".format(self.total_steps, evaluate_reward))
        self.writer.add_scalar('evaluate_step_rewards_{}'.format(self.env_name), evaluate_reward, global_step=self.total_steps)
        # Save the rewards and models
        np.save('./data_train/MAPPO_env_{}_number_{}_seed_{}.npy'.format(self.env_name, self.number, self.seed), np.array(self.evaluate_rewards))
        self.agent_n_adv.save_model(self.env_name, self.number, self.seed, self.total_steps)





    def run_episode_mpe(self, evaluate=False):
        episode_reward = 0
        obs_n, _ = self.env.reset()
        obs_n = np.stack([obs_n[a] for a in obs_n], axis=0)[:3]
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        if self.args.use_rnn:  # If use RNN, before the beginning of each episode，reset the rnn_hidden of the Q network.
            self.agent_n_adv.actor.rnn_hidden = None
            self.agent_n_adv.critic.rnn_hidden = None
        for episode_step in range(self.args.episode_limit):
            # img = self.env.render()
            # time.sleep(.1)

            a_n, a_logprob_n = self.agent_n.choose_action(obs_n, evaluate=evaluate)  # Get actions and the corresponding log probabilities of N agents

            obs_n_adv = np.hstack((obs_n, np.tile(a_n, (3, 1)) / 4))
            a_n_adv, a_logprob_n_adv = self.agent_n_adv.choose_action(obs_n_adv, evaluate=evaluate)  # Get actions and the corresponding log probabilities of N agents
            for i in range(3):
                row, col =  np.where(index_to_one_hot(int(a_n_adv[i]), int(36)).reshape(6, 6) == 1)
                self.env.aec_env.world.agents[i].ADV_action = [row[0], col[0]]
                self.env.aec_env.world.agents[i].org_action = a_n[i]

            a_mod1 = np.concatenate((a_n,  [0, 0, 0]), axis = 0)
            a_mod2 = {a: a_mod1[i] for i, a in enumerate(self.env.possible_agents)}
            s_adv = np.array(obs_n_adv).flatten()  # In MPE, global state is the concatenation of all agents' local obs.
            v_n_adv = self.agent_n_adv.get_value(s_adv)  # Get the state values (V(s)) of N agents

            s = np.array(obs_n).flatten()  # In MPE, global state is the concatenation of all agents' local obs.
            v_n = self.agent_n.get_value(s)  # Get the state values (V(s)) of N agents
            obs_next_n, r_n, done_n, _, _ = self.env.step(a_mod2)
            obs_next_n = np.stack([obs_next_n[a] for a in obs_next_n], axis=0)[:3]
            r_n = np.stack([r_n[a] for a in r_n], axis=0)[:3]
            done_n = np.stack([done_n[a] for a in done_n], axis=0)[:3]

            episode_reward += r_n[0]

            # if not evaluate:
            #     if self.args.use_reward_norm:
            #         r_n = self.reward_norm(r_n)
            #     elif args.use_reward_scaling:
            #         r_n = self.reward_scaling(r_n)

                # Store the transition

            # self.replay_buffer.store_transition(episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n)
            self.replay_buffer.store_transition(episode_step, obs_n_adv, s_adv, v_n_adv, a_n_adv,
                                                a_logprob_n_adv, r_n, done_n)

            obs_n = obs_next_n[:3]
            if all(done_n):
                break

        if not evaluate:
            # An episode is over, store v_n in the last step
            s_adv = np.array(obs_n_adv).flatten()
            v_n_adv = self.agent_n_adv.get_value(s_adv)
            self.replay_buffer.store_last_value(episode_step + 1, v_n_adv)

        return episode_reward, episode_step + 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in MPE environment")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=50, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=50, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (the number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the rnn")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the mlp")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="GAE parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling. Here, we do not use it.")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=float, default=True, help="Whether to use relu, if False, we will use tanh")
    parser.add_argument("--use_rnn", type=bool, default=True, help="Whether to use RNN")
    parser.add_argument("--add_agent_id", type=float, default=False, help="Whether to add agent_id. Here, we do not use it.")
    parser.add_argument("--use_value_clip", type=float, default=False, help="Whether to use value clip.")

    args = parser.parse_args()
    runner = Runner_MAPPO_MPE(args, env_name="simple_spread", number=1, seed=0)
    runner.run()
