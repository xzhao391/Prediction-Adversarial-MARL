# noqa: D212, D415
"""
# Simple Spread

```{figure} mpe_simple_spread.gif
:width: 140px
:name: simple_spread
```

This environment is part of the <a href='..'>MPE environments</a>. Please read that page first for general information.

| Import               | `from pettingzoo.mpe import simple_spread_v3` |
|----------------------|-----------------------------------------------|
| Actions              | Discrete/Continuous                           |
| Parallel API         | Yes                                           |
| Manual Control       | No                                            |
| Agents               | `agents= [agent_0, agent_1, agent_2]`         |
| Agents               | 3                                             |
| Action Shape         | (5)                                           |
| Action Values        | Discrete(5)/Box(0.0, 1.0, (5))                |
| Observation Shape    | (18)                                          |
| Observation Values   | (-inf,inf)                                    |
| State Shape          | (54,)                                         |
| State Values         | (-inf,inf)                                    |


This environment has N agents, N landmarks (default N=3). At a high level, agents must learn to cover all the landmarks while avoiding collisions.

More specifically, all agents are globally rewarded based on how far the closest agent is to each landmark (sum of the minimum distances). Locally, the agents are penalized if they collide with other agents (-1 for each collision). The relative weights of these rewards can be controlled with the
`local_ratio` parameter.

Agent observations: `[self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, communication]`

Agent action space: `[no_action, move_left, move_right, move_down, move_up]`

### Arguments



`N`:  number of agents and landmarks

`local_ratio`:  Weight applied to local reward and global reward. Global reward weight will always be 1 - local reward weight.

`max_cycles`:  number of frames (a step for each agent) until game terminates

`continuous_actions`: Whether agent action spaces are discrete(default) or continuous

`dynamic_rescaling`: Whether to rescale the size of agents and landmarks based on the screen size

"""

import numpy as np
from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn
import random
import math
class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        N=3,
        local_ratio=0.5,
        max_cycles=150,
        continuous_actions=False,
        render_mode=None,
        dynamic_rescaling=False,
    ):
        EzPickle.__init__(
            self,
            N=N,
            local_ratio=local_ratio,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        assert (
            0.0 <= local_ratio <= 1.0
        ), "local_ratio is a proportion. Must be between 0 and 1."
        scenario = Scenario()
        world = scenario.make_world(N)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            local_ratio=local_ratio,
            dynamic_rescaling=dynamic_rescaling,
        )
        self.metadata["name"] = "simple_spread_v3"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self, N=3):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 6
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            if i < 3:
                agent.size = .3
                agent.max_speed = 2.0
            else:
                agent.size = .4
                agent.max_speed = 1.6
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.id = i

        # set random initial states
        values = [[-3.5, -2.2, 2.2, 3.5], [-1.2, 1.2, 2.2, 3.5], [2.2, 3.5, 2.2, 3.5],
                  [-3.5, -2.2, -1.2, 1.2], [1.2, 3.5, -1.2, 1.2],
                  [-3.5, -2.2, -3.5, -2.2], [-1.2, 1.2, -3.5, -2.2], [2.2, 3.5, -3.5, -2.2]]

        for i, agent in enumerate(world.agents):
            random_value = random.choice(values)  # Randomly select one
            values.remove(random_value)
            agent.state.p_pos = np.array([random.uniform(random_value[0], random_value[1]),
                                          random.uniform(random_value[2], random_value[3])])
            agent.state.goal = -agent.state.p_pos
            agent.state.heading = math.atan2(agent.state.goal[1] - agent.state.p_pos[1],
                                         agent.state.goal[0] - agent.state.p_pos[0])
            agent.state.p_vel = 0.5 * np.array([np.cos(agent.state.heading), np.sin(agent.state.heading)])
            agent.state.c = np.zeros(world.dim_c)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        collision_reward = 0.0
        if agent.ADV_action[0] == 5:
            collision_reward += .05
        if agent.ADV_action[1] == 5:
            collision_reward += .05
        min_dist_value = 10
        for other in world.agents:
            if other is agent:
                continue
            dist = np.linalg.norm(agent.state.p_pos - other.state.p_pos)
            min_dist = agent.size + other.size
            if dist < min_dist:
                collision_reward += 5.0  # Strong reward for successful collision

            if dist < min_dist_value:
                min_dist_value = dist
                closest_agent = other

        direction = closest_agent.state.p_pos - agent.state.p_pos
        dist = np.linalg.norm(closest_agent.state.p_pos - agent.state.p_pos)
        vel_alignment = np.dot(agent.state.p_vel, direction / (np.linalg.norm(direction) + 1e-6))
        if dist >= 0.7 and dist < 1.3:
            collision_reward += .2 / (dist**2 + 1e-2)  # Encourage getting closer
            if vel_alignment > 0:
                collision_reward += 0.02 * vel_alignment / (dist**2 + 1e-2)  # Higher if moving directly toward them
        return collision_reward


    def observation(self, agent, world):
        # Normalize values (assume world range ~ [-5, 5])
        norm = lambda x: x / 5.0

        # Own features
        p_pos = norm(agent.state.p_pos)
        goal_rel_pos = norm(agent.state.goal - agent.state.p_pos)
        p_vel = norm(agent.state.p_vel)
        heading = agent.state.heading / np.pi  # Normalize to [-1, 1]

        own_features = np.concatenate([p_pos, goal_rel_pos, p_vel, [heading]])

        # Other agents' relative positions and velocities
        other_features = []
        for other in world.agents:
            if other is agent:
                continue
            rel_pos = norm(other.state.p_pos - agent.state.p_pos)
            rel_vel = norm(other.state.p_vel)
            rel_heading = (other.state.heading - agent.state.heading) / np.pi  # relative heading
            other_features.append(np.concatenate([rel_pos, rel_vel, [rel_heading]]))

        if other_features:
            other_features = np.concatenate(other_features)
        else:
            other_features = np.zeros(0)

        obs = np.concatenate([own_features, other_features])
        return obs
