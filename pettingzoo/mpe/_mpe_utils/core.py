import numpy as np
from RVO import RVO_update, reach, compute_V_des, reach
import math
import time
from scipy.optimize import minimize
class EntityState:  # physical/external base state of all entities
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None


class AgentState(
    EntityState
):  # state of agents (including communication and internal/mental state)
    def __init__(self):
        super().__init__()
        # communication utterance
        self.c = None


class Action:  # action of the agent
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None


class Entity:  # properties and state of physical world entity
    def __init__(self):
        # name
        self.name = ""
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass


class Landmark(Entity):  # properties of landmark entities
    def __init__(self):
        super().__init__()


class Agent(Entity):  # properties of agent entities
    def __init__(self):
        super().__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = True
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = 0
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None


class World:  # multi-agent world
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e2
        self.contact_margin = 1e-3
        self.prev_disturbs = [np.zeros((3 + 2, 3, 2)), np.zeros((3 + 2, 3, 2)), np.zeros((3 + 2, 3, 2))]


    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        human_action = self.apply_human_action(self.agents[3:])
        for i in range(3):
            p_force[i + 3] = human_action[i]
        human_traj =  self.human_pred(self.agents[3:], p_force[3:])
        safe_actions = self.supervisor(self.agents, p_force[:3], human_traj)
        for i in range(3):
            p_force[i] = safe_actions[i]
            self.agents[i].state.p_force = p_force[i]
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    def supervisor(self, agents, cur_actions, forecast):
        for j in range(2):
            self.agents[j].robot_pred = []
        for index, agent in enumerate(agents[:3]):
            state = np.copy(agent.state.p_pos)
            vel = np.copy(agent.state.p_vel)
            heading = np.copy(agent.state.heading)

            risky_action = agent.ADV_action[0]
            safe_action =  agent.ADV_action[1]
            avail_action = [np.array([0, 0]), np.array([2.2, 0]), np.array([-2.2, 0]),
                            np.array([0, 1.3]), np.array([0, -1.3])]
            if risky_action < 5:
                risky_action_expand = avail_action[risky_action]
            if safe_action < 5:
                safe_action_expand =  avail_action[safe_action]

            lower_bound, upper_bound = -.2, .2 # Example bounds
            bounds = [(lower_bound, upper_bound)] * (6*(3+2))  # 3x3 matrix flattened
            # Initial guess for disturb (zero disturbance)
            initial_disturb = self.prev_disturbs[index].flatten()
            pos_risk = np.copy(state)
            vel_risk = np.copy(vel)
            heading_risk = np.copy(heading)

            pos_safe = np.copy(state)
            vel_safe = np.copy(vel)
            heading_safe = np.copy(heading)

            traj1 = self.agents[0].robot_pred
            traj2 = self.agents[1].robot_pred

            # case 1: both 5
            risk_traj = np.zeros((2, 3))
            safe_traj = np.zeros((2, 3))

            if risky_action == safe_action:
                disturb = np.zeros((5, 2, 3))
            elif safe_action == 5:
                for t in range(3):
                    pos_risk, vel_risk, heading_risk = self.dynamic(agent.max_speed, pos_risk, vel_risk, heading_risk, risky_action_expand)
                    risk_traj[:, t] = pos_risk
                traj0 = [risk_traj]
                res = minimize(self.objective_risk, initial_disturb, args=(forecast, index, traj0, traj1, traj2),
                               method='SLSQP', bounds=bounds, options={'maxiter': 30, 'eps': 5e-3, 'ftol': 2e-2})
                disturb = res.x.reshape(5, 2, 3)
            elif risky_action == 5:
                for t in range(3):
                    pos_safe, vel_safe, heading_safe = self.dynamic(agent.max_speed, pos_safe, vel_safe, heading_safe, safe_action_expand)
                    safe_traj[:, t] = pos_safe
                traj0 = [risk_traj]
                res = minimize(self.objective_safe, initial_disturb, args=(forecast, index, traj0, traj1, traj2),
                               method='SLSQP', bounds=bounds, options={'maxiter': 30, 'eps': 5e-3, 'ftol': 2e-2})
                disturb = res.x.reshape(5, 2, 3)
            else:
                for t in range(3):
                    pos_risk, vel_risk, heading_risk = self.dynamic(agent.max_speed, pos_risk, vel_risk, heading_risk,
                                                                    risky_action_expand)
                    pos_safe, vel_safe, heading_safe = self.dynamic(agent.max_speed, pos_safe, vel_safe, heading_safe,
                                                                    safe_action_expand)
                    risk_traj[:, t] = pos_risk
                    safe_traj[:, t] = pos_safe
                traj0 = [risk_traj, safe_traj]
                res = minimize(self.objective_both, initial_disturb, args=(forecast, index, traj0, traj1, traj2),
                               method='SLSQP', bounds=bounds, options={'maxiter': 30, 'eps': 5e-3, 'ftol': 2e-2})
                disturb = res.x.reshape(5, 2, 3)

            self.prev_disturbs[index] = disturb
            forecast= forecast + disturb[:3, :, :]

            ind_disturb = np.zeros((2,3))
            if agents[index].org_action == risky_action:
                ind_disturb = disturb[3, :, :]
            elif agents[index].org_action == safe_action:
                ind_disturb = disturb[4, :, :]

            a = cur_actions[index]
            robot_pred = np.zeros((2, 3))

            pos_cur = np.copy(state)
            vel_cur = np.copy(vel)
            heading_cur = np.copy(heading)
            crash = False
            for t in range(3):
                if t > 0:
                    pos_cur -= ind_disturb[:, t-1]
                pos_cur, vel_cur, heading_cur = self.dynamic(agent.max_speed, pos_cur, vel_cur, heading_cur, a)
                pos_cur += ind_disturb[:, t]

                robot_pred[:, t] = pos_cur
                crash = crash or self.circles_collide(pos_cur, [forecast[0, :, t], forecast[1, :, t], forecast[2, :, t]], 0.36, .4)
                if index == 1 or index == 2:
                    crash = crash or self.circles_collide(pos_cur, [self.agents[0].robot_pred[:, t]], 0.36, .3)
                if index == 2:
                    crash = crash or self.circles_collide(pos_cur, [self.agents[1].robot_pred[:, t]], 0.36, .3)

            if crash:
                safety_rooms = []
                robot_pred_list = np.zeros((5,2,3))
                for action_id, a in enumerate(avail_action):
                    robot_pred = np.zeros((2, 3))
                    pos_cur = np.copy(state)
                    vel_cur = np.copy(vel)
                    heading_cur = np.copy(heading)
                    if (action_id == risky_action and action_id == safe_action) or (action_id != risky_action and action_id != safe_action):
                        for t in range(3):
                            pos_cur, vel_cur, heading_cur = self.dynamic(agent.max_speed, pos_cur, vel_cur, heading_cur,
                                                                         a)
                            robot_pred[:, t] = pos_cur
                        robot_pred_list[action_id] = robot_pred
                        safety_rooms.append(
                            self.compute_result(np.zeros((5, 2, 3)), forecast, index, robot_pred, traj1,
                                                traj2, 0))
                    elif action_id == risky_action:
                        robot_pred_list[action_id] = risk_traj + disturb[3]
                        safety_rooms.append(
                            self.compute_result(disturb, forecast, index, risk_traj, traj1, traj2, 3))
                    elif action_id == safe_action:
                        robot_pred_list[action_id] = safe_traj + disturb[4]
                        safety_rooms.append(
                            self.compute_result(disturb, forecast, index, safe_traj, traj1, traj2, 4))

                safety_index = safety_rooms.index(max(safety_rooms))
                robot_pred = robot_pred_list[safety_index]
                cur_actions[index] = avail_action[safety_index]
            self.agents[index].robot_pred = robot_pred
        return cur_actions

    def objective_both(self, disturb, forecast, id, traj0, traj1, traj2):
        """Optimized objective function to minimize redundant function calls."""
        results = [
                      self.compute_result(disturb, forecast, id, traj0[0], traj1, traj2, 3)
                  ] + [
                      self.compute_result(disturb, forecast, id, traj0[1], traj1, traj2, 4)
                  ]
        return -results[0] + .5*results[1]

    def objective_risk(self, disturb, forecast, id, traj0, traj1, traj2):
        """Optimized objective function to minimize redundant function calls."""
        results = self.compute_result(disturb, forecast, id, traj0[0], traj1, traj2, 3)
        return -results

    def objective_safe(self, disturb, forecast, id, traj0, traj1, traj2):
        """Optimized objective function to minimize redundant function calls."""
        results = self.compute_result(disturb, forecast, id, traj0[0], traj1, traj2, 4)
        return results


    def compute_result(self, disturb, forecast, id, traj0, traj1, traj2, index):
        disturb = disturb.reshape(5, 2, 3)
        human_disturb = disturb[:3, :, :]  # Disturbances for the 3 human agents
        veh_disturb = disturb[index, :, :]  # Disturbance for this particular vehicle

        # 2) Pre-compute the vehicle trajectories (with disturbance)
        #    human_traj shape: (3, 2, time_steps)
        human_traj = forecast + human_disturb

        #    Original vehicle trajectory shape: (2, time_steps)
        origin_veh_traj = traj0
        #    Add disturbance
        veh_traj = origin_veh_traj + veh_disturb


        hum_T = human_traj.transpose(2, 0, 1)  # (time_steps, 3, 2)
        veh_T = veh_traj.T  # (time_steps, 2)
        diff = hum_T - veh_T[:, None, :]
        dist_humans = (diff ** 2).sum(axis=2)
        min_dist_humans = dist_humans.min(axis=1)
        rooms = min_dist_humans

        # 6) Loop over timesteps only to incorporate "other vehicle" checks, if needed
        if id in [1, 2]:
            diff1_sq = ((traj1 - veh_traj) ** 2).sum(axis=0) + .2
            rooms = np.minimum(rooms, diff1_sq)

        if id == 2:
            diff2_sq = ((traj2 - veh_traj) ** 2).sum(axis=0) + .2
            rooms = np.minimum(rooms, diff2_sq)
        return rooms[2]  # Minimize worst-case safety room

    def circles_collide(self, s1, circles, r1, r2):
        for s2 in circles:
            distance = np.linalg.norm(s1 - s2)
            if distance <= (r1 + r2):
                return True
        return False


    def human_pred(self, agents, p_force):
        fut_traj = np.zeros((3, 2, 3))
        for i, agent in enumerate(agents):
            pos = np.copy(agent.state.p_pos)
            vel = np.copy(agent.state.p_vel)
            heading = np.copy(agent.state.heading)  # Assume each entity has a heading attribute
            action = p_force[i]  # Assume each entity has a steering angle input
            for t in range(3):
                pos, vel, heading= self.dynamic(agent.max_speed, pos, vel, heading, action)
                fut_traj[i, :, t]= pos
        return fut_traj

    def dynamic(self, max_speed, pos, vel, heading, action):
        acc = action[0]
        steer = action[1]

        # Update velocity with damping
        heading_vec = np.array([np.cos(heading), np.sin(heading)])
        speed = np.dot(vel, heading_vec)

        speed += acc * self.dt

        # Clamp speed to max
        if max_speed is not None:
            speed = np.clip(speed, 0.5, max_speed)  # allow reverse

        # Update heading (theta)
        heading += (speed / 1) * math.tan(steer) * self.dt
        heading = self.wrap_to_pi(heading)

        dx = speed * np.cos(heading) * self.dt
        dy = speed * np.sin(heading) * self.dt
        pos += np.array([dx, dy])
        pos[0] = np.clip(pos[0], -4.0, 4.0)
        pos[1] = np.clip(pos[1], -4.0, 4.0)
        vel = np.array([speed * np.cos(heading), speed * np.sin(heading)])
        return pos, vel, heading

    def wrap_to_pi(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def apply_human_action(self, agents):
        X = [agent.state.p_pos for agent in agents]
        V = [agent.state.p_vel for agent in agents]
        V_max = [1.6 for i in range(len(X))]
        goal = [agent.state.goal for agent in agents]
        V_des = compute_V_des(X, goal, V_max)
        a = RVO_update(X, V_des, V)
        return a

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = (
                    np.random.randn(*agent.action.u.shape) * agent.u_noise
                    if agent.u_noise
                    else 0.0
                )
                p_force[i] = agent.action.u + noise
        return p_force


    # integrate physical state
    def integrate_state(self, p_force):
        wheelbase = 1.0  # You can make this a per-entity parameter if needed

        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue

            # Extract state variables
            pos = entity.state.p_pos
            vel = entity.state.p_vel
            heading = entity.state.heading  # Assume each entity has a heading attribute
            action = p_force[i]  # Assume each entity has a steering angle input
            pos, vel, heading = self.dynamic(entity.max_speed, pos, vel, heading, action)
            # Store back to state
            entity.state.p_pos = pos
            entity.state.p_vel = vel
            entity.state.heading = heading

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = (
                np.random.randn(*agent.action.c.shape) * agent.c_noise
                if agent.c_noise
                else 0.0
            )
            agent.state.c = agent.action.c + noise

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if entity_a is entity_b:
            return [None, None]  # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]
