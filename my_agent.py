from grid2op.gym_compat import GymEnv, BoxGymObsSpace, BoxGymActSpace, MultiDiscreteActSpace, DiscreteActSpace
from gymnasium.spaces import Discrete, Dict, MultiDiscrete, MultiBinary
from grid2op.Observation import BaseObservation
from grid2op.Converter import ToVect, Converter
from grid2op.Action import BaseAction
from l2rpn_baselines.utils.gymAgent import GymAgent
from stable_baselines3 import PPO
from .optimCVXPY import OptimCVXPY
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def make_agent(env, this_directory_path):
    agent = Agent(env, this_directory_path)
    return agent

def load_topo_action_dict(file, env, by_area = False) :
    vect_action_space = np.load(file)["action_space"]
    converter = ToVect(env.action_space)
    best_actions_list = []
    for i in range(len(vect_action_space)) :
        best_actions_list.append(converter.convert_act(vect_action_space[i]))

    if by_area :
        sub_by_area = env._game_rules.legal_action.substations_id_by_area
        action_by_area = { i : [] for i in range(len(sub_by_area.keys()))}

        for i,act in enumerate(best_actions_list) :
            sub_id = int(act.as_dict()['set_bus_vect']['modif_subs_id'][0])
            for i,subs in sub_by_area.items() :
                if sub_id in subs :
                    action_by_area[i].append(act)
        
        return action_by_area
    
    else :
        return {'all_best_actions' : best_actions_list}
    
class GlobalTopoActionSpace(Discrete):
    def __init__(self,topo_actions_list : list, g2op_action_space):
        Discrete.__init__(self, len(topo_actions_list))
        self.topo_actions_list = topo_actions_list
        self.g2op_action_space = g2op_action_space

    def from_gym(self, gym_action):
        return self.topo_actions_list[int(gym_action)]
    
    def close(self):
        if hasattr(self, "_init_env"):
            self._init_env = None  # this doesn't own the environment

class Agent(GymAgent):

    def __init__(self, env, this_directory_path):
        self.this_directory_path = this_directory_path
        self.action_retraining = load_topo_action_dict(os.path.join(this_directory_path, "action", "action_retraining.npz"), env)['all_best_actions']
        self.optimcvxpy_config = self.get_optimcvxpy_config()
        self.optim_agent = self.initialize_optimcvxpy_agent(env)
        self.initialize(env)

    def get_optimcvxpy_config(self):
        return {
            "margin_th_limit": 0.93,
            "alpha_por_error": 0.5,
            "rho_danger": 0.99,
            "rho_safe": 0.9,
            "penalty_curtailment_unsafe": 15,
            "penalty_redispatching_unsafe": 0.005,
            "penalty_storage_unsafe": 0.0075,
            "penalty_curtailment_safe": 0.0,
            "penalty_redispatching_safe": 0.0,
            "penalty_storage_safe": 0.0,
            "weight_redisp_target": 1.0,
            "weight_storage_target": 1.0,
            "weight_curtail_target": 1.0,
            "margin_rounding": 0.01,
            "margin_sparse": 5e-3,
            "max_iter": 100000,
            "areas": True
        }

    def initialize_optimcvxpy_agent(self, env):
        action_space_paths = {
            "N1_safe": os.path.join(self.this_directory_path, "action", "action_N1_safe.npz"),
            "N1_interm": os.path.join(self.this_directory_path, "action", "action_N1_interm.npz"),
            "N1_unsafe": os.path.join(self.this_directory_path, "action", "action_N1_unsafe.npz"),
            "12_unsafe": os.path.join(self.this_directory_path, "action", "action_12_unsafe.npz"),
        }
        return OptimCVXPY(env,
                          env.action_space,
                          action_space_path_N1_safe=action_space_paths["N1_safe"],
                          action_space_path_N1_interm=action_space_paths["N1_interm"],
                          action_space_path_N1_unsafe=action_space_paths["N1_unsafe"],
                          action_space_path_12_unsafe=action_space_paths["12_unsafe"],
                          config=self.optimcvxpy_config,
                          verbose=True)

    def initialize(self, env):
        self.env = env
        self.gymenv = self.make_gym_env()
        nn_kwargs = {'verbose': 1}
        super().__init__(self.env.action_space,
                         self.gymenv.action_space,
                         self.gymenv.observation_space,
                         gymenv=self.gymenv,
                         nn_kwargs=nn_kwargs)

    def make_gym_env(self):
        env = self.env
        gymenv = GymEnv(env)
        gymenv.observation_space.close()
        gymenv.observation_space = BoxGymObsSpace(env.observation_space, attr_to_keep=["rho"])
        gymenv.action_space.close()
        gymenv.action_space = GlobalTopoActionSpace(self.action_retraining, env.action_space)
        gymenv.action_space = MultiDiscreteActSpace(env.action_space, attr_to_keep=["set_bus"])
        return gymenv

    def save_state(self, savestate_path):
        self.save(savestate_path)

    def load_state(self, loadstate_path):
        self.load(loadstate_path)

    def save(self, path):
        path_agent = os.path.join(path, "agent.model")
        self.nn_model.save(path_agent)

    def load(self, path):
        custom_objects = dict()
        custom_objects["action_space"] = self.gymenv.action_space
        custom_objects["observation_space"] = self.gymenv.observation_space
        path_agent = os.path.join(path, "agent.model")
        self.nn_model = PPO.load(path_agent, env=self.gymenv, custom_objects=custom_objects)

    def get_act(self, gym_obs, reward, done):
        action, _ = self.nn_model.predict(gym_obs, deterministic=True)
        return action

    def act(self, observation: BaseObservation, reward: float, done: bool) -> BaseAction:
        # Integrate OptimCVXPY
        grid2op_act = self.optim_agent.act(observation)
        return grid2op_act

    def build(self):
        self.nn_model = PPO(policy="MlpPolicy", env=self.gymenv, **self._nn_kwargs)

    def train(self, env, n_envs, remaining_time):
        print("I AM TRAINING")
        self.initialize(env)
        self.nn_model.learn(total_timesteps=25000)
