import os
import sys
import time
import gymnasium as gym
import numpy as np
import traci
from gymnasium import spaces
import subprocess 


class StateTrafficEnv(gym.Env):
    def __init__(self, render_mode="console", simulation_time=3600, traffic_scale=1):
        super(StateTrafficEnv, self).__init__()
        self.render_mode = render_mode

        # define minimum time before next action can be taken (s)
        self.min_waiting_time = 5  

        # define action space
        self.action_space = spaces.Discrete(3)  # 3 main traffic light states
        self.states = {
            0: "ggrr",
            1: "rggr",
            2: "rrrg"    
        }
        self.initial_state = 0  # starting with first state 
        self.current_state = self.initial_state
        
        self.current_step = 0
        self.time_limit = simulation_time
        self.time_in_state = 0

        # define transitions between main states
        # key: (current_state, action) -> value: list of (length (s), intermediate_state)
        self.transitions = {
            (0, 1) : [
                (3, "ygrr"),
                (2, "rgrr"),
                (2, "rgur")
            ],
            (1, 0) : [
                (3, "rgyr"),
                (3, "rgrr"),
                (2, "ugrr")
            ],
            (1, 2) : [
                (3, "ryyr"),
                (5, "rrrr"),
                (2, "rrru")
            ],
            (2, 1) : [
                (3, "rrry"),
                (1, "rrrr"),
                (2, "rrur"),
                (1, "rrgr"),
                (2, "rugr")
            ],
            (2, 0) : [
                (3, "rrry"),
                (3, "rrrr"),
                (1, "urrr"),
                (1, "uurr"),
                (1, "gurr")
            ],
            (0, 2) : [
                (3, "yyrr"),
                (1, "rrrr"),
                (2, "rrru"),
            ]
        }

        # define observation space
        self.observation_space = spaces.Dict({
            # TODO: ask about shape and max values
            "occupancy": spaces.Box(low=0.0, high=100.0, shape=(4,), dtype=np.float32),
            "vehicle_count": spaces.Box(low=0, high=7200, shape=(4,), dtype=np.int32),
            "current_state": spaces.Discrete(3),
            "time_in_state": spaces.Box(low=0, high=simulation_time, shape=(), dtype=np.int32),
            "all_states_used": spaces.Discrete(2),
        })

        self.observation_history = {
            "occupancy": [[] for _ in range(4)],
            "vehicle_count": [[] for _ in range(4)],
            "state": []
        }

        # set up SUMO
        if 'SUMO_HOME' in os.environ:
            sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

        if self.render_mode == 'console':
            #Check if mac or linux
            if sys.platform == "darwin":
                self.sumo_binary = "/opt/homebrew/Cellar/sumo/1.20.0/bin/sumo" 
            else:
                self.sumo_binary = "/home/nikita/sumo/bin/sumo"
        else:
            #Check if mac or linux
            if sys.platform == "darwin":
                self.sumo_binary = "/opt/homebrew/Cellar/sumo/1.20.0/bin/sumo-gui" 
            else:
                self.sumo_binary = "/home/nikita/sumo/bin/sumo-gui"
        
        self.sumo_cmd = [
            self.sumo_binary,
            "-c",
            "/home/nikita/dev/project_lab/traffic_lights/state_graph_traffic_lights/sumo_files/SmartCity.sumocfg",
            "--start",
            "-e", str(simulation_time),
            "--quit-on-end",
            "--scale", str(traffic_scale)
        ]

        print("Starting SUMO with command:", ' '.join(self.sumo_cmd))

        traci.start(self.sumo_cmd)

        # get TraCI ids
        self.detectors = traci.inductionloop.getIDList()
        self.edges = traci.edge.getIDList()


    def _translate_to_sumo_phase(self, state):
        """
        Translates the given state to the corresponding SUMO traffic light state string.
        :param state: The state to translate.
        :return: The corresponding SUMO traffic light state string.
        """
        traffic_light_idx_correspondence = {
            0: 2,
            1: 2,
            2: 3,
            3: 3,
            4: 0,
            5: 0,
        }
        mapped_phase = [state[traffic_light_idx_correspondence[i]] for i in range(6)]
        return mapped_phase
    
    def set_sumo_phase(self, state):
        """
        Sets the SUMO traffic light state to the given state.
        :param state: The state to set.
        """
        tls = traci.trafficlight.getIDList()    
        traci.trafficlight.setRedYellowGreenState(tls[3], self._translate_to_sumo_phase(state))


    def _get_obs(self):
        """
        Returns the observation for the environment.
        :return: The observation for the environment. Occupancy and vehicle count.
        """

        occupancy = []
        vehicle_count = []
        for idx in range(len(self.detectors)):
            try:
                occupancy.append(np.mean(self.observation_history["occupancy"][idx])) # possibly change to last value
            except Exception:
                occupancy.append(0.0)

            try:
                vehicle_count.append(np.mean(self.observation_history["vehicle_count"][idx])) # possibly change to last value
            except Exception:
                vehicle_count.append(0)

        obs = {
            "occupancy": np.array(occupancy, dtype=np.float32),
            "vehicle_count": np.array(vehicle_count, dtype=np.int32),
            "current_state": self.current_state,
            "time_in_state": self.time_in_state,
            "all_states_used": int(len(set(self.observation_history["state"][-(min(10, len(self.observation_history["state"]))):])) == 3)
        }

        return obs
    

    def _get_detector_data(self):
        """
        Returns the raw data from the detectors.
        :return: The raw data from the detectors.
        """

        for idx in range(len(self.detectors)):
            try:
                self.observation_history["occupancy"][idx].append(traci.inductionloop.getLastStepOccupancy(self.detectors[idx]))
            except Exception:
                self.observation_history["occupancy"][idx].append(0.0)

            try:
                self.observation_history["vehicle_count"][idx].append(traci.inductionloop.getLastStepVehicleNumber(self.detectors[idx]))
            except Exception:
                self.observation_history["vehicle_count"][idx].append(0)

        return self.observation_history
    

    def _reset_observation_history(self):
        """
        Resets the observation history.
        """
        self.observation_history["occupancy"] = [[] for _ in range(4)]
        self.observation_history["vehicle_count"] = [[] for _ in range(4)]


    def _get_reward(self):
        """
        Returns the reward for the current step.
        :return: The reward for the current step.
        """
        # Example: CO2 emission as negative reward
        reward = 0.0
        for edge in self.edges:
            try:
                reward -= traci.edge.getCO2Emission(edge) / 10000
            except Exception:
                pass
        reward += 1 - np.exp(0.01 * self.time_in_state)  # small penalty for staying in one state
        # print(f"        [REWARD] time_in_state={self.time_in_state} CO2 reward={reward + np.exp(0.001 * self.time_in_state):.3f}")
        # print(f"        [REWARD] time reward={-np.exp(0.001 * self.time_in_state):.3f}")
        # print(f"        [REWARD] total reward={reward:.3f}")
        reward -= 10 if len(set(self.observation_history["state"][-(min(10, len(self.observation_history["state"]))):])) < 3 else 0.0  # penalty for not changing states
        return reward


    def _get_info(self):
        """
        Returns the information for the environment.
        :return: The information for the environment.
        """
        pass


    def _skip_n_steps(self, n, observe=True, reward=False):
        """
        Steps the simulation n times, collecting detector data.
        :param n: The number of steps (seconds) to skip.
        """
        reward_total = 0.0
        for _ in range(n):
            traci.simulationStep()
            if observe:
                self._get_detector_data()
            if reward:
                reward_total += self._get_reward()
            self.current_step += 1
            self.time_in_state += 1
        return reward_total / n


    def reset(self, seed=None, options=None):
        """
        Resets the environment.
        :param seed: The seed for the environment.
        :return: The initial observation and info.
        """
        self._reset_observation_history()

        traci.load(self.sumo_cmd[1:])
        time.sleep(0.001)

        self.current_step = 0
        self.current_state = self.initial_state
        self.observation_history["state"] = [self.current_state]
        self.time_in_state = 0

        # set initial traffic light state
        self.set_sumo_phase(self.states[self.current_state])

        # step once to get initial observation
        self._skip_n_steps(1, observe=True)

        return self._get_obs(), {}
    

    def step(self, action):
        """
        Steps the environment by taking the given action.
        
        :param self: The environment instance.
        :param action: The action to take (0, 1, or 2).
        """

        self._reset_observation_history()

        if action not in [0, 1, 2]:
            raise ValueError("Invalid action. Action must be 0, 1, or 2.")
        
        # stay in the same state
        if action == self.current_state:
            reward = self._skip_n_steps(self.min_waiting_time, reward=True)
        else: 
            transition_key = (self.current_state, action)
            if transition_key in self.transitions:
                reward = 0.0
                for length, intermediate_state in self.transitions[transition_key]:
                    # set intermediate state
                    self.set_sumo_phase(intermediate_state)
                    # step simulation
                    reward += self._skip_n_steps(length, observe=True) # do not collect data during transitions (maybe)
                self.current_state = action
                self.observation_history["state"].append(self.current_state)
                self.time_in_state = 0
                self.set_sumo_phase(self.states[self.current_state])
                reward += self._skip_n_steps(self.min_waiting_time, reward=True)  # step mean waiting time collect data after transition
            else: # shouldn't happen if nothing goes extremely wrong
                raise ValueError(f"Invalid transition from state {self.current_state} to {action}.") 
            
        # get observation
        observation = self._get_obs()

        # check if done
        done = self.current_step >= self.time_limit
        # print(f"        [STEP] step={self.current_step} action={action} reward={reward:.3f} done={done}")

        return observation, reward, done, False, {}  # info is empty dict

    def close(self):
        # Always clean up
        traci.close()