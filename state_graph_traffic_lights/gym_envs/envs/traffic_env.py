"""
This module contains the traffic light environment for the traffic light control problem.
"""

import os
import sys
import time
from statistics import mean

import gym_envs.envs.traffic_light_support_functions as tlsf
import gymnasium as gym
import numpy as np
import traci
from gymnasium import spaces
from sumolib import checkBinary


class TrafficEnv(gym.Env):
    """
    The TrafficEnv class is a custom environment for the traffic light control problem.
    """

    metadata = {"render_modes": ["human", "console"]}

    def __init__(
        self,
        render_mode,
        starting_phases,
        phase_lengths,
        traffic_scale=1,
        trafficlight_order=(5, 4, 0, 1, 3, 2),
        phase_number=6,
        cycle_time=60,
        simulation_time=3600,
        phase_change_step=1,
    ):
        self.traffic_scale = traffic_scale
        self.traci_order = trafficlight_order
        self.cycle_time = cycle_time
        self.current_phase = 1
        self.traffic_flow = {"vehicle_count": [0, 0, 0, 0], "occupancy": [0, 0, 0, 0]}
        self.detectors = []
        self.time_limit = simulation_time
        self.current_step = 0
        self.edge_list = ("-2321.18.83#0", "-2321.18.83#1", "-2318.0.00", "2319.0.00")

        if "SUMO_HOME" in os.environ:
            sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

        if render_mode == "console":
            # self.sumo_binary = "/opt/homebrew/Cellar/sumo/1.19.0/bin/sumo"
            # Check if mac or linux
            self.sumo_binary = "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo.exe"
        else:
            # self.sumo_binary = "/opt/homebrew/Cellar/sumo/1.19.0/bin/sumo-gui"
            # self.sumo_binary = checkBinary('sumo-gui')
            # Check if mac or linux
            self.sumo_binary = "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo-gui.exe"
        self.sumo_cmd = [
            self.sumo_binary,
            "-c",
            "SUMO_SmartCity_design_based_2022/SmartCity.sumocfg",
            "--start",
            "-e",
            str(simulation_time),
            "--quit-on-end",
            "--scale",
            str(traffic_scale),
        ]
        traci.start(self.sumo_cmd)

        self.observation_space = spaces.Dict(
            {
                # Megkérdezni Tamásékat a max értékről
                "occupancy": spaces.Box(
                    low=0.0, high=100.0, shape=(4,), dtype=np.float32
                ),
                "vehicle_count": spaces.Box(
                    low=0, high=7200, shape=(4,), dtype=np.int32
                ),
            }
        )

        self.actions = [1, 2, 3]
        self.action_space = spaces.Discrete(len(self.actions))

    def _get_obs(self):
        self._calc_traffic_flow()

        return {
            "occupancy": np.array(self.traffic_flow["occupancy"], dtype=np.float32),
            "vehicle_count": np.array(
                self.traffic_flow["vehicle_count"], dtype=np.int32
            ),
            "last_phase": self.current_phase,
        }

    def _calc_traffic_flow(self):
        for i in range(len(self.detectors)):
            self.traffic_flow["vehicle_count"][i] = (
                traci.inductionloop.getLastIntervalVehicleNumber(f"e1_{i}")
            )
            self.traffic_flow["occupancy"][i] = (
                traci.inductionloop.getLastIntervalOccupancy(f"e1_{i}")
            )

    def _reset_traffic_measurements(self):
        self.traffic_flow = {"vehicle_count": [0, 0, 0, 0], "occupancy": [0, 0, 0, 0]}

    def _get_info(self):
        pass

    def reset(self, seed=None, options=None):
        self.current_step = 0

        traci.route.setParameter(
            "f_0", "vehsPerHour", str(np.random.randint(low=100, high=2000))
        )
        traci.route.setParameter(
            "f_1", "vehsPerHour", str(np.random.randint(low=100, high=2000))
        )
        traci.route.setParameter(
            "f_2", "vehsPerHour", str(np.random.randint(low=100, high=2000))
        )
        traci.route.setParameter(
            "f_3", "vehsPerHour", str(np.random.randint(low=100, high=2000))
        )
        traci.route.setParameter(
            "f_4", "vehsPerHour", str(np.random.randint(low=100, high=2000))
        )
        traci.route.setParameter(
            "f_5", "vehsPerHour", str(np.random.randint(low=100, high=2000))
        )

        traci.load(self.sumo_cmd[1:])
        time.sleep(0.001)

        self.edge_list = traci.edge.getIDList()
        self.detectors = traci.inductionloop.getIDList()

        self._reset_traffic_measurements()

        return self._get_obs(), {}

    def step(self, action):
        # TODO
        obs = self._get_obs()
        reward = "something"  # TODO

        # done = traci.simulation.getTime() / 1000 > self.time_limit
        done = self.current_step > self.time_limit
        # print("Current time: ", self.current_step, " seconds.")
        info = {}
        return obs, reward, done, False, info

    def render(self, mode="human"):
        pass

    def close(self):
        traci.close()
