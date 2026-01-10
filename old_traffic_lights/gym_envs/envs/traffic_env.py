"""
This module contains the traffic light environment for the traffic light control problem.
"""
import os
import sys
import time
from statistics import mean

import gymnasium as gym
import numpy as np
import traci
from gymnasium import spaces
from sumolib import checkBinary

import gym_envs.envs.traffic_light_support_functions as tlsf


class TrafficEnv(gym.Env):
    """
    The TrafficEnv class is a custom environment for the traffic light control problem.
    """
    metadata = {'render_modes': ['human', 'console']}

    def __init__(self, render_mode, starting_phases, phase_lengths, traffic_scale=1,
                 trafficlight_order=(5, 4, 0, 1, 3, 2), phase_number=6,
                 cycle_time=60, simulation_time=3600, phase_change_step=1):
        """
        The constructor for the TrafficEnv class.

        :parameter render_mode (str): The render mode for the environment. Must be either 'human' or
        'console'.
        :parameter starting_phases (np.array): The starting phases for the traffic lights.
        :parameter phase_lengths (np.array): The lengths of each phase for the traffic lights.
        :parameter traffic_scale (int): The scale of the traffic in the simulation.
        :parameter trafficlight_order (tuple of int): The order of the traffic lights in the
        simulation.
        :parameter phase_number (int): The number of phases in the simulation.
        :parameter cycle_time (int): The cycle time for the traffic lights.
        :parameter simulation_time (int): The simulation time for the traffic lights.
        :parameter phase_change_step (int): The step for changing the phase.
        :return None
        """
        self.traffic_scale = traffic_scale
        self.traci_order = trafficlight_order
        self.cycle_time = cycle_time
        self.phase_plan = np.empty((5, phase_number, cycle_time), dtype=np.int8)
        self.initial_phase_plan = tlsf.generate_phase_plan(starting_phases, phase_lengths)
        self.traffic_flow = {
            "vehicle_count": [],
            "occupancy": []
        }
        self.detectors = []
        self.time_limit = simulation_time
        self.current_step = 0
        self.edge_list = ("660942467#1","-24203041#0","660942464")

        # Create a np array with 5 initial Phaseplans
        for i in range(5):
            self.phase_plan[i] = self.initial_phase_plan

        if 'SUMO_HOME' in os.environ:
            sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

        if render_mode == 'console':
            #self.sumo_binary = "/opt/homebrew/Cellar/sumo/1.19.0/bin/sumo"
            #Check if mac or linux
            if sys.platform == "darwin":
                self.sumo_binary = "/opt/homebrew/Cellar/sumo/1.20.0/bin/sumo"
            else:
                self.sumo_binary = '/var/lib/flatpak/app/org.eclipse.sumo/x86_64/stable/bc8bf960e2dcde54fcf3c014883f5316456dfcced8b394291a38a6ff707d92ba/files/bin/sumo'
        else:
            #self.sumo_binary = "/opt/homebrew/Cellar/sumo/1.19.0/bin/sumo-gui"
            #self.sumo_binary = checkBinary('sumo-gui')
            #Check if mac or linux
            if sys.platform == "darwin":
                self.sumo_binary = "/opt/homebrew/Cellar/sumo/1.20.0/bin/sumo-gui"
            else:
                self.sumo_binary = '/var/lib/flatpak/app/org.eclipse.sumo/x86_64/stable/bc8bf960e2dcde54fcf3c014883f5316456dfcced8b394291a38a6ff707d92ba/files/bin/sumo-gui'
        self.sumo_cmd = [
            self.sumo_binary,
            "-c",
            "sumo_files/osm.sumocfg",
            "--start",
            "-e", str(simulation_time),
            "--quit-on-end",
            "--scale", str(traffic_scale)
        ]
        traci.start(self.sumo_cmd)

        self.observation_space = spaces.Dict({
            # Megkérdezni Tamásékat a max értékről
            "occupancy": spaces.Box(low=0.0, high=100.0, shape=(6,), dtype=np.float32),
            "vehicle_count": spaces.Box(low=0, high=7200, shape=(6,), dtype=np.int32),
            "last_five_phaseplan": spaces.MultiDiscrete(
                nvec=np.full((5,phase_number, cycle_time), 4), dtype=np.int8,
                start=np.full((5,phase_number, cycle_time), 1))
        })

        self.actions = tlsf.generate_phase_combinations(5, 36,step = phase_change_step)
        self.action_space = spaces.Discrete(len(self.actions))

    def _get_obs(self):
        """
        Returns the observation for the environment.
        :return: The observation for the environment. Occupancy, vehicle count and the last
        five phase plans.
        """

        occupancy = []
        vehicle_count = []
        for idx in range(len(self.detectors)):
            try:
                occupancy.append(mean(self.traffic_flow["occupancy"][idx]))
            except Exception:
                occupancy.append(0)

            try:
                vehicle_count.append(sum(self.traffic_flow["vehicle_count"][idx]))
            except Exception:
                vehicle_count.append(0)

        #for detector in self.detectors:
        #    occopancy.append(self.traffic_flow["occupancy"])
        #    vehicle_count.append(traci.inductionloop.getLastIntervalVehicleNumber(detector))
        return {
            "occupancy": np.array(occupancy, dtype=np.float32),
            "vehicle_count": np.array(vehicle_count, dtype=np.int32),
            "last_five_phaseplan": self.phase_plan
        }

    def _calc_traffic_flow(self):
        """
        Calculate the traffic flow for the environment for the last interval and saves it to the
        internal variable.
        """
        for idx, detector in enumerate(self.detectors):
            self.traffic_flow["occupancy"][idx].append(
                traci.inductionloop.getLastIntervalOccupancy(detector))
            self.traffic_flow["vehicle_count"][idx].append(
                traci.inductionloop.getLastIntervalVehicleNumber(detector))


    def _reset_traffic_flow(self):
        """
        Resets the traffic flow for the environment.
        """
        self.traffic_flow = {
            "vehicle_count": [[] for _ in range(len(self.detectors))],
            "occupancy": [[] for _ in range(len(self.detectors))]
        }



    def _get_info(self):
        """
        Returns the information for the environment.
        :return: The information for the environment.
        """
        pass


    def reset(self, seed=None, options=None):
        """
        Resets the environment.
        :param seed: The seed for the environment.
        :param options: The options for the environment.
        :return: The observation for the environment.
        """
        #super().reset(seed=seed)
        #try:
        #    traci.close()
        #    time.sleep(0.001)
        #except ConnectionError:
        #    pass
        self.current_step = 0

        #Set traffic flow
        #TODO, hogy válaszható paraméter legyen!!!!!
        #self.sumo_cmd[-1] = str(self.np_random.integers(2, 6) / 4.0)
        #traci.start(self.sumo_cmd, port=8813)
        traci.load(self.sumo_cmd[1:])
        time.sleep(0.001)
        for i in range(0, 300):
            traci.simulationStep()
            self.current_step += 1
        for i in range(5):
            self.phase_plan[i] = self.initial_phase_plan

        self.edge_list = traci.edge.getIDList()
        self.detectors = traci.inductionloop.getIDList()

        self._reset_traffic_flow()

        return self._get_obs(), {}

    def step(self, action):
        """
        Takes a step in the environment.
        :param action: The action to take.
        :return: The observation, reward, done and info for the environment.
        """
        #Rotate the phase plan and put the new phase to the end
        self.phase_plan = np.roll(self.phase_plan, 1, axis=1)
        #self.phase_plan[:, -1] = tlsf.change_phase_plan(self.actions[action])
        self.phase_plan[-1] = tlsf.change_phase_plan(self.actions[action[0]], self.cycle_time)
        travel_time = 0
        mean_speed = []
        sum_emission = 0
        self._reset_traffic_flow()
        #TODO imeplementálni a közbeeső idő checket
        start_time = self.current_step
        total_co2 = 0
        for j in range (0,5):
            for i in range(0, self.cycle_time):
                traci.simulationStep()
                self.current_step += 1
                tls = traci.trafficlight.getIDList()
                traci.trafficlight.setRedYellowGreenState(
                    tls[0], tlsf.get_phase_column_for_step(self.phase_plan[-1], i, self.traci_order)
                )

                if i % 5 == 4:
                    self._calc_traffic_flow()

                if (self.current_step - start_time) == 5 * self.cycle_time - 5:
                    obs = self._get_obs()

                for edge in self.edge_list:
                    mean_speed.append(traci.edge.getLastStepMeanSpeed(edge)/ 13.89 / len(self.edge_list))
                    sum_emission += traci.edge.getCO2Emission(edge)
                step_co2 = sum(
                    traci.vehicle.getCO2Emission(veh_id) for veh_id in traci.vehicle.getIDList())
                total_co2 += step_co2
        # The reward is the negative of the traveltimes
        # Reward = átlag sebessség edge-kre
        reward = mean(mean_speed)
        # reward = -sum_emission

        #done = traci.simulation.getTime() / 1000 > self.time_limit
        done = self.current_step > self.time_limit
        #print("Current time: ", self.current_step, " seconds.")
        info = {"total_co2": total_co2}
        return obs, reward, done, False, info

    def render(self, mode='human'):
        pass

    def close(self):
        traci.close()