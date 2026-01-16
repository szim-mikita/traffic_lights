"""
This module contains the state traffic light environment for the traffic light control problem.
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


class StateTrafficEnvAI(gym.Env):
    """
    The StateTrafficEnv class is a custom environment for the traffic light control problem.
    AI is only called when in a main state (not during transitions).
    """

    metadata = {"render_modes": ["human", "console"]}

    def __init__(
        self,
        render_mode,
        starting_phases=None,
        phase_lengths=None,
        traffic_scale=1,
        trafficlight_order=(5, 4, 0, 1, 3, 2),
        phase_number=6,
        cycle_time=60,
        simulation_time=3600,
        phase_change_step=1,
        min_green_time=5,  # Minimum green time before calling AI
        same_state_wait=10,  # Wait time if AI chooses same state
    ):
        self.traffic_scale = traffic_scale
        self.traci_order = trafficlight_order
        self.cycle_time = cycle_time
        self.min_green_time = min_green_time
        self.same_state_wait = same_state_wait
        
        # State management
        self.current_state = 1  # Current main state (1, 2, or 3)
        self.transition_queue = []  # Queue of states to transition through
        self.time_in_current_state = 0  # Time spent in current state/phase
        self.is_ready_for_action = False  # Flag to indicate when AI should be called
        
        # Traffic flow measurements
        self.traffic_flow = {"vehicle_count": [0, 0, 0, 0], "occupancy": [0, 0, 0, 0]}
        self.detectors = []
        self.time_limit = simulation_time
        self.current_step = 0
        self.edge_list = ("-2321.18.83#0", "-2321.18.83#1", "-2318.0.00", "2319.0.00")
        
        # Define traffic light states using SUMO signal strings
        # Format: Each character represents a lane/connection (r=red, y=yellow, g=green, G=green priority)
        # You'll need to adjust these based on your actual junction configuration
        self.state_definitions = {
            # Main states
            1: "GGrrrrrrrrr",  # State 1: Lanes 1,2 green (example, adjust to your junction)
            2: "GrrGrrrrrrr",  # State 2: Lanes 1,3 green
            3: "rrrrrrGrrr",  # State 3: Lane 4 green
            
            # Transition states - yellows and reds for clearance
            "TR_1to2": "yyrrrrrrrr",  # Transition from state 1 to 2
            "TR_1to3": "yyrrrrrrrr",  # Transition from state 1 to 3
            "TR_2to1": "yrryrrrrrr",  # Transition from state 2 to 1
            "TR_2to3": "yrryrrrrrr",  # Transition from state 2 to 3
            "TR_3to1": "rrrrrryrrr",  # Transition from state 3 to 1
            "TR_3to2": "rrrrrryrrr",  # Transition from state 3 to 2
        }
        
        # Intergreen time matrix (in seconds)
        # intergreen_matrix[from_state][to_state] = time needed
        self.intergreen_matrix = {
            1: {1: 0, 2: 4, 3: 5},  # From state 1 to states 1,2,3
            2: {1: 3, 2: 0, 3: 4},  # From state 2 to states 1,2,3
            3: {1: 6, 2: 5, 3: 0},  # From state 3 to states 1,2,3
        }
        
        # Transition path definitions
        # Maps (from_state, to_state) to list of intermediate transition states
        self.transition_paths = {
            (1, 2): ["TR_1to2"],
            (1, 3): ["TR_1to3"],
            (2, 1): ["TR_2to1"],
            (2, 3): ["TR_2to3"],
            (3, 1): ["TR_3to1"],
            (3, 2): ["TR_3to2"],
        }
        
        # SUMO setup
        if "SUMO_HOME" in os.environ:
            sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

        if render_mode == "console":
            self.sumo_binary = '/var/lib/flatpak/app/org.eclipse.sumo/x86_64/stable/bc8bf960e2dcde54fcf3c014883f5316456dfcced8b394291a38a6ff707d92ba/files/bin/sumo'
        else:
            self.sumo_binary = '/var/lib/flatpak/app/org.eclipse.sumo/x86_64/stable/bc8bf960e2dcde54fcf3c014883f5316456dfcced8b394291a38a6ff707d92ba/files/bin/sumo-gui'

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

        # Define observation and action spaces
        self.observation_space = spaces.Dict(
            {
                "occupancy": spaces.Box(
                    low=0.0, high=100.0, shape=(4,), dtype=np.float32
                ),
                "vehicle_count": spaces.Box(
                    low=0, high=7200, shape=(4,), dtype=np.int32
                ),
                "current_state": spaces.Discrete(3),  # States 1, 2, 3
                "time_in_state": spaces.Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.float32
                ),
            }
        )

        self.actions = [1, 2, 3]  # The three main states
        self.action_space = spaces.Discrete(len(self.actions))

    def _get_obs(self):
        """Get current observation."""
        self._calc_traffic_flow()
        return {
            "occupancy": np.array(self.traffic_flow["occupancy"], dtype=np.float32),
            "vehicle_count": np.array(
                self.traffic_flow["vehicle_count"], dtype=np.int32
            ),
            "current_state": self.current_state,
            "time_in_state": np.array([self.time_in_current_state], dtype=np.float32),
        }

    def _calc_traffic_flow(self):
        """Calculate traffic flow metrics from detectors."""
        for i in range(len(self.detectors)):
            self.traffic_flow["vehicle_count"][i] = (
                traci.inductionloop.getLastIntervalVehicleNumber(f"e1_{i}")
            )
            self.traffic_flow["occupancy"][i] = (
                traci.inductionloop.getLastIntervalOccupancy(f"e1_{i}")
            )

    def _reset_traffic_measurements(self):
        """Reset traffic flow measurements."""
        self.traffic_flow = {"vehicle_count": [0, 0, 0, 0], "occupancy": [0, 0, 0, 0]}

    def _get_info(self):
        """Get additional info (for debugging/logging)."""
        return {
            "current_state": self.current_state,
            "time_in_state": self.time_in_current_state,
            "is_ready_for_action": self.is_ready_for_action,
            "transition_queue": self.transition_queue.copy(),
            "current_step": self.current_step,
        }

    def _set_traffic_light_state(self, state_key):
        """
        Set the traffic light state using setRedYellowGreenState.
        
        Args:
            state_key: Either an integer (1,2,3) for main states or string for transitions
        """
        try:
            signal_state = self.state_definitions[state_key]
            traci.trafficlight.setRedYellowGreenState(self.junction_id, signal_state)
        except Exception as e:
            print(f"Error setting traffic light state {state_key}: {e}")

    def _start_transition(self, target_state):
        """
        Initiate a transition to a new state.
        
        Args:
            target_state: The target main state (1, 2, or 3)
        """
        if target_state == self.current_state:
            # Same state chosen, just wait
            self.time_in_current_state = 0 # why tho
            self.is_ready_for_action = False
            return
        
        # Build transition queue
        transition_key = (self.current_state, target_state)
        
        if transition_key in self.transition_paths:
            # Add all intermediate transition states to queue
            self.transition_queue = self.transition_paths[transition_key].copy()
            # Add final target state at the end
            self.transition_queue.append(target_state)
        else:
            # Direct transition (shouldn't happen with proper graph)
            self.transition_queue = [target_state]
        
        # Start first transition
        self.time_in_current_state = 0
        self.is_ready_for_action = False
        self._advance_transition_queue() # can we immediately jump into transition state?

    def _advance_transition_queue(self):
        """Advance to the next state in the transition queue."""
        if not self.transition_queue:
            return
        
        next_state = self.transition_queue.pop(0)
        
        # Set the traffic light to the next state
        self._set_traffic_light_state(next_state)
        
        # If it's a main state, update current_state
        if isinstance(next_state, int):
            self.current_state = next_state
        
        # Reset timer for this state
        self.time_in_current_state = 0

    def _process_current_state(self):
        """
        Process the current state and determine if we should advance.
        This is called every simulation step.
        """
        self.time_in_current_state += 1
        
        # Check if we're in a transition state
        if self.transition_queue:
            # We're in a transition - check if it's time to advance
            current_is_main_state = isinstance(self.current_state, int) and not self.transition_queue
            
            # If we just started a transition state, get its duration
            if self.time_in_current_state == 1:
                # Get the intergreen time for this transition
                # We need to know what we're transitioning from/to
                pass  # Duration will be checked below
            
            # For transition states, use intergreen time
            # We need to determine from/to states for the intergreen matrix
            if len(self.transition_queue) > 0:
                next_item = self.transition_queue[0]
                
                # Determine required time
                if isinstance(next_item, int):
                    # Next is a main state, so we're currently in yellow/transition
                    # Use intergreen time from previous main state to next main state
                    from_state = self.current_state if isinstance(self.current_state, int) else self.current_state
                    
                    # For now, use a simple approach: each transition state lasts its intergreen time
                    # You may need to refine this based on your specific transition logic
                    required_time = 3  # Default transition time in seconds
                    
                    if self.time_in_current_state >= required_time:
                        self._advance_transition_queue()
                else:
                    # Next is another transition state
                    required_time = 2  # Intermediate transition
                    if self.time_in_current_state >= required_time:
                        self._advance_transition_queue()
        
        else:
            # We're in a main state
            # Check if we've waited the minimum green time
            if not self.is_ready_for_action:
                if self.time_in_current_state >= self.min_green_time:
                    self.is_ready_for_action = True
            else:
                # We're ready but AI hasn't acted yet within reasonable time
                # This case is handled by the step() function
                pass

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        self.current_step = 0
        
        # Randomize traffic flows
        for i in range(6):
            traci.route.setParameter( # TODO: try this with 0
                f"f_{i}", "vehsPerHour", str(np.random.randint(low=100, high=2000))
            )

        # Reload simulation
        traci.load(self.sumo_cmd[1:])
        time.sleep(0.001)
        
        # Get detector and edge info
        self.edge_list = traci.edge.getIDList()
        self.detectors = traci.inductionloop.getIDList()
        
        # Reset state
        self.current_state = 1
        self.transition_queue = []
        self.time_in_current_state = 0
        self.is_ready_for_action = False
        
        # Set initial traffic light state
        self._set_traffic_light_state(1)
        
        self._reset_traffic_measurements()
        
        # Run simulation until ready for first action
        while not self.is_ready_for_action:
            traci.simulationStep()
            self.current_step += 1
            self._process_current_state()
        
        return self._get_obs(), self._get_info()

    def step(self, action):
        """
        Execute one step in the environment.
        This should only be called when is_ready_for_action is True.
        
        Args:
            action: The target state (0, 1, or 2 corresponding to states 1, 2, 3)
        
        Returns:
            observation, reward, done, truncated, info
        """
        # Map action to state
        target_state = self.actions[action]
        
        # Start transition to target state
        self._start_transition(target_state)
        
        # Determine wait time based on whether state changed
        if target_state == self.current_state:
            # Same state: wait same_state_wait seconds
            wait_time = self.same_state_wait # useless (?)
        else:
            # Different state: complete transition, then wait min_green_time
            # First, complete all transitions
            while self.transition_queue:
                # Get required time for current state/transition
                if isinstance(self.transition_queue[0], int):
                    # Next is main state
                    from_state = self.current_state
                    to_state = self.transition_queue[0]
                    required_time = self.intergreen_matrix.get(from_state, {}).get(to_state, 3)
                else:
                    # Transition state
                    required_time = 3  # Default
                
                # Simulate for required time
                for _ in range(required_time):
                    traci.simulationStep()
                    self.current_step += 1
                    self._process_current_state()
                    
                    # Check if simulation ended
                    if self.current_step >= self.time_limit:
                        break
                
                if self.current_step >= self.time_limit:
                    break
            
            # Now wait min_green_time in the new state
            wait_time = self.min_green_time
        
        # Wait the required time
        for _ in range(wait_time):
            if self.current_step >= self.time_limit:
                break
            traci.simulationStep()
            self.current_step += 1
            self._process_current_state()
        
        # Mark ready for next action
        self.is_ready_for_action = True
        
        # Get new observation
        obs = self._get_obs()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self.current_step >= self.time_limit
        
        # Get info
        info = self._get_info()
        
        return obs, reward, done, False, info

    def _calculate_reward(self):
        """
        Calculate reward based on traffic metrics.
        Customize based on your optimization objectives.
        """
        # Example: minimize total waiting time and queue length
        total_vehicles = sum(self.traffic_flow["vehicle_count"])
        avg_occupancy = mean(self.traffic_flow["occupancy"]) if self.traffic_flow["occupancy"] else 0
        
        # Negative reward for congestion
        reward = -0.1 * total_vehicles - 0.05 * avg_occupancy
        
        return reward

    def render(self, mode="human"):
        """Render the environment (handled by SUMO GUI if enabled)."""
        pass

    def close(self):
        """Close the environment and SUMO connection."""
        traci.close()
