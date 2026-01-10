import gymnasium as gym
import matplotlib.pyplot as plt
import onnxruntime as ort
import numpy as np
import os
import sys
import time

import traci
from sumolib import checkBinary


import gym_envs.envs.traffic_light_support_functions as tlsf

EDGE_LIST = ("660942467#1","-24203041#0","660942464")

def main_with_ai(num_episodes=10, traffic_scale=1.0, onnx_path='models/model_baseline.onnx', use_phaseplan=False):
    env = gym.make("TrafficEnv-V0", render_mode="console", starting_phases=[1, 1, 2, 2, 1, 2],
                   phase_lengths=[
                       [19, 2, 14, 3, 22],
                       [41, 2, 13, 4],
                       [2, 34, 3, 21],
                       [2, 13, 3, 42],
                       [40, 2, 14, 3, 1],
                       [2, 13, 3, 42]
                   ], simulation_time=3600, phase_change_step=5, traffic_scale=traffic_scale)

    co2_emissions = []
    mean_speed = []

    for episode in range(num_episodes):
        obs, info = env.reset()

        ort_session = ort.InferenceSession(onnx_path)

        input_names = [input.name for input in ort_session.get_inputs()]
        print("ONNX input names:", input_names)

        done = False
        total_reward = 0
        total_co2 = 0

        while not done:
            occupancy = np.array(obs["occupancy"], dtype=np.float32).reshape(1, -1)
            vehicle_count = np.array(obs["vehicle_count"], dtype=np.float32).reshape(1, -1)
            last_five_phaseplan = np.array(obs["last_five_phaseplan"], dtype=np.float32)
            last_five_phaseplan = np.expand_dims(last_five_phaseplan, axis=0)

            # print("Last five phaseplan shape:", last_five_phaseplan.shape)

            onnx_inputs = {
                "l_x_occupancy_": occupancy,
                "l_x_vehicle_count_": vehicle_count,
                # "l_x_last_five_phaseplan_": last_five_phaseplan
            }

            if use_phaseplan:
                onnx_inputs["l_x_last_five_phaseplan_"] = last_five_phaseplan

            output = ort_session.run(None, onnx_inputs)
            action = np.argmax(output[0])
            # print("Action:", action)

            obs, reward, terminated, truncated, info = env.step(np.array([action]))
            # print(obs, reward, terminated, truncated, info)
            total_co2 += info['total_co2']
            total_reward += reward
            done = terminated or truncated

            # time.sleep(0.01)

        mean_speed.append(total_reward)
        co2_emissions.append(total_co2)

    avg_co2 = np.mean(co2_emissions)
    avg_mean_speed = np.mean(mean_speed)
    env.close()
    return avg_mean_speed

def main_without_ai(num_episodes=10, traffic_scale=1.0):
    co2_emissions = []
    mean_speed = []

    for episode in range(num_episodes):
        if 'SUMO_HOME' in os.environ:
            sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

        sumo_binary = "/var/lib/flatpak/app/org.eclipse.sumo/x86_64/stable/bc8bf960e2dcde54fcf3c014883f5316456dfcced8b394291a38a6ff707d92ba/files/bin/sumo"

        sumo_cmd = [
            sumo_binary,
            "-c",
            "sumo_files/osm.sumocfg",
            "--start",
            "-e", str(3900),
            "--quit-on-end",
            "--scale", str(traffic_scale)
        ]

        traci.start(sumo_cmd)
        simulation_step = 0
        total_co2 = 0

        while simulation_step < 3900:
            traci.simulationStep()
            step_co2 = sum(traci.vehicle.getCO2Emission(veh_id) for veh_id in traci.vehicle.getIDList())
            simulation_step += 1
            total_co2 += step_co2

            for edge in EDGE_LIST:
                mean_speed.append(traci.edge.getLastStepMeanSpeed(edge)/ 13.89 / len(EDGE_LIST))

            # time.sleep(0.01)

        traci.close()
        co2_emissions.append(total_co2)

    return np.mean(mean_speed) # change to np.mean(co2_emissions) if you want to plot CO2 emissions

if __name__ == "__main__":
    traffic_scales = [0.5, 0.75, 1.0, 1.25, 1.5]
    ai_pp_results = []
    ai_no_pp_results = []
    baseline_results = []
    fixed_results = []

    for scale in traffic_scales:
        print(f"\nRunning simulations for traffic scale {scale}...")
        ai_pp = main_with_ai(num_episodes=1, traffic_scale=scale, onnx_path='models/model_phaseplan_2.0.onnx', use_phaseplan=True)
        ai_no_pp = main_with_ai(num_episodes=1, traffic_scale=scale, onnx_path='models/model_no_phaseplan_2.0.onnx', use_phaseplan=False)
        baseline = main_with_ai(num_episodes=1, traffic_scale=scale, onnx_path='models/model_baseline.onnx', use_phaseplan=False)
        fixed = main_without_ai(num_episodes=1, traffic_scale=scale)
        ai_pp_results.append(ai_pp)
        ai_no_pp_results.append(ai_no_pp)
        fixed_results.append(fixed)
        baseline_results.append(baseline)

    x = np.arange(len(traffic_scales))
    width = 0.2

    fig, ax = plt.subplots()
    ax.bar(x - 1.5 * width, ai_pp_results, width, label='AI-phaseplan')  # First bar
    ax.bar(x - 0.5 * width, ai_no_pp_results, width, label='AI-no phaseplan')  # Second bar
    ax.bar(x + 0.5 * width, fixed_results, width, label='Fixed phaseplan')  # Third bar
    ax.bar(x + 1.5 * width, baseline_results, width, label='Baseline')  # Fourth bar

    # ax.set_ylabel("CO₂ (mg)")
    # ax.set_title("CO₂-kibocsátás: Különböző forgalomnagyságrendek mellett.")
    ax.set_ylabel("Mean speed")
    ax.set_title("Mean speed: Different traffic scales")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Traffic {s}" for s in traffic_scales])
    ax.legend()
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("plots/mean_speed_three_2.0.png")
    plt.show()