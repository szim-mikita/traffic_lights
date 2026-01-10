"""
This module contains functions for generating and modifying phase plans for traffic lights.

Functions:
- generate_phase(start, lengths): Generates a phase sequence based on a starting phase and phase
 lengths.
- generate_phase_plan(starting_phases, phase_lengths): Generates a phase plan based on starting 
phases and phase lengths.
- change_phase_plan(phase_tuple, cycle_time): Changes the phase plan based on a new phase tuple and
 cycle time.

These functions are used in the TrafficEnv class in the traffic_env.py file to manage the phases of
traffic lights in a traffic simulation.
"""
import numpy as np


def generate_phase(start, lengths):
    """
    Generates a phase sequence based on a starting phase and phase lengths.

    Parameters:
    start (int): The starting phase. Must be in the range [1, 4].
    Red = 1, Green = 2, Yellow = 3, RedYellow = 4.
    lengths (list of int): The lengths in seconds of each phase.

    Returns:
    np.array: The generated phase sequence.
    """
    # Define the phase sequence
    phase_sequence = np.array(['1', '2', '3', '4'])

    # Initialize the phase
    phase = []

    # Get the index of the starting phase in the phase sequence
    start_index = np.where(phase_sequence == str(start))[0][0]

    # Iterate over each length in the phase lengths
    for length in lengths:
        # Get the phase value for this length
        phase_value = phase_sequence[start_index % len(phase_sequence)]

        # Add the phase value to the phase for the specified length
        phase.extend([phase_value] * length)

        # Move to the next phase in the sequence
        start_index += 1

    return np.array(phase)


def generate_phase_plan(starting_phases, phase_lengths):
    """
    Generates a phase plan based on starting phases and phase lengths.

    :parameter starting_phases (list of int): The starting phases for each phase sequence.
    :parameter phase_lengths (list of lists of ints): The lengths of each phase for
    each phase sequence.
    :return np.array: The generated phase plan.
    """
    # Initialize the phase plan
    phase_plan = []

    # Iterate over each starting phase and corresponding phase length
    for start, lengths in zip(starting_phases, phase_lengths):
        # Generate the phase for this starting phase and phase length
        phase = generate_phase(start, lengths)

        # Add the phase to the phase plan
        phase_plan.append(phase)

    return np.array(phase_plan, dtype=object)


def change_phase_plan(phase_tuple, cycle_time):
    """
    Changes the phase plan based on a new phase tuple and cycle time.

    :parameter phase_tuple (tuple of int): The new phase tuple.
    :parameter cycle_time (int): The cycle time.

    :return np.array: The changed phase plan.
    """
    new_green_1, new_green_4 = phase_tuple
    # Itt hardcodolva vannak a zöldidők !!!
    # Phase order: 1, 2, 3, 4, 5, 6
    phase_lengths = [
        [new_green_4 + 6, 2, new_green_1, 3, cycle_time - new_green_4 - 6 - 2 - new_green_1 - 3],
        [14 + new_green_1 + new_green_4, 2, cycle_time - (new_green_4 + new_green_1 + 20), 4],
        [2, new_green_1 + new_green_4 + 7, 3, cycle_time - 2 - new_green_1 - new_green_4 - 7 - 3],
        [2, new_green_4, 3, cycle_time - 2 - new_green_4 - 3],
        [13 + new_green_1 + new_green_4, 2, cycle_time - (new_green_4 + new_green_1 + 19), 3, 1],
        [2, new_green_4, 3, cycle_time - 2 - new_green_4 - 3],
    ]
    starting_phases = [1, 1, 2, 2, 1, 2]
    new_phase_plan = generate_phase_plan(starting_phases, phase_lengths)
    return new_phase_plan

def generate_phase_combinations(min_length, max_sum, step=1):
    """
    Generates phase combinations based on a minimum length, maximum sum, and step size.

    :parameter min_length: The minimum length of a phase.
    :parameter max_sum: The maximum sum of two phases.
    :parameter step: The step size for the iteration.
    :return: A list of phase combinations.
    """
    combinations = []
    for i in range(min_length, max_sum, step):
        for j in range(min_length, max_sum, step):
            if i + j <= max_sum:
                combinations.append([i, j])
    return combinations

def get_phase_column_for_step(phase_plan, step, traci_order):
    """
    Gets the phase column for a given step based on the phase plan and Traci order.
    :param phase_plan:
    :param step:
    :param traci_order:
    :return:
    """
    # Define the dictionary to translate numbers to letters
    translate_dict = {'1': 'r', '2': 'u', '3': 'G', '4': 'y'}

    # Get the column for the current step
    column = phase_plan[:, step % phase_plan.shape[1]]

    # Initialize the reordered column
    reordered_column = np.empty_like(column, dtype=str)

    # Reorder the column according to the Traci order
    for i, order in enumerate(traci_order):
        reordered_column[order] = column[i]

    # Translate numbers to letters
    for i, value in enumerate(reordered_column):
        reordered_column[i] = translate_dict[str(value)]

    # Convert the array to a single string
    reordered_column = ''.join(reordered_column)
    return reordered_column
