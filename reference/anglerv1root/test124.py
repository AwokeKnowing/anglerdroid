import math
import numpy as np

def original_code(current_v1, current_v2, steps, step_by, min, max):
    i = -steps * step_by
    vels = []

    while not math.isclose(i, (steps * step_by) - step_by):
        j = -steps * step_by
        while j < steps * step_by + 0.00001:
            vh1, vh2 = current_v1 + i, current_v2 + j
            if min <= vh1 <= max and min <= vh2 <= max:
                vels.append((vh1, vh2))
            j += step_by
        i += step_by

    return vels

def vectorized_solution(current_v1, current_v2, steps, step_by, min_val, max_val):
    vh1_values = np.arange(current_v1 - steps * step_by, current_v1 + (steps + 1) * step_by, step_by)
    vh2_values = np.arange(current_v2 - steps * step_by, current_v2 + (steps + 1) * step_by, step_by)
    vh1, vh2 = np.meshgrid(vh1_values, vh2_values)
    mask = (min_val <= vh1) & (vh1 <= max_val) & (min_val <= vh2) & (vh2 <= max_val)
    vels = [(vh1[i, j], vh2[i, j]) for i in range(mask.shape[0]) for j in range(mask.shape[1]) if mask[i, j]]
    return vels

current_v1 = 0.5
current_v2 = 0.5
steps = 10
step_by = 0.1
min_value = 0.2
max_value = 0.8

original_result = original_code(current_v1, current_v2, steps, step_by, min_value, max_value)
vectorized_result = vectorized_solution(current_v1, current_v2, steps, step_by, min_value, max_value)

print("Original Code Result:")
for val in original_result:
    print(val)

print("\nVectorized Solution Result:")
for val in vectorized_result:
    print(val)
