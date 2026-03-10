import math
import timeit

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

def updated_code(current_v1, current_v2, steps, step_by, min, max):
    step_max = (steps * step_by) - step_by
    i = -(step_max - step_by)
    vels = []

    while not math.isclose(i, step_max):
        j = -steps * step_by
        while not math.isclose(j, step_max):
            vh1, vh2 = current_v1 + i, current_v2 + j
            if min <= vh1 <= max and min <= vh2 <= max:
                vels.append((vh1, vh2))
            j += step_by
        i += step_by

    return vels

current_v1 = 0.5
current_v2 = 0.5
steps = 4
step_by = 0.02
min_value = -1.0
max_value = 1.0

# Time the original code
original_time = timeit.timeit(lambda: original_code(current_v1, current_v2, steps, step_by, min_value, max_value), number=10000)

# Time the updated code
updated_time = timeit.timeit(lambda: updated_code(current_v1, current_v2, steps, step_by, min_value, max_value), number=10000)

print("Original Code Time (ms):", original_time * 1000)
print("Updated Code Time (ms):", updated_time * 1000)

vels = updated_code(current_v1, current_v2, steps, step_by, min_value, max_value)
for [v1,v2] in vels:
  print(v1,v2)

