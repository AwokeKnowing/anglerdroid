import unittest
import math

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

def optimized_code(current_v1, current_v2, steps, step_by, min, max):
    min_limit = min
    max_limit = max

    step_by *= 1.00001
    vels = []

    for i in range(-steps, steps + 1):
        vh1 = current_v1 + i * step_by
        if min_limit <= vh1 <= max_limit:
            for j in range(-steps, steps + 1):
                vh2 = current_v2 + j * step_by
                if min_limit <= vh2 <= max_limit:
                    vels.append((vh1, vh2))

    return vels

class TestCodeEquivalence(unittest.TestCase):

    def test_equivalence(self):
        current_v1 = 0.5
        current_v2 = 0.5
        steps = 10
        step_by = 0.1
        min_value = 0.2
        max_value = 0.8

        original_result = original_code(current_v1, current_v2, steps, step_by, min_value, max_value)
        optimized_result = optimized_code(current_v1, current_v2, steps, step_by, min_value, max_value)

        # Check that each pair of values is within a small tolerance
        for orig, opt in zip(original_result, optimized_result):
            self.assertAlmostEqual(orig[0], opt[0], places=9)
            self.assertAlmostEqual(orig[1], opt[1], places=9)

if __name__ == '__main__':
    unittest.main()
