"""sim_curriculum/rewards/base_reward.py — Common reward components (stub).

Shared reward shaping for all curriculum scenarios.
"""


def goal_reached_reward(distance_to_goal_m: float, threshold_m: float = 0.35) -> float:
    """Reward for reaching goal.
    
    Args:
        distance_to_goal_m: Current distance to goal (meters)
        threshold_m: Goal reached threshold (default: 0.35m from local_executive)
    
    Returns:
        +10.0 if goal reached, 0.0 otherwise
    """
    return 10.0 if distance_to_goal_m <= threshold_m else 0.0


def collision_penalty(collision_detected: bool) -> float:
    """Penalty for collision.
    
    Args:
        collision_detected: True if collision occurred
    
    Returns:
        -10.0 if collision, 0.0 otherwise
    """
    return -10.0 if collision_detected else 0.0


def time_penalty(elapsed_steps: int, penalty_per_step: float = 0.01) -> float:
    """Penalty for time/steps taken.
    
    Args:
        elapsed_steps: Number of steps taken so far
        penalty_per_step: Penalty per step (default: 0.01)
    
    Returns:
        Cumulative time penalty (negative)
    """
    return -elapsed_steps * penalty_per_step


def clearance_bonus(min_clearance_m: float, safe_clearance_m: float = 0.15) -> float:
    """Bonus for maintaining safe clearance from obstacles.
    
    Args:
        min_clearance_m: Minimum clearance to any obstacle (meters)
        safe_clearance_m: Safe clearance threshold (default: 15cm)
    
    Returns:
        Bonus scaled by clearance margin
    """
    if min_clearance_m >= safe_clearance_m:
        return 1.0
    elif min_clearance_m > 0:
        return min_clearance_m / safe_clearance_m
    else:
        return 0.0


def smoothness_bonus(angular_velocity_rads: float, max_omega: float = 1.0) -> float:
    """Bonus for smooth motion (low angular velocity).
    
    Args:
        angular_velocity_rads: Absolute angular velocity (rad/s)
        max_omega: Max angular velocity for full penalty (default: 1.0 rad/s)
    
    Returns:
        Smoothness bonus (0.0-1.0)
    """
    return max(0.0, 1.0 - abs(angular_velocity_rads) / max_omega)


def safety_scale_penalty(fwd_scale: float, threshold: float = 0.5) -> float:
    """Penalty for aggressive motion under safety throttling.
    
    Args:
        fwd_scale: Forward velocity scale from safety layer (0.0-1.0)
        threshold: Throttling threshold (default: 0.5)
    
    Returns:
        Penalty for motion while heavily throttled
    """
    if fwd_scale < threshold:
        # Penalize attempting to move when safety is limiting
        return -(threshold - fwd_scale) * 0.5
    return 0.0


if __name__ == "__main__":
    print("=== Base Reward Components (stub) ===")
    print()
    print("Example rewards:")
    print(f"  Goal reached (0.1m): {goal_reached_reward(0.1):.2f}")
    print(f"  Collision: {collision_penalty(True):.2f}")
    print(f"  Time penalty (100 steps): {time_penalty(100):.2f}")
    print(f"  Clearance (0.2m): {clearance_bonus(0.2):.2f}")
    print(f"  Smoothness (0.3 rad/s): {smoothness_bonus(0.3):.2f}")
    print(f"  Safety scale (0.3): {safety_scale_penalty(0.3):.2f}")
