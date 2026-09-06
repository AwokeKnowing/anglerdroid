"""Episode metrics for offline sim practice (no live drive)."""

from __future__ import annotations

from sim.world import (
    EGO_PX_SIZE,
    RCX,
    RCY,
    create_scenario,
    doorway_crossed,
)
from sim.robot import Robot
from sim.policy import create_policy


def run_episode(
    scenario: str,
    policy_name: str,
    steps: int = 400,
    dt: float = 0.033,
    apply_safety: bool = True,
    start_x: float | None = None,
    start_y: float | None = None,
    start_theta: float = 0.0,
) -> dict:
    """Run one episode; return collisions + doorway_cross (when scenario=doorway)."""
    obs, height = create_scenario(scenario)
    x0 = RCX * EGO_PX_SIZE if start_x is None else start_x
    y0 = RCY * EGO_PX_SIZE if start_y is None else start_y
    robot = Robot(x0, y0, start_theta)
    policy = create_policy(policy_name)
    policy.reset()

    crossed = False
    crossed_step = -1
    path_len = 0.0
    prev = (robot.x, robot.y)

    for step in range(steps):
        robot.update_ego_maps(obs, height)
        if robot.check_collision():
            return {
                "collisions": max(1, robot.collision_count),
                "collision_step": step,
                "final_x": robot.x,
                "final_y": robot.y,
                "final_theta": robot.theta,
                "doorway_crossed": crossed,
                "doorway_crossed_step": crossed_step,
                "path_len_m": path_len,
                "steps": step + 1,
            }

        scales = {
            "fwd": robot.safety.fwd_scale,
            "bwd": robot.safety.bwd_scale,
            "ang": robot.safety.ang_scale,
        }
        pose = {"x": robot.x, "y": robot.y, "theta": robot.theta}
        v_cmd, w_cmd = policy.act(robot.ego_obs, robot.ego_height, scales, pose)
        if policy_name == "housebot" and scales["fwd"] <= 0 and v_cmd > 0:
            raise AssertionError(f"housebot v>0 at fwd=0 step={step}")
        robot.step(v_cmd, w_cmd, dt, apply_safety=apply_safety)

        path_len += ((robot.x - prev[0]) ** 2 + (robot.y - prev[1]) ** 2) ** 0.5
        prev = (robot.x, robot.y)

        if scenario == "doorway" and not crossed and doorway_crossed(robot.x, robot.y):
            crossed = True
            crossed_step = step

        if robot.check_collision():
            return {
                "collisions": max(1, robot.collision_count),
                "collision_step": step,
                "final_x": robot.x,
                "final_y": robot.y,
                "final_theta": robot.theta,
                "doorway_crossed": crossed,
                "doorway_crossed_step": crossed_step,
                "path_len_m": path_len,
                "steps": step + 1,
            }

    return {
        "collisions": robot.collision_count,
        "collision_step": -1,
        "final_x": robot.x,
        "final_y": robot.y,
        "final_theta": robot.theta,
        "doorway_crossed": crossed,
        "doorway_crossed_step": crossed_step,
        "path_len_m": path_len,
        "steps": steps,
    }
