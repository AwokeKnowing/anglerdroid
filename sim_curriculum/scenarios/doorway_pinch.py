"""sim_curriculum/scenarios/doorway_pinch.py — Narrow passage scenario (stub).

Hand-coded baseline scenario for doorway navigation.
Future: FM generates variants with different widths, angles, and clutter.
"""


class DoorwayPinchScenario:
    """Narrow passage requiring precise turning and clearance management."""
    
    def __init__(self, width_cm: float = 50.0, angle_deg: float = 0.0):
        """Initialize doorway pinch scenario.
        
        Args:
            width_cm: Doorway width in cm (robot width = 42cm)
            angle_deg: Approach angle in degrees (0 = straight-on)
        """
        self.width_cm = width_cm
        self.angle_deg = angle_deg
        
        # Robot geometry (from robot_config.py)
        self.robot_width_cm = 42.0  # side-to-side
        self.robot_length_cm = 30.0  # front-back
        
        self.clearance_cm = width_cm - self.robot_width_cm
    
    def reset(self):
        """Reset scenario to initial state."""
        raise NotImplementedError("Host-sim integration required")
    
    def step(self, action):
        """Execute one step with given action."""
        raise NotImplementedError("Host-sim integration required")
    
    def get_reward(self, state, action, next_state):
        """Compute reward for transition."""
        # Goal reached: +10
        # Collision: -10
        # Time penalty: -0.01 per step
        # Clearance bonus: +clearance_margin (encourages centering)
        raise NotImplementedError("Reward function not implemented")
    
    def is_terminal(self, state):
        """Check if episode is complete (success or failure)."""
        raise NotImplementedError("Host-sim integration required")


if __name__ == "__main__":
    print("Doorway Pinch Scenario (stub)")
    print("NOT IMPLEMENTED: Requires host-sim integration")
    print()
    scenario = DoorwayPinchScenario(width_cm=50.0, angle_deg=0.0)
    print(f"Doorway width: {scenario.width_cm}cm")
    print(f"Robot width: {scenario.robot_width_cm}cm")
    print(f"Clearance: {scenario.clearance_cm}cm")
