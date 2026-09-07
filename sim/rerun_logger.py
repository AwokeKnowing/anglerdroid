"""Rerun logging for host simulator visualization.

Logs sim state to RRD file for post-run inspection.
Compatible with Rerun 0.37 API (matches src/rerun_log.py).
"""

import numpy as np

try:
    import rerun as rr
    import rerun.blueprint as rrb
    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False


class SimRerunLogger:
    """Log simulator state to Rerun.
    
    Entities (matching live layout from src/rerun_log.py):
      - world/obs_map: Ego-space obstacle map
      - world/height_map: Height map
      - world/robot: Robot pose + footprint
      - world/trajectory: Past positions
      - sensors/rgb: Simulated webcam (stub)
      - autonomy/goal: Current goal waypoint
    """
    
    def __init__(self, recording_id='host_sim'):
        """Initialize Rerun logger.
        
        Args:
            recording_id: Recording ID for Rerun
        """
        if not RERUN_AVAILABLE:
            raise ImportError("Rerun SDK not available. Install with: pip install rerun-sdk")
        
        self.recording_id = recording_id
        
        # Initialize recording
        rr.init(recording_id, spawn=False)
        
        # Blueprint (optional): Set up default view layout
        self._setup_blueprint()
        
    def _setup_blueprint(self):
        """Configure Rerun view layout."""
        try:
            # Create a simple 2-column layout
            blueprint = rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Spatial2DView(origin="world", name="Top-down"),
                    rrb.Vertical(
                        rrb.TextDocumentView(origin="info", name="Info"),
                        rrb.TimeSeriesView(origin="metrics", name="Metrics"),
                    ),
                ),
            )
            rr.send_blueprint(blueprint)
        except Exception as e:
            # Blueprint API can be finicky; continue without it
            print(f"Warning: Could not set Rerun blueprint: {e}")
    
    def log_frame(self, step, obs_map, height_map, robot_pose, trail, rgb_frame=None):
        """Log a single simulation frame.
        
        Args:
            step: Simulation step number
            obs_map: (H, W) uint8 ego-space obstacle map
            height_map: (H, W) uint8 ego-space height map
            robot_pose: dict with 'x', 'y', 'theta'
            trail: list of (x, y) positions
            rgb_frame: Optional (H, W, 3) RGB frame
        """
        # Set time
        rr.set_time_sequence("step", step)
        
        # Log obstacle map as image
        rr.log(
            "world/obs_map",
            rr.Image(obs_map),
        )
        
        # Log height map as image (with color map)
        rr.log(
            "world/height_map",
            rr.DepthImage(height_map.astype(np.float32)),
        )
        
        # Log robot pose as 2D point + heading
        x, y, theta = robot_pose['x'], robot_pose['y'], robot_pose['theta']
        rr.log(
            "world/robot/position",
            rr.Points2D([[x, y]], radii=[0.15]),
        )
        
        # Robot heading as arrow
        import math
        dx = 0.3 * math.cos(theta)
        dy = 0.3 * math.sin(theta)
        rr.log(
            "world/robot/heading",
            rr.Arrows2D(
                origins=[[x, y]],
                vectors=[[dx, dy]],
            ),
        )
        
        # Log trajectory
        if len(trail) > 1:
            trail_array = np.array(trail, dtype=np.float32)
            rr.log(
                "world/trajectory",
                rr.LineStrips2D([trail_array]),
            )
        
        # Log RGB (if available)
        if rgb_frame is not None:
            rr.log(
                "sensors/rgb",
                rr.Image(rgb_frame),
            )
        
        # Log text info
        info_text = f"Step: {step}\nPos: ({x:.2f}, {y:.2f})\nθ: {math.degrees(theta):.1f}°"
        rr.log(
            "info",
            rr.TextDocument(info_text),
        )
    
    def save(self, path):
        """Save recording to RRD file.
        
        Args:
            path: Path to save .rrd file
        """
        rr.save(path)
        print(f"Saved Rerun recording: {path}")
