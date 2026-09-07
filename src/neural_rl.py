"""neural_rl.py – Neural RL mid-layer policy stub (ASPIRE-inspired).

Tiny learned policy: obs_map → (v, ω)
- Trained via RL in sim (not included in this stub)
- Deployed as ONNX model (optional) or falls back to VFH/MPPI
- Budget-gated: skip inference if compute budget exceeded
- Safety-wrapped: outputs scaled by safety layer
- Continual learning: policy refined from execution traces

NOT shipping weights in repo → deploy .onnx separately or use fallback planner.

Integration:
- local_executive.py: add 'neural_rl' planner option
- main.py: enable with --planner neural_rl
"""

from __future__ import annotations

import time
import threading
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

# Optional ONNX runtime (graceful degradation if missing)
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None


class NeuralRLPolicy:
    """Lightweight neural policy for (v, ω) control from ego costmap.
    
    Policy interface:
    - Input: ego-space obstacle map (240x320 uint8) + goal vector (2D)
    - Output: (fwd_mps, ang_rads) continuous control
    
    Design:
    - Small CNN backbone (~50k params): ResNet-8 or MobileNetV3-Tiny
    - RL training: SAC or PPO in Isaac Sim / host-sim
    - Inference budget: <5ms @ 30Hz (allows 25ms for rest of stack)
    - Fallback: VFH or MPPI if model missing, budget exceeded, or inference fails
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        inference_budget_ms: float = 5.0,
        fallback_planner: str = "vfh"
    ):
        """Initialize neural RL policy.
        
        Args:
            model_path: Path to ONNX model file, or None to skip neural inference
            inference_budget_ms: Max inference time (ms) before fallback
            fallback_planner: Planner to use if neural fails ("vfh" or "mppi")
        """
        self._lock = threading.Lock()
        self._model_path = model_path
        self._inference_budget_ms = inference_budget_ms
        self._fallback_planner = fallback_planner
        
        self._session = None  # type: Optional[Any]
        self._input_name = None
        self._output_name = None
        
        self._inference_count = 0
        self._fallback_count = 0
        self._inference_time_ms = 0.0
        
        self._active = False
        self._goal_xy = None  # type: Optional[Tuple[float, float]]
        
        # Try to load ONNX model
        if model_path and ONNX_AVAILABLE:
            self._load_model(model_path)
        else:
            if model_path and not ONNX_AVAILABLE:
                print(f"neural_rl: ONNX runtime not available, using {fallback_planner} fallback")
            else:
                print(f"neural_rl: no model path provided, using {fallback_planner} fallback")
    
    def _load_model(self, model_path: str) -> bool:
        """Load ONNX model for inference."""
        try:
            if not Path(model_path).exists():
                print(f"neural_rl: model not found at {model_path}")
                return False
            
            # Create ONNX inference session with CPU execution provider
            # For Jetson: add 'TensorrtExecutionProvider' before CPU
            providers = ['CPUExecutionProvider']
            self._session = ort.InferenceSession(model_path, providers=providers)
            
            # Get input/output names
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            
            print(f"neural_rl: loaded model from {model_path}")
            print(f"neural_rl: input={self._input_name}, output={self._output_name}")
            return True
            
        except Exception as e:
            print(f"neural_rl: failed to load model: {e}")
            self._session = None
            return False
    
    def set_goal(self, x: float, y: float) -> None:
        """Set navigation goal in world frame (meters)."""
        with self._lock:
            self._goal_xy = (float(x), float(y))
            self._active = True
    
    def set_wander_mode(self, enable: bool) -> None:
        """Enable/disable wander mode (continuous exploration)."""
        with self._lock:
            if enable:
                self._goal_xy = None
                self._active = True
            else:
                self._active = False
    
    def cancel(self) -> None:
        """Cancel active navigation."""
        with self._lock:
            self._active = False
            self._goal_xy = None
    
    def is_active(self) -> bool:
        """Check if policy is actively driving."""
        with self._lock:
            return self._active
    
    def _preprocess_obs(
        self,
        obs_map: np.ndarray,
        pose: Tuple[float, float, float]
    ) -> np.ndarray:
        """Preprocess observation for neural network input.
        
        Args:
            obs_map: Ego-space obstacle map (H, W) uint8
            pose: (x, y, theta) world frame
        
        Returns:
            Preprocessed input (1, C, H, W) float32
        """
        # Simple preprocessing: normalize to [0, 1], add channel dim, batch dim
        obs = obs_map.astype(np.float32) / 255.0
        obs = obs[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)
        return obs
    
    def _compute_goal_vector(
        self,
        pose: Tuple[float, float, float]
    ) -> Tuple[float, float]:
        """Compute goal vector in robot frame.
        
        Args:
            pose: (x, y, theta) world frame
        
        Returns:
            (goal_x, goal_y) in robot frame (forward, left)
        """
        if self._goal_xy is None:
            return (1.0, 0.0)  # wander: forward bias
        
        px, py, ptheta = pose
        gx, gy = self._goal_xy
        
        # World frame delta
        dx = gx - px
        dy = gy - py
        
        # Rotate to robot frame
        cos_th = np.cos(ptheta)
        sin_th = np.sin(ptheta)
        
        goal_fwd = dx * cos_th + dy * sin_th
        goal_left = -dx * sin_th + dy * cos_th
        
        # Normalize
        norm = np.sqrt(goal_fwd**2 + goal_left**2)
        if norm > 1e-6:
            goal_fwd /= norm
            goal_left /= norm
        
        return (float(goal_fwd), float(goal_left))
    
    def _infer(self, obs_batch: np.ndarray) -> Optional[Tuple[float, float]]:
        """Run neural network inference.
        
        Args:
            obs_batch: Preprocessed input (1, C, H, W)
        
        Returns:
            (fwd_mps, ang_rads) or None on failure
        """
        if self._session is None:
            return None
        
        try:
            t0 = time.perf_counter()
            
            # Run inference
            outputs = self._session.run(
                [self._output_name],
                {self._input_name: obs_batch}
            )
            
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._inference_time_ms = elapsed_ms
            
            # Check budget
            if elapsed_ms > self._inference_budget_ms:
                print(f"neural_rl: budget exceeded {elapsed_ms:.1f}ms > {self._inference_budget_ms}ms")
                return None
            
            # Parse output: assume [batch, 2] with [fwd_mps, ang_rads]
            action = outputs[0][0]  # (2,)
            fwd_mps = float(action[0])
            ang_rads = float(action[1])
            
            self._inference_count += 1
            return (fwd_mps, ang_rads)
            
        except Exception as e:
            print(f"neural_rl: inference failed: {e}")
            return None
    
    def tick(
        self,
        obs_map: np.ndarray,
        pose: Tuple[float, float, float],
        dt: float
    ) -> Optional[dict]:
        """Compute control command from observation.
        
        Args:
            obs_map: Ego-space obstacle map (H, W) uint8
            pose: (x, y, theta) world frame
            dt: Time step (seconds)
        
        Returns:
            Command dict with keys: fwd_mps, ang_rads, source (neural/fallback)
            None if inactive or error
        """
        with self._lock:
            if not self._active:
                return None
            
            goal_xy = self._goal_xy
            fallback = self._fallback_planner
        
        # Try neural inference
        if self._session is not None:
            obs_batch = self._preprocess_obs(obs_map, pose)
            result = self._infer(obs_batch)
            
            if result is not None:
                fwd_mps, ang_rads = result
                return {
                    "fwd_mps": fwd_mps,
                    "ang_rads": ang_rads,
                    "source": "neural",
                    "inference_ms": self._inference_time_ms
                }
        
        # Fallback to VFH/MPPI (not implemented in stub → return None)
        # In production: import VFH or MPPI and compute fallback action
        self._fallback_count += 1
        
        print(f"neural_rl: fallback to {fallback} (not implemented in stub)")
        return None
    
    def get_debug_state(self) -> dict:
        """Return debug info for logging/UI."""
        with self._lock:
            return {
                "active": self._active,
                "goal_xy": self._goal_xy,
                "model_loaded": self._session is not None,
                "inference_count": self._inference_count,
                "fallback_count": self._fallback_count,
                "inference_ms": round(self._inference_time_ms, 2),
                "fallback_planner": self._fallback_planner
            }


# Mock ONNX model for testing (simple linear policy)
def create_mock_onnx_model(output_path: str = "models/neural_rl_mock.onnx") -> None:
    """Create a tiny mock ONNX model for testing (requires onnx package).
    
    Model: obs_map → flatten → linear(2) → tanh → (v, ω)
    NOT a real policy, just for testing inference pipeline.
    """
    try:
        import onnx
        from onnx import helper, TensorProto
        
        # Input: (1, 1, 240, 320) float32
        obs_input = helper.make_tensor_value_info(
            "obs_map", TensorProto.FLOAT, [1, 1, 240, 320]
        )
        
        # Output: (1, 2) float32
        action_output = helper.make_tensor_value_info(
            "action", TensorProto.FLOAT, [1, 2]
        )
        
        # Simple: flatten → dense(2) → tanh
        # This is a placeholder — not trainable, just for structure testing
        flatten_node = helper.make_node(
            "Flatten", ["obs_map"], ["flat"], axis=1
        )
        
        # Dense layer: W shape (76800, 2), b shape (2,)
        import numpy as np
        W = np.random.randn(76800, 2).astype(np.float32) * 0.01
        b = np.zeros(2, dtype=np.float32)
        
        W_init = helper.make_tensor("W", TensorProto.FLOAT, [76800, 2], W.flatten().tolist())
        b_init = helper.make_tensor("b", TensorProto.FLOAT, [2], b.tolist())
        
        matmul_node = helper.make_node("MatMul", ["flat", "W"], ["logits"])
        add_node = helper.make_node("Add", ["logits", "b"], ["pre_tanh"])
        tanh_node = helper.make_node("Tanh", ["pre_tanh"], ["action"])
        
        # Build graph
        graph = helper.make_graph(
            [flatten_node, matmul_node, add_node, tanh_node],
            "neural_rl_mock",
            [obs_input],
            [action_output],
            [W_init, b_init]
        )
        
        # Build model
        model = helper.make_model(graph, producer_name="kevin_aspire")
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, output_path)
        
        print(f"Mock ONNX model created: {output_path}")
        
    except ImportError:
        print("onnx package required to create mock model (pip install onnx)")


if __name__ == "__main__":
    # Demo: create mock model and test inference pipeline
    print("=== Neural RL Policy Stub Demo ===\n")
    
    # Create mock model
    # create_mock_onnx_model("models/neural_rl_mock.onnx")
    
    # Test policy (no model → fallback)
    policy = NeuralRLPolicy(
        model_path=None,
        inference_budget_ms=5.0,
        fallback_planner="vfh"
    )
    
    policy.set_goal(2.5, 1.0)
    
    # Fake observation
    obs_map = np.random.randint(0, 255, (240, 320), dtype=np.uint8)
    pose = (0.0, 0.0, 0.0)
    
    cmd = policy.tick(obs_map, pose, 0.033)
    
    if cmd is not None:
        print(f"Command: fwd={cmd['fwd_mps']:.2f} ang={cmd['ang_rads']:.2f} source={cmd['source']}")
    else:
        print("No command (fallback not implemented in stub)")
    
    print(f"\nDebug state: {policy.get_debug_state()}")
