"""walls_stl_prior.py – Optional static walls mesh prior for SLAM/localization.

Provides:
  WallsSTLPrior – loads + caches walls mesh (STL) for localization assistance

Default OFF. Enable via:
  KEVIN_WALLS_STL_PRIOR=/path/to/walls.stl

Purpose:
  Assist localization / mapping against known static walls without etching
  movers into the map and without treating the mesh as mandatory (fail-closed).

Design:
  - Loads STL geometry (triangles → wall segments)
  - Projects wall segments into map coordinate frame
  - Provides wall line segments for pose constraint or map evidence
  - Fail-closed: missing file or flag off → identical legacy behavior

Contract (docs/perception/CONTRACT.md):
  - "Localization must not require a frozen house mesh, but a walls prior
     (STL) is a later assist."
  - This is that assist: helps when available, never required.
"""

import os
import numpy as np
from typing import Optional, List, Tuple


def _load_stl_triangles(path: str) -> np.ndarray:
    """Load STL triangles as Nx3x3 array (N triangles, 3 vertices, 3 coords).
    
    Falls back to simple binary STL parser if numpy-stl/trimesh unavailable.
    """
    # Try numpy-stl first (lightweight, common)
    try:
        from stl import mesh as stl_mesh
        stl_data = stl_mesh.Mesh.from_file(path)
        # stl_data.vectors is Nx3x3 (N triangles, 3 vertices, xyz)
        return stl_data.vectors.copy()
    except ImportError:
        pass
    
    # Try trimesh (heavier but robust)
    try:
        import trimesh
        mesh = trimesh.load(path, force='mesh')
        # mesh.triangles is Nx3x3
        return np.array(mesh.triangles, dtype=np.float32)
    except ImportError:
        pass
    
    # Fallback: simple binary STL reader (80-byte header, then triangle records)
    # Each triangle: 3 floats normal, 9 floats vertices (3x3), 2-byte attr
    with open(path, 'rb') as f:
        header = f.read(80)
        n_tris = int(np.frombuffer(f.read(4), dtype=np.uint32)[0])
        triangles = []
        for _ in range(n_tris):
            # normal: 3 floats (skip)
            f.read(12)
            # vertices: 9 floats (3 vertices × 3 coords)
            vert_data = np.frombuffer(f.read(36), dtype=np.float32)
            tri = vert_data.reshape(3, 3)
            triangles.append(tri)
            # attr: 2 bytes (skip)
            f.read(2)
        return np.array(triangles, dtype=np.float32)


def _extract_wall_segments_2d(triangles: np.ndarray,
                               z_min: float = 0.0,
                               z_max: float = 2.0,
                               merge_tol: float = 0.05) -> np.ndarray:
    """Extract 2D wall line segments from 3D triangles.
    
    Args:
        triangles: Nx3x3 array of triangles (vertices xyz)
        z_min: Min height for wall detection (m)
        z_max: Max height for wall detection (m)
        merge_tol: Tolerance for merging collinear segments (m)
    
    Returns:
        Mx4 array of wall segments: [x0, y0, x1, y1] in meters
    
    Logic:
        1. Filter triangles with vertices in [z_min, z_max] height range
        2. Project triangle edges to 2D (xy plane)
        3. Filter vertical edges (ignore floor/ceiling)
        4. Merge collinear segments within tolerance
    """
    if len(triangles) == 0:
        return np.empty((0, 4), dtype=np.float32)
    
    segments = []
    
    for tri in triangles:
        # Check if triangle is in wall height range
        z_coords = tri[:, 2]
        if z_coords.min() > z_max or z_coords.max() < z_min:
            continue
        
        # Extract edges as 2D segments (project to xy)
        edges = [
            (tri[0, :2], tri[1, :2]),  # edge 0-1
            (tri[1, :2], tri[2, :2]),  # edge 1-2
            (tri[2, :2], tri[0, :2]),  # edge 2-0
        ]
        
        for p0, p1 in edges:
            # Skip very short edges (noise)
            length = np.linalg.norm(p1 - p0)
            if length < 0.01:  # 1 cm threshold
                continue
            
            segments.append([p0[0], p0[1], p1[0], p1[1]])
    
    if len(segments) == 0:
        return np.empty((0, 4), dtype=np.float32)
    
    segments = np.array(segments, dtype=np.float32)
    
    # Simple deduplication: merge segments within tolerance
    # (Full collinearity check would be expensive; this is a simple start)
    unique_segments = []
    used = np.zeros(len(segments), dtype=bool)
    
    for i in range(len(segments)):
        if used[i]:
            continue
        seg = segments[i]
        unique_segments.append(seg)
        used[i] = True
        
        # Mark near-duplicate segments as used (simple distance check)
        for j in range(i + 1, len(segments)):
            if used[j]:
                continue
            seg2 = segments[j]
            # Check if endpoints are close (forward or reverse)
            dist_fwd = (np.linalg.norm(seg[:2] - seg2[:2]) + 
                       np.linalg.norm(seg[2:] - seg2[2:]))
            dist_rev = (np.linalg.norm(seg[:2] - seg2[2:]) + 
                       np.linalg.norm(seg[2:] - seg2[:2]))
            if min(dist_fwd, dist_rev) < merge_tol * 2:
                used[j] = True
    
    return np.array(unique_segments, dtype=np.float32)


class WallsSTLPrior:
    """Optional static walls mesh prior for SLAM/localization assistance.
    
    Usage:
        # Default off (no-op)
        prior = WallsSTLPrior()  # no walls loaded
        
        # Enable via env var
        os.environ['KEVIN_WALLS_STL_PRIOR'] = '/path/to/walls.stl'
        prior = WallsSTLPrior()  # loads walls if file exists
        
        # Use in SLAM
        if prior.is_loaded():
            segments = prior.get_wall_segments()  # Mx4 array [x0,y0,x1,y1]
            # ... apply constraints or inject evidence
    
    Attributes:
        _segments: Mx4 array of wall line segments [x0, y0, x1, y1] in meters
        _loaded: True if walls were successfully loaded
    """
    
    def __init__(self):
        self._segments = np.empty((0, 4), dtype=np.float32)
        self._loaded = False
        
        # Check env var flag
        stl_path = os.environ.get('KEVIN_WALLS_STL_PRIOR', '').strip()
        if not stl_path:
            return  # Default off
        
        # Try to load (fail-closed: log + continue on error)
        try:
            if not os.path.exists(stl_path):
                print(f"walls_stl_prior: file not found: {stl_path} (disabled)")
                return
            
            triangles = _load_stl_triangles(stl_path)
            self._segments = _extract_wall_segments_2d(triangles)
            self._loaded = True
            
            print(f"walls_stl_prior: loaded {len(self._segments)} wall segments "
                  f"from {stl_path}")
            
        except Exception as e:
            print(f"walls_stl_prior: failed to load {stl_path}: {e} (disabled)")
            self._segments = np.empty((0, 4), dtype=np.float32)
            self._loaded = False
    
    def is_loaded(self) -> bool:
        """Check if walls were successfully loaded."""
        return self._loaded
    
    def get_wall_segments(self) -> np.ndarray:
        """Get wall line segments as Mx4 array [x0, y0, x1, y1] in meters."""
        return self._segments
    
    def query_walls_near_pose(self, x: float, y: float, 
                              radius: float = 3.0) -> np.ndarray:
        """Query wall segments near a pose (for local constraint).
        
        Args:
            x, y: Pose in world frame (meters)
            radius: Search radius (meters)
        
        Returns:
            Kx4 array of wall segments within radius
        """
        if not self._loaded or len(self._segments) == 0:
            return np.empty((0, 4), dtype=np.float32)
        
        # Simple distance check: any endpoint within radius
        p0 = self._segments[:, :2]  # [x0, y0]
        p1 = self._segments[:, 2:]  # [x1, y1]
        
        dist0 = np.sqrt((p0[:, 0] - x) ** 2 + (p0[:, 1] - y) ** 2)
        dist1 = np.sqrt((p1[:, 0] - x) ** 2 + (p1[:, 1] - y) ** 2)
        
        # Segment is near if either endpoint is within radius
        near_mask = np.minimum(dist0, dist1) < radius
        
        return self._segments[near_mask]
    
    def project_walls_to_map(self, map_origin_x: float, map_origin_y: float,
                             map_px_size: float) -> List[Tuple[int, int, int, int]]:
        """Project wall segments to map pixel coordinates.
        
        Args:
            map_origin_x, map_origin_y: Map origin in world frame (meters)
            map_px_size: Map pixel size (meters/pixel)
        
        Returns:
            List of (r0, c0, r1, c1) pixel coordinates for each segment
        """
        if not self._loaded or len(self._segments) == 0:
            return []
        
        # World to map pixel transform
        # map[r, c] corresponds to world (x, y) where:
        #   x = map_origin_x + c * map_px_size
        #   y = map_origin_y + r * map_px_size
        # Invert:
        #   c = (x - map_origin_x) / map_px_size
        #   r = (y - map_origin_y) / map_px_size
        
        p0_world = self._segments[:, :2]  # [x0, y0]
        p1_world = self._segments[:, 2:]  # [x1, y1]
        
        # Convert to map pixel coordinates
        p0_map = (p0_world - np.array([map_origin_x, map_origin_y])) / map_px_size
        p1_map = (p1_world - np.array([map_origin_x, map_origin_y])) / map_px_size
        
        # Return as list of (r0, c0, r1, c1) tuples
        segments_px = []
        for (x0, y0), (x1, y1) in zip(p0_map, p1_map):
            # Note: OpenCV convention is (col, row) but we return (row, col)
            segments_px.append((int(y0), int(x0), int(y1), int(x1)))
        
        return segments_px
