#!/usr/bin/env python3
"""Generate a tiny synthetic walls STL fixture for testing.

Creates a simple 3m × 2m rectangular room (4 walls).
"""

import numpy as np
import struct


def write_binary_stl(path, triangles):
    """Write triangles to binary STL file.
    
    Args:
        path: Output file path
        triangles: List of triangle vertex arrays, each (3, 3) shape
    """
    with open(path, 'wb') as f:
        # 80-byte header
        f.write(b'Generated test walls fixture' + b'\x00' * (80 - 28))
        
        # Number of triangles (uint32)
        f.write(struct.pack('<I', len(triangles)))
        
        # Each triangle: normal (3 floats), vertices (9 floats), attr (uint16)
        for tri in triangles:
            # Compute normal (cross product of edges)
            v0, v1, v2 = tri
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-8:
                normal = normal / norm_len
            else:
                normal = np.array([0.0, 0.0, 1.0])
            
            # Write normal
            f.write(struct.pack('<fff', *normal))
            
            # Write vertices
            for v in tri:
                f.write(struct.pack('<fff', *v))
            
            # Write attribute (unused, 0)
            f.write(struct.pack('<H', 0))


def make_rectangle_wall(x0, y0, x1, y1, z0=0.0, z1=2.0):
    """Create two triangles forming a vertical rectangular wall.
    
    Args:
        x0, y0: Start point (meters)
        x1, y1: End point (meters)
        z0: Bottom height (meters)
        z1: Top height (meters)
    
    Returns:
        List of two triangles (each 3x3 array)
    """
    # Four corners of the wall rectangle
    p0 = np.array([x0, y0, z0])
    p1 = np.array([x1, y1, z0])
    p2 = np.array([x1, y1, z1])
    p3 = np.array([x0, y0, z1])
    
    # Two triangles (CCW winding)
    tri1 = np.array([p0, p1, p2])
    tri2 = np.array([p0, p2, p3])
    
    return [tri1, tri2]


def generate_test_walls_stl(output_path):
    """Generate a simple 3m × 2m rectangular room (4 walls)."""
    
    # Room dimensions (meters)
    room_w = 3.0  # x-axis
    room_h = 2.0  # y-axis
    wall_height = 2.0  # z-axis
    
    triangles = []
    
    # Wall 1: back wall (x = 0)
    triangles.extend(make_rectangle_wall(0.0, 0.0, 0.0, room_h, 0.0, wall_height))
    
    # Wall 2: right wall (y = room_h)
    triangles.extend(make_rectangle_wall(0.0, room_h, room_w, room_h, 0.0, wall_height))
    
    # Wall 3: front wall (x = room_w)
    triangles.extend(make_rectangle_wall(room_w, room_h, room_w, 0.0, 0.0, wall_height))
    
    # Wall 4: left wall (y = 0)
    triangles.extend(make_rectangle_wall(room_w, 0.0, 0.0, 0.0, 0.0, wall_height))
    
    write_binary_stl(output_path, triangles)
    print(f"Generated {len(triangles)} triangles (4 walls) → {output_path}")


if __name__ == '__main__':
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'walls_test.stl')
    generate_test_walls_stl(output_path)
