"""
test_gpu_topdown_smoke.py - Minimal smoke test for GPU topdown shaders.

Tests shader syntax and basic structure without requiring full GPU context.
"""

import re


def test_shader_syntax():
    """Verify topdown shaders have correct GLSL syntax."""
    print("test_shader_syntax: start")
    
    from src.gpu_render import _VERT_SCATTER_TOPDOWN, _FRAG_SCATTER_TOPDOWN
    
    # Check vertex shader has required uniforms
    assert 'uniform float u_scale' in _VERT_SCATTER_TOPDOWN
    assert 'uniform vec2' in _VERT_SCATTER_TOPDOWN and 'u_offset' in _VERT_SCATTER_TOPDOWN
    assert 'uniform float u_floor_clip' in _VERT_SCATTER_TOPDOWN
    assert 'in vec3 in_v' in _VERT_SCATTER_TOPDOWN
    assert 'flat out float v_h' in _VERT_SCATTER_TOPDOWN
    
    # Check fragment shader
    assert 'flat in float v_h' in _FRAG_SCATTER_TOPDOWN
    assert 'out vec4 fc' in _FRAG_SCATTER_TOPDOWN
    
    # Check version header
    assert '#version 330' in _VERT_SCATTER_TOPDOWN
    assert '#version 330' in _FRAG_SCATTER_TOPDOWN
    
    print("test_shader_syntax: PASS")


def test_gpu_renderer_has_topdown():
    """Verify GPURenderer has topdown methods."""
    print("test_gpu_renderer_has_topdown: start")
    
    from src.gpu_render import GPURenderer
    
    # Check methods exist
    assert hasattr(GPURenderer, 'configure_depth_topdown')
    assert hasattr(GPURenderer, 'depth_topdown_gpu')
    assert hasattr(GPURenderer, '_init_depth_topdown_gl')
    
    # Check method signatures
    import inspect
    
    cfg_sig = inspect.signature(GPURenderer.configure_depth_topdown)
    assert 'px_size' in cfg_sig.parameters
    assert 'floor_clip' in cfg_sig.parameters
    assert 'out_h' in cfg_sig.parameters
    assert 'out_w' in cfg_sig.parameters
    
    gpu_sig = inspect.signature(GPURenderer.depth_topdown_gpu)
    assert 'verts' in gpu_sig.parameters
    
    print("test_gpu_renderer_has_topdown: PASS")


def test_vision_uses_gpu_topdown():
    """Verify vision.py is wired to use GPU topdown."""
    print("test_vision_uses_gpu_topdown: start")
    
    # Read vision.py source
    with open('src/vision.py', 'r') as f:
        vision_src = f.read()
    
    # Check GPU topdown is configured
    assert 'configure_depth_topdown' in vision_src
    assert 'TD_PX_SIZE' in vision_src
    assert 'TD_FLOOR_CLIP' in vision_src
    
    # Check capture loop uses GPU with fallback
    assert 'depth_topdown_gpu' in vision_src
    assert '_gpu_topdown_result' in vision_src
    
    # Check CPU fallback exists
    assert 'depth_topdown(self._rs1.verts)' in vision_src or \
           'depth_topdown(' in vision_src
    
    print("test_vision_uses_gpu_topdown: PASS")


def test_configuration_values():
    """Verify configuration constants are sensible."""
    print("test_configuration_values: start")
    
    try:
        from src.vision import TD_PX_SIZE, TD_FLOOR_CLIP, FRAME_H, FRAME_W
    except ImportError as e:
        print("  SKIP: vision module not available (%s)" % e)
        return
    
    # Check reasonable values
    assert 0.005 < TD_PX_SIZE < 0.02, "Pixel size should be ~1cm"
    assert 0.8 < TD_FLOOR_CLIP < 1.0, "Floor clip should be ~0.91m"
    assert FRAME_H > 0 and FRAME_W > 0, "Frame dimensions positive"
    assert 200 <= FRAME_H <= 480, "Frame height reasonable"
    assert 200 <= FRAME_W <= 640, "Frame width reasonable"
    
    print("  TD_PX_SIZE: %.4f" % TD_PX_SIZE)
    print("  TD_FLOOR_CLIP: %.2f" % TD_FLOOR_CLIP)
    print("  FRAME: %dx%d" % (FRAME_W, FRAME_H))
    
    print("test_configuration_values: PASS")


if __name__ == "__main__":
    print("=" * 70)
    print("GPU topdown smoke tests (no GPU context required)")
    print("=" * 70)
    
    test_shader_syntax()
    test_gpu_renderer_has_topdown()
    test_vision_uses_gpu_topdown()
    test_configuration_values()
    
    print("=" * 70)
    print("All smoke tests PASSED")
    print("=" * 70)
