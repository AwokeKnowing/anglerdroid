# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

"""
OpenCV and Numpy Point cloud Software Renderer

This sample is mostly for demonstration and educational purposes.
It really doesn't offer the quality or performance that can be
achieved with hardware acceleration.

Usage:
------
Mouse: 
    Drag with left button to rotate around pivot (thick small axes), 
    with right button to translate and the wheel to zoom.

Keyboard: 
    [p]     Pause
    [r]     Reset View
    [d]     Cycle through decimation values
    [z]     Toggle point scaling
    [c]     Toggle color source
    [s]     Save PNG (./out.png)
    [e]     Export points to ply (./out.ply)
    [q\ESC] Quit
"""

import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs

#import numba  
#import numba.cuda


from cuda import cudart

import cupy as cp

import pyrr
import glfw

from OpenGL.GL import *  # noqa F403
import OpenGL.GL.shaders



def format_cudart_err(err):
    return (
        f"{cudart.cudaGetErrorName(err)[1].decode('utf-8')}({int(err)}): "
        f"{cudart.cudaGetErrorString(err)[1].decode('utf-8')}"
    )


def check_cudart_err(args):
    if isinstance(args, tuple):
        assert len(args) >= 1
        err = args[0]
        if len(args) == 1:
            ret = None
        elif len(args) == 2:
            ret = args[1]
        else:
            ret = args[1:]
    else:
        err = args
        ret = None

    assert isinstance(err, cudart.cudaError_t), type(err)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(format_cudart_err(err))

    return ret


class CudaOpenGLMappedBuffer:
    def __init__(self, gl_buffer, flags=0):
        self._gl_buffer = int(gl_buffer)
        self._flags = int(flags)

        self._graphics_ressource = None
        self._cuda_buffer = None

        self.register()

    @property
    def gl_buffer(self):
        return self._gl_buffer

    @property
    def cuda_buffer(self):
        assert self.mapped
        return self._cuda_buffer

    @property
    def graphics_ressource(self):
        assert self.registered
        return self._graphics_ressource

    @property
    def registered(self):
        return self._graphics_ressource is not None

    @property
    def mapped(self):
        return self._cuda_buffer is not None

    def __enter__(self):
        return self.map()

    def __exit__(self, exc_type, exc_value, trace):
        self.unmap()
        return False

    def __del__(self):
        self.unregister()

    def register(self):
        if self.registered:
            return self._graphics_ressource
        self._graphics_ressource = check_cudart_err(
            cudart.cudaGraphicsGLRegisterBuffer(self._gl_buffer, self._flags)
        )
        return self._graphics_ressource

    def unregister(self):
        if not self.registered:
            return self
        self.unmap()
        self._graphics_ressource = check_cudart_err(
            cudart.cudaGraphicsUnregisterResource(self._graphics_ressource)
        )
        return self

    def map(self, stream=None):
        if not self.registered:
            raise RuntimeError("Cannot map an unregistered buffer.")
        if self.mapped:
            return self._cuda_buffer

        check_cudart_err(
            cudart.cudaGraphicsMapResources(1, self._graphics_ressource, stream)
        )

        ptr, size = check_cudart_err(
            cudart.cudaGraphicsResourceGetMappedPointer(self._graphics_ressource)
        )

        self._cuda_buffer = cp.cuda.MemoryPointer(
            cp.cuda.UnownedMemory(ptr, size, self), 0
        )

        return self._cuda_buffer

    def unmap(self, stream=None):
        if not self.registered:
            raise RuntimeError("Cannot unmap an unregistered buffer.")
        if not self.mapped:
            return self

        self._cuda_buffer = check_cudart_err(
            cudart.cudaGraphicsUnmapResources(1, self._graphics_ressource, stream)
        )

        return self


class CudaOpenGLMappedArray(CudaOpenGLMappedBuffer):
    def __init__(self, dtype, shape, gl_buffer, flags=0, strides=None, order='C'):
        super().__init__(gl_buffer, flags)
        self._dtype = dtype
        self._shape = shape
        self._strides = strides
        self._order = order

    @property
    def cuda_array(self):
        assert self.mapped
        return cp.ndarray(
            shape=self._shape,
            dtype=self._dtype,
            strides=self._strides,
            order=self._order,
            memptr=self._cuda_buffer,
        )

    def map(self, *args, **kwargs):
        super().map(*args, **kwargs)
        return self.cuda_array

    




VERTEX_SHADER = """
#version 330

in vec3 position;

uniform mat4 transform;

void main() {
    gl_Position = transform * vec4(position, 1.0f);
}
"""


FRAGMENT_SHADER = """
#version 330

out vec4 outColor;

void main() {
    outColor = vec4(0.0f, 0.7f, 0.0f, 1.0f);
}
"""

class PointCloudGL:
    def __init__(self, max_vertices=1000000, point_size=2.0):
        self.max_vertices = max_vertices

        self.shader = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
        )
        self.positionLoc = glGetAttribLocation(self.shader, "position")
        self.transformLoc = glGetUniformLocation(self.shader, "transform")

        glUseProgram(self.shader)
        glEnable(GL_DEPTH_TEST)
        glPointSize(point_size)

        self.cuda_vertex_buffer = self.setup_buffers(self.max_vertices)

    def setup_buffers(self, max_vertices):

        ftype = np.float32

        vertex_bytes = 3 * max_vertices * ftype().nbytes
        flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard

        VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, vertex_bytes, None, GL_DYNAMIC_DRAW)
        vertex_buffer = CudaOpenGLMappedArray(ftype, (max_vertices, 3), VBO, flags)

        return vertex_buffer
    
    def draw(self, num_points, width, height):
        glViewport(0, 0, width, height)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        rot_x = pyrr.Matrix44.from_x_rotation(0)
        rot_y = pyrr.Matrix44.from_y_rotation(0)

        glUniformMatrix4fv(self.transformLoc, 1, GL_FALSE, rot_x * rot_y)

        glBindBuffer(GL_ARRAY_BUFFER, self.cuda_vertex_buffer.gl_buffer)

        glEnableVertexAttribArray(self.positionLoc)
        glVertexAttribPointer(self.positionLoc, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glDrawArrays(GL_POINTS, 0, num_points)

    def __del__(self):
        self.cuda_vertex_buffer.unregister()

    

class WindowsGLFW:
    def __init__(self, width=640, height=480, title="GL Window", aspect_w=None, aspect_h=None, fps=False):
        if not glfw.init(): 
            return
        
        self.fps = fps
        self.winCount = 0
        self.fps_stats = {}
        self.windows = {}

        win = self.create_window('main',width, height, title, aspect_w, aspect_h)
               
        #shortcut to main glfw window
        self.active = 'main'
        self.win = win
        self.size = (width, height)
        self.width = width
        self.height = height
        

    def title(self,title=None,win_name=None,store=True):
        if win_name is None:
            win_name = self.active

        window = self.windows[win_name]['win']

        if title is None:
            return glfw.get_window_title(window)
        else:
            glfw.set_window_title(window,title)
            if store:
                self.windows[win_name]['title'] = title


    def create_window(self, name,width=640, height=480, title="GL Window", aspect_w=None,aspect_h=None):
        if aspect_w is None: 
            aspect_w = width
        if aspect_h is None:
            aspect_h = height
        
        window = glfw.create_window(width, height, title, None, None)
        self.winCount += 1
        glfw.set_window_user_pointer(window, self.winCount) # used to compare. eg see onSize below
        glfw.set_window_aspect_ratio(window, aspect_w,aspect_h)
        glfw.make_context_current(window)
        glfw.swap_interval(0)
        
        self.fps_stats[name] = {'last_time':self.time(),'frames':0}

        def onSize(window,w,h):
            obj = glfw.get_window_user_pointer(window)
            for k, v in self.windows.items():
                obj2=glfw.get_window_user_pointer(v['win'])
                if not obj is obj2:
                    continue

                v['size']=(w,h)
                if self.win is v['win']:
                    self.size=(w,h)
                    self.width = w
                    self.height = h
                break

        glfw.set_window_size_callback(window, onSize)

        self.windows[name] = {'size':(width, height), 'win': window,'title':title, 'fps':60}

        return window
    
    
    def closed(self, win_name=None):
        if win_name is None:
            return glfw.window_should_close(self.win)
        else:
            return glfw.window_should_close(self.windows[win_name]['win'])
        
    
    def time(self):
        return glfw.get_time()
    
    
    def activate(self,win_name):
        w = self.windows[win_name]

        self.active = win_name
        self.win = w.win
        self.size=self.size = w.size 
        self.width,self.height = self.size 


    def update_fps(self):
        stat = self.fps_stats[self.active]
        t = glfw.get_time()
        dt = t - stat['last_time']
        stat['frames'] += 1
        if dt >= 1.0:
            win = self.windows[self.active]    
            win['fps'] = stat['frames'] / dt
            stat['last_time'] = t
            stat['frames'] = 0
            self.title(f"{win['title']} ({win['fps']:.1f} fps)", store=False)

    
    def swap(self):
        if self.fps:
            self.update_fps()

        glfw.swap_buffers(self.win)
        glfw.poll_events()

    
    def __del__(self):
        try:
            glfw.terminate()
        except TypeError:
            pass





class AppState:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(0), math.radians(0)
        self.translation = np.array([0, 0, 0], dtype=np.float32)
        self.distance = 0.0
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = False
        self.color = True
        self.color_scale = 1.0*10
        self.yrot = 0
        self.obst_px_size = .02
        self.obst_canvas_size = (448//1, 448//1)

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 0
        self.translation[:] = 0, 0, 0

    @property
    def rotation(self):
        #print("pitch",math.degrees(self.pitch),"yaw",math.degrees(self.yaw))
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        Rz, _ = cv2.Rodrigues((0, 0, math.radians(0)))
        return np.dot(Rx, Rz, Ry).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)


state = AppState()

# Configure depth and color streams
pipeline_td = rs.pipeline()
config_td = rs.config()
config_td.enable_device("815412070676")

pipeline_wrapper_td = rs.pipeline_wrapper(pipeline_td)
pipeline_profile_td = config_td.resolve(pipeline_wrapper_td)
device_td = pipeline_profile_td.get_device()

config_td.enable_stream(rs.stream.depth, 848,480,rs.format.z16, 60)
config_td.enable_stream(rs.stream.color, 424,240,rs.format.bgr8, 60)

# Start streaming
pipeline_td.start(config_td)

# Get stream profile and camera intrinsics
profile_td = pipeline_td.get_active_profile()
depth_profile_td = rs.video_stream_profile(profile_td.get_stream(rs.stream.depth))


# Configure depth and color streams
pipeline_fw = rs.pipeline()
config_fw = rs.config()
config_fw.enable_device("944622074292")

pipeline_wrapper_fw = rs.pipeline_wrapper(pipeline_fw)
pipeline_profile_fw = config_fw.resolve(pipeline_wrapper_fw)
device_fw = pipeline_profile_fw.get_device()

config_fw.enable_stream(rs.stream.depth, 424,240,rs.format.z16, 60)
config_fw.enable_stream(rs.stream.color, 424,240,rs.format.bgr8, 60)

# Start streaming
pipeline_fw.start(config_fw)

# Get stream profile and camera intrinsics
profile_fw = pipeline_fw.get_active_profile()
depth_profile_fw = rs.video_stream_profile(profile_fw.get_stream(rs.stream.depth))


# Processing blocks
pc_td = rs.pointcloud()
decimate_td = rs.decimation_filter()
decimate_td.set_option(rs.option.filter_magnitude, 2 )

# Processing blocks
pc_fw = rs.pointcloud()
decimate_fw = rs.decimation_filter()
decimate_fw.set_option(rs.option.filter_magnitude, 4)



def mouse_cb(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN: state.mouse_btns[0] = True
    if event == cv2.EVENT_LBUTTONUP:   state.mouse_btns[0] = False
    if event == cv2.EVENT_RBUTTONDOWN: state.mouse_btns[1] = True
    if event == cv2.EVENT_RBUTTONUP:   state.mouse_btns[1] = False
    if event == cv2.EVENT_MBUTTONDOWN: state.mouse_btns[2] = True
    if event == cv2.EVENT_MBUTTONUP:   state.mouse_btns[2] = False

    if event == cv2.EVENT_MOUSEMOVE:

        h, w = out.shape[:2]
        dx, dy = x - state.prev_mouse[0], y - state.prev_mouse[1]

        if state.mouse_btns[0]:
            state.yaw += float(dx) / w * 2
            state.pitch -= float(dy) / h * 2
            print("yaw pitch",state.yaw,state.pitch)

        elif state.mouse_btns[1]:
            dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
            state.translation -= np.dot(state.rotation, dp)

        elif state.mouse_btns[2]:
            dz = math.sqrt(dx**2 + dy**2) * math.copysign(0.01, -dy)
            state.translation[2] += dz
            state.distance -= dz

        #print(state.translation)

    if event == cv2.EVENT_MOUSEWHEEL:
        dz = math.copysign(0.1, flags)
        state.translation[2] += dz
        state.distance -= dz

    state.prev_mouse = (x, y)


cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow(state.WIN_NAME, state.obst_canvas_size[1], state.obst_canvas_size[0])
cv2.setMouseCallback(state.WIN_NAME, mouse_cb)


tnow = time.time()
times={}
def tstep(t):
    global times, tnow
    nt=time.time()
    times[t]=(nt-tnow)*1000
    tnow=nt



def rodrigues(vec_xyz):
    theta = np.linalg.norm(vec_xyz)
    k = vec_xyz / theta if theta > 1e-15 else np.zeros(3)
    k_cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * k_cross + (1 - np.cos(theta)) * (k_cross @ k_cross)

def rvec(rx, ry, rz):
    Rx = rodrigues(np.array([math.radians(rx), 0, 0]))
    Ry = rodrigues(np.array([0, math.radians(ry), 0]))
    Rz = rodrigues(np.array([0, 0, math.radians(rz)]))
    return np.dot(Rx, Rz, Ry).astype(np.float32)

def tvec(tx, ty, tz):
    return np.array([tx, ty, tz],dtype=np.float32)

def depthToPointParams(shape, intrinsics, extrinsics, scale=.001, clip=4):
    # assumes camera is at 0,0,0 facing down (map view) where z is up
    rows, cols = shape
    ppx, ppy, fx, fy = intrinsics
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    projx = -(c.astype(np.float32) - ppx) / fx
    projy = -(r.astype(np.float32) - ppy) / fy

    # extrinsics are transform from points to bot base (not base to cam)
    rx, ry, rz,  tx, ty, tz = extrinsics
    rv = rvec(rx, ry, rz)
    tv = tvec(tx, ty, tz)
    
    # if we don't have rotation we want to avoid the extra dot product
    if np.allclose([rx, ry, rz], 0):
        rv = None          

    return (projx, projy, rv, tv, scale, clip)


def depthToPointCloud(depthz, depth_to_point_params):
    projx, projy, rv, tv, scale, clip = depth_to_point_params
    depth = depthz.astype(np.float32) 
    depth *= scale # 1000 mm => 0.001 meters
    valid = np.ravel( (depth > .02) & (depth < clip)) #remove from the depth image all values above a given value (meters).
   
    z = -np.ravel(depth)[valid]
    x = np.ravel(depth * projx)[valid]
    y = np.ravel(depth * projy)[valid]

    verts = np.dstack((x, y, z)).reshape(-1, 3)

    if rv is not None:
        np.dot(verts, rv, out=verts)

    verts += tv
    
    return verts


out = np.empty((512,512,3), dtype=np.uint8) #1px per cm

#botmask = cv2.imread('anglerdroid/data/masks/botmask108.png',cv2.IMREAD_GRAYSCALE)
botmask = cv2.imread('anglerdroid/data/masks/botmask240.png',cv2.IMREAD_GRAYSCALE)
np.invert(botmask,out=botmask)

depth_data_td = np.zeros((240,424),dtype=np.uint16)
depth_image_td = np.zeros((240,424),dtype=np.uint8)
align_image_td = np.zeros((240,424),dtype=np.uint8)
color_image_td = np.zeros((240,424,3),dtype=np.uint8)

#depth_image_fw = np.zeros_like(botmask,dtype=np.uint16)
depth_data_fw = np.zeros((60,106),dtype=np.uint16)
depth_image_fw = np.zeros((60,106),dtype=np.uint8)
color_image_fw = np.zeros((240,424,3),dtype=np.uint8)

thresh_td = np.zeros_like(depth_image_td,dtype=np.uint8)

#intrinsics = depth_frame_fw.profile.as_video_stream_profile().intrinsics
intrinsics_td = (209.71139526367188, 116.33296203613281, 209.92787170410156, 209.92787170410156)
intrinsics_fw = ( 53.14219284057617,  29.85569000244141,  52.77927017211914,  52.77927017211914)

extrinsics_td = (    0.0, 0.0,  0.0, 0.0, 0.0, .9465)
extrinsics_fw = (25.6-90, 0.0, 90.0, -0.11, -0.025, .4235)

depth2point_td = depthToPointParams(depth_data_td.shape, intrinsics_td, extrinsics_td, scale=.001, clip=.963)
depth2point_fw = depthToPointParams(depth_data_fw.shape, intrinsics_fw, extrinsics_fw, scale=.001, clip=4.0)

h, w = out.shape[:2] 
view_aspect = float(h)/w
px_size=.01
scale = np.array([1/px_size,-1/px_size], dtype=np.float32)
center = np.array([w/2.0, h/2.0], dtype=np.float32)

frames_td = pipeline_td.wait_for_frames()
frames_fw = pipeline_fw.wait_for_frames()
nframes=0
now = time.time()



do_color=True
paused=False

#def Surfaces(vertices):
#    glBegin(GL_TRIANGLE_STRIP)
#    for i in range(4):
#    #glColor4f(1, 1, 1, 0.3)
#    glVertex3fv(vertices[i ,:])
#    glEnd()


class PointCloudPixelsGL:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.cuda_pixel_buffer = None  

        # Create a texture to use as the temporary render target
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.width, self.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)


    def setup_buffers(self, max_pixels):
        ftype = np.uint8  
        pixel_bytes = 4 * max_pixels * ftype().nbytes
        flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard

        PBO = glGenBuffers(1)  
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO)
        glBufferData(GL_PIXEL_UNPACK_BUFFER, pixel_bytes, None, GL_DYNAMIC_DRAW)

        self.cuda_pixel_buffer = CudaOpenGLMappedArray(ftype, (max_pixels, 4), PBO, flags)


    def draw(self, num_pixels):
        # Ensure the viewport matches the buffer size
        glViewport(0, 0, self.width, self.height)

        # Clear the default framebuffer
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        # Create a temporary framebuffer to render into
        temp_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, temp_fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.texture, 0)

        # Check if the framebuffer is complete
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Framebuffer not complete: {status}")

        # Activate the CUDA-OpenGL interop array 
        with self.cuda_pixel_buffer:

            # After you've modified the buffer in cupy
            # Directly draw (blit) the pixels from the buffer into the texture attached to the framebuffer
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.cuda_pixel_buffer.gl_buffer)
            glDrawPixels(self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE, None)

        # Blit the content of the temporary framebuffer to the default framebuffer
        glBindFramebuffer(GL_READ_FRAMEBUFFER, temp_fbo)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)  
        glBlitFramebuffer(
            0, 0, self.width, self.height, 
            0, 0, self.width, self.height, 
            GL_COLOR_BUFFER_BIT, GL_NEAREST
        )

        # Cleanup
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glDeleteFramebuffers(1, [temp_fbo])


    def __del__(self):
        self.cuda_pixel_buffer.unregister()
        glDeleteTextures(1, [self.texture]) 






gl = WindowsGLFW(848,480, "CuPy Cuda/OpenGL Map", 1,1,fps=True)
pixels = PointCloudPixelsGL(848,480)
#while not gl.closed():
#    with pixels.cuda_vertex_buffer as V:
#        V[0] = 255
        

pcv = PointCloudGL(848*480)

while not gl.closed():
    with pcv.cuda_vertex_buffer as V:
            
        #V[..., 0] = 0
        #V[..., 1] = 0
        #V[..., 2] = 0

        #V[0,0] = .5
        #V[0,1] = .5
        #V[0,2] = .5


        


        tnow = time.time()
        key = cv2.waitKey(1)

        if key == ord("b"):
            print("b")
            #cv2.imwrite("reference/botmask108.png",thresh_td)
            #cv2.imwrite("reference/botmask240.png",thresh_td)

        if key == ord("r"):     state.reset()

        if key == ord("p"):     
            print("p")
            paused = not paused

        if key == ord("u"):     print("u")
        if key == ord("i"):     print("i")
        if key == ord("d"):     print("d")
        if key == ord("z"):     print("z")

        if key == ord("c"):
            print("c")
            do_color = not do_color

        if key == ord("s"):
            cv2.imwrite('./out.png', out)

        if key == ord("e"):
            print("not implemented save to ply")
            #points.export_to_ply('./out.ply', mapped_frame)

        if key in (27, ord("q")) or cv2.getWindowProperty(state.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
            break

        if any(state.mouse_btns):
            pass

        if paused:
            continue




        
        frames_td = pipeline_td.wait_for_frames()
        frames_fw = pipeline_fw.wait_for_frames()
        tstep('wait_for_frames')
        
        depth_frame_td = frames_td.get_depth_frame()
        #depth_frame_td = decimate_td.process(depth_frame_td)
        #intrinsics = depth_frame_td.profile.as_video_stream_profile().intrinsics
        depth_frame_fw = frames_fw.get_depth_frame()
        #depth_frame_fw = decimate_fw.process(depth_frame_fw)
        #intrinsics = depth_frame_fw.profile.as_video_stream_profile().intrinsics

        depth_data_td[:,:] = np.asanyarray(depth_frame_td.get_data())[::2,::2]
        depth_data_fw[:,:] = np.asanyarray(depth_frame_fw.get_data())[::4,::4] 
        
        depth_image_fw[:,:]=( depth_data_fw )*.05
        tstep('get depth frames')

        if do_color:
            color_frame_td = frames_td.get_color_frame()
            color_frame_fw = frames_fw.get_color_frame()
            color_image_fw[:,:,:] = np.asanyarray(color_frame_fw.get_data())
            color_image_td[:,:,:] = np.asanyarray(color_frame_td.get_data())
            tstep('get color frames')
        
        zroi_min = 840
        zroi_max = 965
        zroi_scale = 2
        depth_image_td[:,:] =  255-(depth_data_td.clip(zroi_min,zroi_max) -zroi_min)*zroi_scale
        tstep('depth_image_td')
        
        floor_td=118
        thresh_td[:,:]=(depth_image_td > floor_td) * np.uint8(255)
        #thresh_td[:,:]=np.where(depth_image_td>floor_td,255,0)
        tstep('threshold depth')
        
        #cv2.imshow("td thresh",thresh_td)
        #cv2.imshow("td threshm",botmask)
        np.bitwise_or(thresh_td,botmask,out=align_image_td)
        np.bitwise_and(thresh_td,botmask,out=thresh_td)    
        tstep('show no bot')

        depth_data_td[thresh_td==0]=0 #mask floor and bot before deproject to points
        verts_td = depthToPointCloud(depth_data_td, depth2point_td)
        tstep('tdcloud')
        

        verts_fw = depthToPointCloud(depth_data_fw, depth2point_fw)
        tstep('get fw verts')

        verts = np.concatenate((verts_td, verts_fw), axis=0)

        verts = np.dot(verts - state.pivot, state.rotation) + state.pivot - state.translation

        # now we're top view so we clip the ground (to view ground: #verts = verts[verts[:, 2] > .38])
        verts = verts[verts[:, 2] < .38]
        tstep('thresh fw verts')

        #project 3d vector array to 2d
        proj = verts[:, :-1] * scale + center #ortho
        tstep('proj fw verts')
        

        # make 2d image coordinates and create 4 masks to ignore out-of-bound indices
        px, py = proj.astype(np.uint32).T
        m = (py >= 0) & (py < h) & (px >= 0) & (px < w)

        m1=m.copy()
        m2=m.copy()
        m1[verts_td.shape[0]:]=False
        m2[:verts_td.shape[0]]=False

        # that means we skip legal x values if corresponding y value was bad
        out.fill(0)
        out[py[m1], px[m1],0] = 255 # depth_values #color[u[m], v[m]]
        out[py[m2], px[m2],1] = 255 # depth_values #color[u[m], v[m]]
        tstep('out valid verts')
        

        #cv2.imshow('td depth scaled',depth_image_td)
        cv2.imshow("td thresh not bot",thresh_td)  
        cv2.imshow('startfw',depth_image_fw) 
        cv2.imshow(state.WIN_NAME, out)
        cv2.imshow("td align",align_image_td)
        if do_color:
            cv2.imshow("col td", color_image_td)
            cv2.imshow("col fw", color_image_fw)
        tstep('show images')

        
        dt = time.time() - now
        nframes += 1
        if dt >= 5.0:
            cv2.setWindowTitle(state.WIN_NAME, "Topdown View %dFPS (%.2fms) " % (math.ceil(nframes/dt), 1000*dt/nframes))
            
            now = time.time()
            nframes = 0
            print("times")
            for k,kt in times.items():
                print(k,kt)

        pcv.draw(len(V), gl.width, gl.height)
        gl.swap()
    

# Stop streaming
pipeline_td.stop()
pipeline_fw.stop()