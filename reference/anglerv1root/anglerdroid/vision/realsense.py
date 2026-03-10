import pyrealsense2 as rs
import numpy as np
import open3d as o3d

class RealsenseCamera:
    
    def __init__(self,serial,with_color=True, extrinsics=np.identity(4)):
        self.serial =  serial
        self.extrinsics=extrinsics

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 90)
        if with_color:
            config.enable_stream(rs.stream.color, 320, 240, rs.format.rgb8, 60)
        
        profile = pipeline.start(config)
        
        depth_sensor = profile.get_device().first_depth_sensor()
        visual_presets = {
            'Custom': 0,
            'Default': 1,
            'Hand': 2,
            'HighAccuracy': 3,
            'HighDensity': 4,
            'MediumDensity': 5
        }
        depth_sensor.set_option(rs.option.visual_preset, visual_presets['HighAccuracy'])
        depth_scale = depth_sensor.get_depth_scale()

        if with_color:
            align_to = rs.stream.color
            align = rs.align(align_to)
        else:
            align=None

        #public props
        self.pipeline = pipeline
        self.config = config
        self.profile = profile
        self.depth_sensor = depth_sensor
        self.depth_scale = depth_scale
        self.align = align
        self.with_color=with_color
        

    def stop(self):
        self.pipeline.stop()

    def frame(self,clip_meters=3,stride=8):
        clipping_distance = clip_meters / self.depth_scale

        # Get frameset of color and depth
        frames = self.pipeline.wait_for_frames()

        if self.with_color:
            # Align the depth frame to color frame
            aligned_frames = self.align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                return None
            
            rs_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        else:
            aligned_depth_frame = frames.get_depth_frame()
            if not aligned_depth_frame:
                return None


            rs_intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

        # Camera calibration parameters (from RealSense intrinsics)
        fx, fy = rs_intrinsics.fx, rs_intrinsics.fy  # Focal lengths
        cx, cy = rs_intrinsics.ppx, rs_intrinsics.ppy  # Principal points

        # Create the camera intrinsic matrix
        camera_matrix = np.array([[fx,  0, cx],
                                  [ 0, fy, cy],
                                  [ 0,  0,  1]])
        
        flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]

        if self.with_color:
            intrinsics = o3d.camera.PinholeCameraIntrinsic(320, 240, fx, fy, cx, cy)
            depth_data = np.array(aligned_depth_frame.get_data())
            depth_image = o3d.geometry.Image(depth_data)
            color_data = np.asarray(color_frame.get_data())
            color_image = o3d.geometry.Image(color_data)

            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color_image, depth_image, depth_scale=1.0/self.depth_scale,
                    depth_trunc=clip_meters,
                    convert_rgb_to_intensity = False)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics,self.extrinsics)
        else:
            intrinsics = o3d.camera.PinholeCameraIntrinsic(848, 480, fx, fy, cx, cy)
            depth_data = np.array(aligned_depth_frame.get_data())
            depth_image = o3d.geometry.Image(depth_data)
            rgbd_image = depth_image
          
            pcd = o3d.geometry.PointCloud.create_from_depth_image(
                    depth_image, intrinsics,self.extrinsics, depth_scale=1.0/self.depth_scale,
                    depth_trunc=clip_meters,stride=stride)
            #pcd.paint_uniform_color([.5,.8,.5])
            
        
        pcd.transform(flip_transform)

        # when with_color is False rgbd_image is just the depth image
        return pcd, rgbd_image, intrinsics


if __name__ == '__main__':
    topdown = RealsenseCamera("815412070676")
    pcd, rgbd = topdown.frame()