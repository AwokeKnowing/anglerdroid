# pyrealsense2 is required.
# Please see instructions in https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python
import pyrealsense2 as rs
import open3d
from open3d import *
import numpy as np
from enum import IntEnum

from datetime import datetime
from open3d.visualization import *
from open3d.geometry import *
from open3d.camera import *
import anglerdroid


class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5

def rotate_pcd_to_plane(pcd, plane_model):
    """
    Rotate the point cloud so that the plane is flat on the z-axis.

    Parameters:
        pcd (open3d.geometry.PointCloud): The original point cloud.
        plane_model (list): List of plane equation coefficients [a, b, c, d].

    Returns:
        open3d.geometry.PointCloud: The rotated point cloud.
    """
    # Extract the normal vector from the plane equation
    normal_vector = np.array(plane_model[:3])

    # Compute the rotation matrix to align the normal vector to the z-axis
    z_axis = np.ndarray([0, 0, 1], dtype=np.float64)
    rotation_axis = np.cross(normal_vector, z_axis)
    rotation_angle = np.arccos(np.dot(normal_vector, z_axis) / (np.linalg.norm(normal_vector) * np.linalg.norm(z_axis)))
    rotation_matrix = open3d.geometry.get_rotation_matrix_from_axis_angle(rotation_angle)

    # Apply the rotation matrix to the point cloud
    rotated_pcd = pcd.rotate(rotation_matrix)

    return rotated_pcd

def get_intrinsic_matrix(frame):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    out = PinholeCameraIntrinsic(640, 480,
            intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
    return out


if __name__ == "__main__":
    topdown = RealsenseCamera("815412070676")
    
    # Create a pipeline
    pipeline = rs.pipeline()

    #Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()
    config.enable_device("815412070676")
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 90)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    # Start streaming
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()

    # Using preset HighAccuracy for recording
    depth_sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_scale = depth_sensor.get_depth_scale()

    # We will not display the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 3 # 3 meter
    clipping_distance = clipping_distance_in_meters / depth_scale
    # print(depth_scale)

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    vis = Visualizer()
    vis.create_window()

    pcd = PointCloud()
    

    voxel_size=.01

    

    # Streaming loop
    frame_count = 0
    try:
        while True:

            dt0=datetime.now()
            temp,rgbd = topdown.frame()

            temp=temp.voxel_down_sample(voxel_size=voxel_size)
            
            #temp.orient_normals_consistent_tangent_plane(10)
            temp.estimate_normals()
            temp.orient_normals_to_align_with_direction(orientation_reference=np.array([0., 0., 1.]))
            #print(temp.normals[0])
            

            if frame_count % 30 == 0:
                plane_model, inliers = temp.segment_plane(distance_threshold=0.01,
                                                ransac_n=3,
                                                num_iterations=1000)
                [a, b, c, d] = plane_model

                
                inpcd=temp.select_by_index(inliers)
                
                obb=inpcd.get_oriented_bounding_box()
                obb.color=(0, 1, 0)
                R=obb.R
                #print(R) #this can be backward because obb direction is random
                sy = np.sqrt(R[0][0] * R[0][0] +  R[1][0] * R[1][0])
                x = np.arctan2(R[2][1] , R[2][2])
                y = np.arctan2(-R[2][0], sy)
                z = np.arctan2(R[1][0], R[0][0])
                z=0
                if(x>np.pi/2):
                    x-=np.pi
                if(x<-np.pi/2):
                    x+=np.pi
                #print(x,y)
                R=open3d.geometry.get_rotation_matrix_from_zyx(np.array([z,y,x]))
                inv_R = np.linalg.inv(R)
                #print(inv_R)
                #print(f"Plane equation: {a:.5f}x + {b:.5f}y + {c:.5f}z + {d:.5f} = 0")
                Rcenter=obb.get_center()
            temp=temp.rotate(inv_R,center=Rcenter)

            #temp = rotate_pcd_to_plane(temp,plane_model)
            #print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
            #print("Displaying pointcloud with planar points in red ...")
            #inlier_cloud = pcd.select_by_index(inliers)
            #inlier_cloud.paint_uniform_color([1.0, 0, 0])
            #outlier_cloud = pcd.select_by_index(inliers, invert=True)

            pcd.points = temp.points
            pcd.colors = temp.colors
            pcd.normals = temp.normals
            #pcd.estimate_normals()

            
            vxgridb = open3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size=voxel_size)

            if frame_count==0:
                vxgrid = vxgridb
            #vxgrid.voxels=vxgridb.voxels
                
                

            #vxgrid.(vxgridb.to_octree(200))
            
            #open3d.visualization.draw([inlier_cloud, outlier_cloud])
             

            axis_aligned_bounding_box = pcd.get_axis_aligned_bounding_box()
            axis_aligned_bounding_box.color = (1, 0, 0)
            #oriented_bounding_box = pcd.get_oriented_bounding_box()
            #oriented_bounding_box.color = (0, 1, 0)

            if frame_count ==0:
                
                #vis.add_geometry(pcd)
                vis.add_geometry(vxgrid)
                vis.add_geometry(axis_aligned_bounding_box)
                #vis.renderer.set_normals(True)
                #vis.add_geometry(obb)

            

            #vis.update_geometry(pcd)
            
            vis.remove_geometry(vxgrid,False)
            vxgrid=vxgridb
            vis.add_geometry(vxgrid,False)
            vis.poll_events()
            vis.update_renderer()

            

            process_time = datetime.now() - dt0
            print("FPS: "+str(1/process_time.total_seconds()))
            frame_count += 1

    finally:
        pipeline.stop()
    vis.destroy_window()