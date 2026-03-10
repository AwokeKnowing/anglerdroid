import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt

from datetime import datetime
from functools import lru_cache

import anglerdroid as a7


class util3d:
    @staticmethod
    def bbox_from_xxyyzz(x1,x2,y1,y2,z1,z2,color=(1, 0, 0)):
        vec= o3d.utility.Vector3dVector(np.asarray([
            [x1, y1, z1],
            [x2, y1, z1],
            [x1, y2, z1],
            [x1, y1, z2],
            [x2, y2, z2],
            [x1, y2, z2],
            [x2, y1, z2],
            [x2, y2, z1]
        ]))

        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(vec)
        bbox.color = color
        return bbox
    
    
    @staticmethod
    def mb_box(sx,sy,sz,t,c):
        c=[1,1,1]
        #create a box and return mesh and bounding box
        rb1 = o3d.geometry.TriangleMesh.create_box(width=sx, height=sy, depth=sz)
        rb1.paint_uniform_color(c)
        rb1.translate(t)
        return rb1.get_axis_aligned_bounding_box(),rb1
    

    @staticmethod
    def calc_unrotate_plane(pcd,ransac_dist=.015,ransac_n=3,ransac_iter=100):
        
        plane, points = pcd.segment_plane(distance_threshold=ransac_dist,
                                          ransac_n=ransac_n,
                                          num_iterations=ransac_iter)
        #[a, b, c, d] = plane
        #print(f"Plane equation: {a:.5f}x + {b:.5f}y + {c:.5f}z + {d:.5f} = 0")
        
        in_pcd=pcd.select_by_index(points)
        
        obb=in_pcd.get_oriented_bounding_box()
        obb.color=(0, 1, 0)
        R=obb.R
        #print(R) #this can be backward because obb direction is random

        sy = np.sqrt(R[0][0] * R[0][0] +  R[1][0] * R[1][0])
        x = np.arctan2(R[2][1] , R[2][2])
        y = np.arctan2(-R[2][0], sy)
        z = 0 #we don't want to rotate on z #np.arctan2(R[1][0], R[0][0])
        
        # at least when detecting floor plane, avoids flip when R is oriented backward
        # todo: add some parameters to specify expected direction eg camera vector
        if(x>np.pi/2):
            x-=np.pi
        if(x<-np.pi/2):
            x+=np.pi
        
        R = o3d.geometry.get_rotation_matrix_from_zyx(np.array([z,y,x]))
        inv_R = np.linalg.inv(R)
        
        Rcenter=obb.get_center()
        
        #in_pcd.paint_uniform_color([1.0, 0, 0])
        #in_pcd=inpcd.rotate(inv_R,center=Rcenter)

        return inv_R, Rcenter,in_pcd


class botextractor:
    @staticmethod
    @lru_cache
    def getRobotBoundingMeshAndBoundingBoxes():
            fl = -1.0
            rb1sx = .44
            rb1sy = .36
            rb1sz = .55
            rb1bb, rb1 = util3d.mb_box(rb1sx,rb1sy,rb1sz,
                                [(-rb1sx/2-.07),-rb1sy/2,fl],
                                [0.9, 0.1, 0.1])

            rb2sx = .36
            rb2sy = .43
            rb2sz = .55
            rb2bb, rb2 = util3d.mb_box(rb2sx,rb2sy,rb2sz,
                                [(-rb2sx/2-.03),-rb2sy/2,fl],
                                [0.9, 0.5, 0.1])
            
            rb3sx = .22
            rb3sy = .46
            rb3sz = .22
            rb3bb, rb3 = util3d.mb_box(rb3sx,rb3sy,rb3sz,
                                [(-rb3sx/2+.00),-rb3sy/2,fl],
                                [0.1, 0.1, 0.1])

            rb4sx = .16
            rb4sy = .22
            rb4sz = 1.0
            rb4bb, rb4 = util3d.mb_box(rb4sx,rb4sy,rb4sz,[(-rb4sx/2-.21),-rb4sy/2,fl],[0.4, 0.4, 0.6])

            boxbot_mesh = rb1+rb2+rb3+rb4
            boxbot_bboxes = (rb1bb,rb2bb,rb3bb,rb4bb)

            return boxbot_mesh,boxbot_bboxes


    @staticmethod
    def extract(pcd, recalculate=False):
        
        if recalculate:
            botextractor.getRobotBoundingMeshAndBoundingBoxes.cache_clear()

        boxbot_mesh, boxbot_bboxes = botextractor.getRobotBoundingMeshAndBoundingBoxes()
        rb1bb,rb2bb,rb3bb,rb4bb = boxbot_bboxes

        rbp1 = rb1bb.get_point_indices_within_bounding_box(pcd.points)
        rbp2 = rb2bb.get_point_indices_within_bounding_box(pcd.points)
        rbp3 = rb3bb.get_point_indices_within_bounding_box(pcd.points)
        rbp4 = rb4bb.get_point_indices_within_bounding_box(pcd.points)

        boxbot_points = list(set(rbp1+rbp2+rbp3+rbp4))
        
        boxbot_pcd = pcd.select_by_index(boxbot_points)
        not_boxbot_pcd = pcd.select_by_index(boxbot_points, invert=True) #select outside points
        
        return not_boxbot_pcd, boxbot_pcd, boxbot_mesh


class AnglerDroidCameras:

    def __init__(self, *, rsTopdownSerial, rsForwardSerial=None, forwardPitchDeg=30, voxelSize=.25):
        self.withColor = False
        self.withForward = rsForwardSerial is not None
        self.voxelSize = voxelSize

        self.isTopdownCalibrated = False
        self.floorCalibrationPoints = o3d.geometry.PointCloud()
        self.isFloorCalibrationPointsStored = False
        
        #topdown camera is rotated 180 so x will be forward
        self.extrinsic_topdown2ego = np.identity(4)
        self.extrinsic_topdown2ego[:3,:3] = o3d.geometry.get_rotation_matrix_from_xyz(
            [np.deg2rad(0),
            np.deg2rad(0),
            np.deg2rad(180)])
        
        self.extrinsic_forward2ego = np.identity(4)
        self.extrinsic_forward2ego[:3,:3] = o3d.geometry.get_rotation_matrix_from_xyz(
            [np.deg2rad(forwardPitchDeg-90), #camera is pointed ~26 degrees down from straight ahead
            np.deg2rad(0),
            np.deg2rad(270)])
                      
        self.calibrated_topdown2ego = np.identity(4)
        self.topdown2ego_Rcenter = None

        self.calibrated_forward2ego = np.identity(4)
        
        self.topdownCam = a7.RealsenseCamera(rsTopdownSerial, self.withColor, self.extrinsic_topdown2ego)
        if self.withForward:
            self.forwardCam = a7.RealsenseCamera(rsForwardSerial, self.withColor, self.extrinsic_forward2ego)


    def getCombinedPointCloud(self, calibrateTopdown=True, calibrateForward=True):
        topdown_pcd, topdown_rgbd, topdown_intrinsic = self.getTopwdownPointCloud(calibrateTopdown)
        
        if self.withForward:
            forward_pcd, forward_rgbd, forward_intrinsic = self.getForwardPointCloud(topdown_pcd,calibrateForward)
       
            topdown_pcd.points.extend(forward_pcd.points)
            topdown_pcd.colors.extend(forward_pcd.colors)            
            topdown_pcd.normals.extend(forward_pcd.normals)

        return topdown_pcd
   

    def getTopwdownPointCloud(self, calibrate=True):
        tempsource, rgbd, intrinsic = self.topdownCam.frame(clip_meters=3, stride=4)
        topdown_pcd = tempsource
        topdown_pcd = tempsource.voxel_down_sample(voxel_size=self.voxelSize)
        
        # must be done on first frame to get initial rotation 
        if calibrate or not self.isTopdownCalibrated:
            self.calibrateTopdown2Ego(topdown_pcd)
            self.isTopdownCalibrated = True
            
        topdown_pcd = topdown_pcd.rotate(self.calibrated_topdown2ego, center=self.topdown2ego_Rcenter)

        return topdown_pcd, rgbd, intrinsic
    

    def getForwardPointCloud(self, topdown_pcd=None, align=True):
        forward_pcd, rgbd, intrinsic = self.forwardCam.frame(clip_meters=3, stride=4)
        forward_pcd = forward_pcd.voxel_down_sample(voxel_size=self.voxelSize)
        
        if self.withColor:
            forward_pcd = forward_pcd.translate([1.57,.20,.64])
        else:
            #forward_pcd = forward_pcd.translate([.88,.40,.32])
            forward_pcd = forward_pcd.translate([-.105,.035,-.525])

        if align: #self.with_forward_alignment and self.frame_count % 30 ==15:
            # try to align with icp. also calculates normals for the points
            self.calibrateForward2Ego(forward_pcd, topdown_pcd)

        forward_pcd.transform(self.calibrated_forward2ego)

        return forward_pcd,rgbd,intrinsic
    
    
    def calibrateTopdown2Ego(self, topdown_pcd):
        self.calibrated_topdown2ego, self.topdown2ego_Rcenter, floor = util3d.calc_unrotate_plane(topdown_pcd)

        if self.isFloorCalibrationPointsStored:
            self.floorCalibrationPoints.points = floor.points
            self.floorCalibrationPoints.colors = floor.colors
            self.floorCalibrationPoints.normals = floor.normals
    

    def calibrateForward2Ego(self,forward_pcd,topdown_pcd):
        estimator = o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxelSize * 2,max_nn=30)
        topdown_pcd.estimate_normals(estimator)
        
        reg_p2p = o3d.pipelines.registration.registration_icp(
            forward_pcd, topdown_pcd, .01, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40)
            )
        #forward_pcd.paint_uniform_color((0,1,0))
        #topdown_pcd.paint_uniform_color((1,0,0))
        print("reg",reg_p2p)
        if  reg_p2p.fitness > .15 and len(reg_p2p.correspondence_set)>250:
            self.calibrated_forward2ego = reg_p2p.transformation
            return True
        else:
            print("bad fit")
            return False
            #forward_pcd.transform(last_forward_transform)

        

class AnglerDroidVisionState:
    def __init__(self):
        self.obstacles_pcd = None
        self.obstacles_pcd2ego = None
        self.obstacles_img = None
        self.obstacles_img2ego = None
        self.obstacles_img_m_per_px = None
        self.obstacles_img_ego_xy = None
        self.ground_pcd = None
        self.ground_basis_pcd = None
        self.boxbot_pcd = None
        self.above_ground_bbox = None
        self.below_ground_bbox = None
        self.cam_topdown_depth = None
        self.cam_topdown_intrinsic = None
        self.cam_forward_depth = None
        self.cam_forward_intrinsic = None
    

class AnglerDroidVision:
    def __init__(self, *, rsTopdownSerial, rsForwardSerial=None):
        self.state = AnglerDroidVisionState()

        self.debug_view = {
            'show_boxbot': True,
            'show_ground': False,
            'show_topdown_roi_box': True,
            'show_axis': False,
            'show_ground_basis_pcd': False,
            'show_voxels': False,
            'show_clusters': False
        }
        
        self.with_shadow_ground = False #*

        self.voxel_detail = 2.0
        self.voxel_size = .1/float(2.**max(0,min(4, self.voxel_detail))) #clamp detail 0-5

        self.forward_alignment_skip = 30
        self.forward_alignment_skip_offset = 15
        
        self.boxbot_mesh = None
        self.boxbot_bboxes = None

        self.vxgrid = None
        self.vxgrid_old = None

        self.clusters = None

        self.topdown_size = (224,224)
        self.topdown_size2d = (224,224)
        self.ground_acc = np.zeros_like(self.topdown_size2d)
        self.morph_kernel = np.ones((5, 5), np.uint8)

        self.vis_topdown = o3d.visualization.Visualizer()
        
        #need to make a function to set view and capture frame and call it 1 fps
        self.vis_topdown.create_window(width=self.topdown_size[0],height=self.topdown_size[1])
        self.vis_topdown.get_render_option().point_size=4/self.voxel_detail

        self.state.obstacles_pcd = o3d.geometry.PointCloud()
        self.state.ground_pcd = o3d.geometry.PointCloud()
        self.state.ground_basis_pcd = o3d.geometry.PointCloud()
        
        #in space centered on topdown cam
        self.state.above_ground_bbox, self.state.below_ground_bbox = self.get3dViewBounds()

        self.cams = AnglerDroidCameras(rsTopdownSerial=rsTopdownSerial, 
                                       rsForwardSerial=rsForwardSerial, 
                                       forwardPitchDeg=26, 
                                       voxelSize=self.voxel_size)

        self.reset()
        
        
    def reset(self):
        self.resetFPS()
        self.refreshPointClouds()
        self.initializeVisualization()


    def resetFPS(self):
        self.frame_count = 0
        self.fps = 60
        self.dt0 = datetime.now()


    def update(self, show2d=True, show3d=True):
        self.dt0 = datetime.now()

        self.refreshPointClouds()
        self.update3D(show2d)
        self.update2D(show3d)

        process_time = datetime.now() - self.dt0
        self.fps = 1 / process_time.total_seconds()

        self.frame_count += 1
        if self.frame_count % 20 == 0:
            print("FPS: "+str(self.fps))


    def refreshPointClouds(self):
        align_fwd = self.frame_count % self.forward_alignment_skip == self.forward_alignment_skip_offset
        pcd = self.cams.getCombinedPointCloud(True, align_fwd)

        pcd, boxbot_pcd, boxbot_mesh = botextractor.extract(pcd)
        
        # save floor 
        ground_pcd = pcd.crop(self.state.below_ground_bbox)
        
        # get rid of the floor
        obstacles_pcd = pcd.crop(self.state.above_ground_bbox)

        # visualization is very slow if we don't update this way (update vs reallocate)
        self.state.ground_pcd.points = ground_pcd.points
        self.state.ground_pcd.colors = ground_pcd.colors
        self.state.ground_pcd.normals = ground_pcd.normals
        
        self.state.obstacles_pcd.points = obstacles_pcd.points
        self.state.obstacles_pcd.colors = obstacles_pcd.colors
        self.state.obstacles_pcd.normals = obstacles_pcd.normals

        if self.debug_view['show_ground_basis_pcd']:
            ground_basis_pcd = self.cams.floorCalibrationPoints()
            self.state.ground_basis_pcd.points = ground_basis_pcd.points
            self.state.ground_basis_pcd.colors = ground_basis_pcd.colors
            self.state.ground_basis_pcd.normals = ground_basis_pcd.normals

        if self.debug_view['show_boxbot'] and self.boxbot_mesh is None:
            self.boxbot_mesh = boxbot_mesh
   
    
    def segmentClusters(self, pcd, applyColor=True):
        labels = np.array(pcd.cluster_dbscan(eps=0.1, min_points=int(2.**self.voxel_detail)))

        try:
            max_label = labels.max()    
        except ValueError:
            max_label = 0
        #print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        #colors[labels < 1] = 0
        if applyColor:
            pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

        return labels
    
    
    def get3dViewBounds(self, with_forward=True, min_obstacle_height=.005):
        #topdown roi volume points (for obstacles, ie excluding floor)
        min_obstacle_height = .005 #obstacles smaller than this will be ignored
        below_ground_height = .5
        
        tdx1 = -0.92
        tdx2 = +0.92
        tdy1 = -0.54
        tdy2 = +0.54
        tdz1 = -0.92
        tdz2 = -0.0
        floor_z = tdz1 + min_obstacle_height 

        if with_forward:
            tdx1 = -0.92
            tdx2 = +3.56
            tdy1 = -2.24
            tdy2 = +2.24

        # because topdown 0,0 is the 'origin' (ego) of robot, we set 
        # the roi in worldspace rather than pcd space
        #above_ground_bbox = topdown_pcd.get_axis_aligned_bounding_box()
        above_ground_bbox = util3d.bbox_from_xxyyzz(tdx1,tdx2,tdy1,tdy2,floor_z,tdz2,(1, 0, 0))
        below_ground_bbox = util3d.bbox_from_xxyyzz(tdx1,tdx2,tdy1,tdy2,floor_z,floor_z-below_ground_height,(0, 1, 0))

        return above_ground_bbox, below_ground_bbox
    

    def topview(self, vis):
        rq = 523.598775 # sigh...
        vis.get_view_control().rotate(  0, rq,  0,  0)
        vis.get_view_control().rotate(-rq,  0,  0,  0)
        vis.get_view_control().rotate(  0,-rq,  0,  0)

        vis.get_view_control().set_zoom(.48)

        #ortho
        vis.get_view_control().change_field_of_view(step=-90)


    def update3D(self, show3d=False):

        if self.debug_view['show_voxels']:
            self.vxgrid = o3d.geometry.VoxelGrid.create_from_point_cloud(self.state.pcd,voxel_size=self.voxel_size)
            self.vis_topdown.remove_geometry(self.vxgrid_old, False)
            self.vis_topdown.add_geometry(self.vxgrid, False)
            self.vxgrid_old = self.vxgrid

        if self.debug_view['show_clusters']:
            # clusters has point info but the pcd is already colored
            self.clusters = self.segmentClusters(self.state.obstacles_pcd)
            
        if self.debug_view['show_topdown_roi_box']:
            # in case we are dynamically adjusting
            self.vis_topdown.update_geometry(self.state.above_ground_bbox)
        
        if self.debug_view['show_ground_basis_pcd']:
            self.vis_topdown.update_geometry(self.state.ground_basis_pcd)
        
           
        self.vis_topdown.update_geometry(self.state.obstacles_pcd)

        self.vis_topdown.poll_events()


    def update2D(self, show2d=False):
        self.vis_topdown.update_renderer()
        
            
        floor_depth = np.array(self.vis_topdown.capture_depth_float_buffer()).astype(np.uint8)
        
        #floor_depth = cv2.resize(floor_depth, floor_depth_size, interpolation= cv2.INTER_LINEAR)
        
        #floor_depth = cv2.cvtColor(floor_depth, cv2.COLOR_BGR2GRAY)
        ret, floor_depth = cv2.threshold(floor_depth, 0, 255, cv2.THRESH_BINARY)
        #floor_depth = cv2.morphologyEx(floor_depth, cv2.MORPH_CLOSE, self.morph_kernel)
        #floor_depth = np.where(floor_depth>0, 255, 0).astype(np.uint8).copy()

        #floor_depth_save = floor_depth
        #cv2.imshow('floor depth save', floor_depth_save)

        if self.with_shadow_ground:
            #recalculate every 4 frames
            if self.frame_count % 4 == 0:
                self.vis_topdown.add_geometry(self.state.ground_pcd, False)

                shadow_ground_depth = np.array(self.vis_topdown.capture_depth_float_buffer(True)).astype(np.uint8)
                self.vis_topdown.remove_geometry(self.state.ground_pcd, False)
                #shadow_ground_depth = cv2.resize(shadow_ground_depth, floor_depth_size, interpolation=cv2.INTER_LINEAR)
                shadow_ground_depth = cv2.morphologyEx(shadow_ground_depth, cv2.MORPH_CLOSE, self.morph_kernel)
                ret, shadow_ground_depth = cv2.threshold(shadow_ground_depth, 0, 255, cv2.THRESH_BINARY_INV)
                self.ground_acc = shadow_ground_depth
                #floor_acc = np.where(shadow_ground_depth>0, 0, 255).astype(np.uint8).copy()
                
                #cover the tail shadow
                cv2.rectangle(self.ground_acc, (100,276), (240-100,423), (0,0,0), 1)
            #print(floor_acc,floor_depth)
            floor_depth = cv2.bitwise_or(self.ground_acc, floor_depth)
            #floor_depth = cv2.morphologyEx(floor_depth, cv2.MORPH_CLOSE, self.morph_kernel)

        self.state.obstacles_img = floor_depth

        if show2d:
            cv2.imshow('floor depth', self.state.obstacles_img)
            #cv2.imwrite('botmask.png',floor_depth)
            #cv2.imshow('floor depth', floor_depth)
            #cv2.imshow('floor depth acc', floor_acc)
            cv2.waitKey(1)


    def initializeVisualization(self):
        self.vis_topdown.add_geometry(self.state.above_ground_bbox) #everything

        if self.debug_view['show_ground_basis_pcd']:
            self.vis_topdown.add_geometry(self.state.ground_basis_pcd)

        if self.debug_view['show_topdown_roi_box']:
            c=-.48
            #above_ground_bbox_topdown = util3d.bbox_from_xxyyzz(tdx1-c,tdx2+c,tdy1-c,tdy2+c,floor_z,tdz2,(1, 0, 0))
            #self.vis_topdown.add_geometry(above_ground_bbox_topdown) #everything

        if self.debug_view['show_boxbot']:
            self.vis_topdown.add_geometry(self.boxbot_mesh) #robot

        if self.debug_view['show_axis']:
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
            self.vis_topdown.add_geometry(mesh_frame)
            #mesh_frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[1, 0, 0])
            #self.vis_topdown.add_geometry(mesh_frame1)

        #if self.debug_view['show_voxels']:
        #    self.vis_topdown.add_geometry(self.vxgrid_old)
        
        if self.debug_view['show_ground']:
            self.vis_topdown.add_geometry(self.state.ground_pcd)

        self.vis_topdown.add_geometry(self.state.obstacles_pcd)

                
        self.topview(self.vis_topdown)


if __name__ == "__main__":
    a7vis = AnglerDroidVision(rsTopdownSerial="815412070676", rsForwardSerial="815412070180")
    try:
        while True:
            a7vis.update()         

    finally:
        cv2.destroyAllWindows()
        a7vis.cams.topdownCam.stop()
        a7vis.cams.forwardCam.stop()
        a7vis.vis_topdown.destroy_window()