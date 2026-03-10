import os
import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

from datetime import datetime
from functools import lru_cache

from .cameras import AnglerDroidCameras
from . import util3d


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


class AnglerDroidVisionState:
    def __init__(self):
        self.forward_view_img = None
        self.obstacles_pcd = None
        self.map_pcd = None
        self.map_img = None
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
        self.map_bbox = None
    

class AnglerDroidVision:
    def __init__(self, *, rsTopdownSerial, rsForwardSerial=None, webForwardDeviceId=None, axon=None):
        self.rsTopdownSerial = rsTopdownSerial
        self.rsForwardSerial = rsForwardSerial
        self.webForwardDeviceId = webForwardDeviceId
        self.axon = axon # for testing we don't want to require axon

        self.state = AnglerDroidVisionState()

        self.debug_view = {
            'show_boxbot': False,
            'show_ground': False,
            'show_uncropped': True,
            'show_topdown_roi_box': False,
            'show_topdown_3dview': True,
            'show_axis': False,
            'show_ground_basis_pcd': False,
            'show_voxels': False,
            'show_clusters': False,
            'save_2d_map_images': False
        }
        
        self.with_color=True
        self.with_shadow_ground = False

        self.voxel_detail = 2.0
        self.voxel_size = .1/float(2.**max(0,min(4, self.voxel_detail))) #clamp detail 0-5

        self.forward_alignment_skip = 4
        self.forward_alignment_skip_offset = 2
        self.topdown_alignment_skip = 4
        self.topdown_alignment_skip_offset = 0
        
        self.boxbot_mesh = None
        self.boxbot_bboxes = None

        self.vxgrid = None
        self.vxgrid_old = None

        self.clusters = None

        self.topdown_size = (192,192)
        self.topdown_size2d = (192,192)
        self.ground_acc = np.zeros_like(self.topdown_size2d)
        self.morph_kernel = np.ones((5, 5), np.uint8)




    def __enter__(self):
        self.vis_topdown = o3d.visualization.Visualizer()
        self.vis_forward = o3d.visualization.Visualizer()
        self.vis_topdownmap = o3d.visualization.Visualizer()
        
        #need to make a function to set view and capture frame and call it 1 fps
        self.vis_topdown.create_window(window_name="topdown", 
                                       width=self.topdown_size[0],
                                       height=self.topdown_size[1],
                                       visible=self.debug_view['show_topdown_3dview'])
        self.vis_topdown.get_render_option().point_size=4/self.voxel_detail

        self.vis_topdownmap.create_window(window_name="topdownmap", 
                                       width=self.topdown_size[0],
                                       height=self.topdown_size[1],
                                       visible=self.debug_view['show_topdown_3dview'])
        self.vis_topdownmap.get_render_option().point_size=4/self.voxel_detail

        self.vis_forward.create_window(window_name="3D view",
                                       width=self.topdown_size[0]*2,
                                       height=self.topdown_size[1]*2)
        self.vis_forward.get_render_option().point_size=4/self.voxel_detail

        self.state.obstacles_pcd = o3d.geometry.PointCloud()
        self.state.ground_pcd = o3d.geometry.PointCloud()
        self.state.ground_basis_pcd = o3d.geometry.PointCloud()
        self.state.map_pcd = o3d.geometry.PointCloud()
        
        #in space centered on topdown cam
        self.state.above_ground_bbox, self.state.below_ground_bbox, self.state.map_bbox = self.get3dViewBounds()

        

        self.cams = AnglerDroidCameras(rsTopdownSerial=self.rsTopdownSerial, 
                                       rsForwardSerial=self.rsForwardSerial, 
                                       rsForwardPitchDeg=26.0, 
                                       voxelSize=self.voxel_size,
                                       rsColorEnabled=self.with_color,
                                       webForwardDeviceId=self.webForwardDeviceId)

        self.reset()

        return self # for with clause


    def __exit__(self, exc_type, exc_value, traceback):
        self.cams.rsTopdownCam.stop()
        self.cams.rsForwardCam.stop()
        self.cams.webForwardCam.release()
        self.vis_topdown.destroy_window()
        self.vis_forward.destroy_window()   
        
        
    def reset(self):
        self.resetFPS()
        self.refreshPointClouds()
        self.initializeVisualization()
        self.initObjectDetections()


    def resetFPS(self):
        self.frame_count = 0
        self.fps = 60
        self.dt0 = datetime.now()

    def imshow(self, title, img):
        if self.axon is None:
            print("anglerdroidvision: using default imshow", flush=True)
            cv2.imshow(title, img)
            cv2.waitKey(1)
        else:
            self.axon['/display/imshow'].put((title, img))


    def update(self, show2d=True, show3d=True):
        self.dt0 = datetime.now()

        self.refreshPointClouds()
        self.update3D(show3d)
        self.update2D(show2d)
        self.updateDetections()

        process_time = datetime.now() - self.dt0
        self.fps = .9*self.fps + .1 * (1 / process_time.total_seconds())

        self.frame_count += 1
        if self.frame_count % 30 == 0:
            print("FPS: "+str(self.fps))


    def initObjectDetections(self):
        return
        self.yoloModel = YOLO(os.path.join(os.path.dirname(__file__), 'yolov8n.pt'))


    def getDetectedObjects(self, img):
        results = self.yoloModel.predict(source=img, save=True, save_txt=True)  # save predictions as labels
        return results
    

    def updateDetections(self):
        #print("detecting")
        if self.frame_count < 20:
            return
        return
        results = self.yoloModel.predict(self.state.forward_view_img)
        #print(dir(results[0]))
        #for r in results:
        #    print(r)
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        #print("plotted")
        # Display the annotated frame
        self.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        #if cv2.waitKey(1) & 0xFF == ord("q"):
        #    return

    def refreshPointClouds(self):
        align_fwd = self.frame_count % self.forward_alignment_skip == self.forward_alignment_skip_offset
        align_tpd = self.frame_count % self.topdown_alignment_skip == self.topdown_alignment_skip_offset
        pcd = self.cams.getCombinedPointCloud(align_tpd, align_fwd)

        pcd, boxbot_pcd, boxbot_mesh = botextractor.extract(pcd)
        
        # save floor 
        ground_pcd = pcd.crop(self.state.below_ground_bbox)
        
        # get rid of the floor
        obstacles_pcd = pcd.crop(self.state.above_ground_bbox)

        # get rid of the floor and low obstacles
        map_pcd = pcd.crop(self.state.map_bbox)

        # undo crops for debugging
        if self.debug_view['show_uncropped']:
            obstacles_pcd = pcd

        # visualization is very slow if we don't update this way (update vs reallocate)
        self.state.ground_pcd.points = ground_pcd.points
        self.state.ground_pcd.colors = ground_pcd.colors
        self.state.ground_pcd.normals = ground_pcd.normals
        
        self.state.obstacles_pcd.points = obstacles_pcd.points
        self.state.obstacles_pcd.colors = obstacles_pcd.colors
        self.state.obstacles_pcd.normals = obstacles_pcd.normals

        self.state.map_pcd.points = map_pcd.points
        self.state.map_pcd.colors = map_pcd.colors
        self.state.map_pcd.normals = map_pcd.normals

        if self.debug_view['show_ground_basis_pcd']:
            ground_basis_pcd = self.cams.floorCalibrationPoints
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
        min_mappable_height = .04
        
        tdx1 = -0.92
        tdx2 = +0.92
        tdy1 = -0.54
        tdy2 = +0.54
        tdz1 = -0.92
        tdz2 = -0.0
        floor_z = tdz1 + min_obstacle_height 
        map_floor_z = tdz1 + min_mappable_height

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
        map_bbox = util3d.bbox_from_xxyyzz(tdx1,tdx2,tdy1,tdy2,map_floor_z,tdz2,(1, 0, 0))

        return above_ground_bbox, below_ground_bbox, map_bbox
    

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
            self.vxgrid = o3d.geometry.VoxelGrid.create_from_point_cloud(self.state.obstacles_pcd,voxel_size=self.voxel_size)
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
        
           
        if self.frame_count % 10 == 0: #3d view updates at slow fps since it's just for viewing
            self.vis_forward.update_geometry(self.state.obstacles_pcd)
            self.vis_forward.poll_events()
            #self.vis_topdown.update_renderer()

        
        
        

        


    def update2D(self, show2d=False):
        # to get the depth we must force an update/render
        self.vis_topdown.update_geometry(self.state.obstacles_pcd)
        self.vis_topdown.poll_events()
        self.vis_topdown.update_renderer()
        floor_depth = np.array(self.vis_topdown.capture_depth_float_buffer()).astype(np.uint8)

        self.vis_topdownmap.update_geometry(self.state.map_pcd)
        self.vis_topdownmap.poll_events()
        self.vis_topdownmap.update_renderer()
        map_depth = np.array(self.vis_topdownmap.capture_depth_float_buffer()).astype(np.uint8)
        
        #floor_depth = cv2.resize(floor_depth, floor_depth_size, interpolation= cv2.INTER_LINEAR)
        
        #floor_depth = cv2.cvtColor(floor_depth, cv2.COLOR_BGR2GRAY)
        ret, floor_depth = cv2.threshold(floor_depth, 0, 255, cv2.THRESH_BINARY)
        ret, map_depth = cv2.threshold(map_depth, 0, 255, cv2.THRESH_BINARY)
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

        self.state.obstacles_img = floor_depth.copy()
        self.state.map_img = map_depth

        if True:#skeleton
            img = map_depth.copy()
            size = np.size(img)
            skel = np.zeros(img.shape,np.uint8)
            
            ret,img = cv2.threshold(img,127,255,0)
            element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
            done = False
            
            while( not done):
                eroded = cv2.erode(img,element)
                temp = cv2.dilate(eroded,element)
                temp = cv2.subtract(img,temp)
                skel = cv2.bitwise_or(skel,temp)
                img = eroded.copy()
            
                zeros = size - cv2.countNonZero(img)
                if zeros==size:
                    done = True
            
            self.imshow("skel",skel)



        if True or self.frame_count %3 == 0:
            self.state.forward_view_img = self.cams.getForwardCamFrame()

        if show2d:
            self.imshow('floor depth', self.state.obstacles_img)
            self.imshow('map depth', self.state.map_img)
            if self.state.forward_view_img is not None:
                self.imshow('webForwardCam', self.state.forward_view_img)
            #cv2.imwrite('exampleview1.png',floor_depth)
            #cv2.imshow('floor depth', floor_depth)
            

            if self.debug_view['save_2d_map_images'] and map_depth is not None:
                cv2.imwrite('{}/{}_{}.{}'.format("captures",datetime.now().strftime("%Y-%m-%d-%H%M%S"), str(self.frame_count).zfill(5), 'png'), map_depth)
            #cv2.waitKey(1)


    def initializeVisualization(self):
        self.vis_topdown.add_geometry(self.state.above_ground_bbox) #everything
        self.vis_forward.add_geometry(self.state.above_ground_bbox) #everything

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

        # open3d voxels don't update, so we have to create and replace them each frame
        #if self.debug_view['show_voxels']:
        #    self.vis_topdown.add_geometry(self.vxgrid_old)
        
        if self.debug_view['show_ground']:
            self.vis_topdown.add_geometry(self.state.ground_pcd)

        
        self.vis_topdown.add_geometry(self.state.obstacles_pcd)
        self.vis_topdownmap.add_geometry(self.state.map_pcd)
        self.vis_forward.add_geometry(self.state.obstacles_pcd)

                
        self.topview(self.vis_topdown)
        self.topview(self.vis_topdownmap)
        self.topview(self.vis_forward)


if __name__ == "__main__":
    with AnglerDroidVision(rsTopdownSerial="815412070676", 
                           rsForwardSerial="944622074292", 
                           webForwardDeviceId='/dev/video12') as a7vis:
        try:
            while True:
                a7vis.update()         

        finally:
            cv2.destroyAllWindows()
        