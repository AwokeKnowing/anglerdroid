import open3d as o3d
import numpy as np
import cv2

from . import util3d

from .realsense import RealsenseCamera

class AnglerDroidCameras:

    def __init__(self, *, 
                 rsTopdownSerial, 
                 rsForwardSerial=None, 
                 rsForwardPitchDeg=30, 
                 voxelSize=.25,
                 rsColorEnabled=True,
                 webForwardDeviceId='/dev/video12'):
        
        self.webForwardCamId = webForwardDeviceId
        self.rsColorEnabled = rsColorEnabled
        self.rsForwardEnabled = rsForwardSerial is not None
        self.voxelSize = voxelSize

        self.isTopdownCalibrated = False
        self.floorCalibrationPoints = o3d.geometry.PointCloud()
        self.isFloorCalibrationPointsStored = True
        
        #topdown camera is rotated 180 so x will be forward
        self.extrinsic_rsTopdown2ego = np.identity(4)
        self.extrinsic_rsTopdown2ego[:3,:3] = o3d.geometry.get_rotation_matrix_from_xyz(
            [np.deg2rad(0),
            np.deg2rad(0),
            np.deg2rad(180)])
        
        self.extrinsic_rsForward2ego = np.identity(4)
        self.extrinsic_rsForward2ego[:3,:3] = o3d.geometry.get_rotation_matrix_from_xyz(
            [np.deg2rad(-90+rsForwardPitchDeg), #camera is pointed ~30 degrees (pitch) down from straight ahead (-90)
            np.deg2rad(0),
            np.deg2rad(270)])
                      
        self.calibrated_rsTopdown2ego = np.identity(4)
        self.rsTopdown2ego_Rcenter = None

        self.calibrated_rsForward2ego = np.identity(4)
        
        self.rsTopdownCam = RealsenseCamera(rsTopdownSerial, self.rsColorEnabled, self.extrinsic_rsTopdown2ego)
        if self.rsForwardEnabled:
            self.rsForwardCam = RealsenseCamera(rsForwardSerial, self.rsColorEnabled, self.extrinsic_rsForward2ego)

        self.webForwardCam = self.getNewWebForwardCam()
        
        #self.yoloModel = YOLO("yolov8n.pt")
        
    
    def getNewWebForwardCam(self):
        # Full list of Video Capture APIs (video backends): https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
        # For webcams, we use V4L2. jetson doesn't capture with h264 so it's mjpg
        cam = cv2.VideoCapture(self.webForwardCamId, cv2.CAP_V4L2)
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cam.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        cam.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
        cam.set(cv2.CAP_PROP_FPS, 30)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        return cam


    def getCombinedPointCloud(self, calibrateTopdown=True, calibrateForward=True):
        topdown_pcd, topdown_rgbd, topdown_intrinsic = self.getTopwdownPointCloud(calibrateTopdown)
        
        if self.rsForwardEnabled:
            forward_pcd, forward_rgbd, forward_intrinsic = self.getForwardPointCloud(topdown_pcd,calibrateForward)
       
            topdown_pcd.points.extend(forward_pcd.points)
            topdown_pcd.colors.extend(forward_pcd.colors)            
            topdown_pcd.normals.extend(forward_pcd.normals)

        return topdown_pcd
   

    def getTopwdownPointCloud(self, calibrate=True):
        topdown_pcd, rgbd, intrinsic = self.rsTopdownCam.frame(clip_meters=3, stride=80)
        if self.rsColorEnabled:
            pass
            #topdown_pcd = topdown_pcd.uniform_down_sample(16)
        else:
            topdown_pcd = topdown_pcd.voxel_down_sample(voxel_size=self.voxelSize)
        
        # must be done on first frame to get initial rotation 
        if calibrate or not self.isTopdownCalibrated:
            self.calibrateTopdown2Ego(topdown_pcd)
            self.isTopdownCalibrated = True
            
        topdown_pcd = topdown_pcd.rotate(self.calibrated_rsTopdown2ego, center=self.rsTopdown2ego_Rcenter)

        return topdown_pcd, rgbd, intrinsic
    

    def getForwardPointCloud(self, topdown_pcd=None, align=True):
        forward_pcd, rgbd, intrinsic = self.rsForwardCam.frame(clip_meters=3, stride=80)
        if self.rsColorEnabled:
            pass
            #forward_pcd = forward_pcd.uniform_down_sample(4)
            #forward_pcd = forward_pcd.uniform_down_sample(16)
        else:
            forward_pcd = forward_pcd.voxel_down_sample(voxel_size=self.voxelSize)      
        
        if self.rsColorEnabled:
            forward_pcd = forward_pcd.translate([-.139,.06,-.52])
        else:
            #forward_pcd = forward_pcd.translate([.88,.40,.32])
            forward_pcd = forward_pcd.translate([-.105,.035,-.525])

        forward_pcd = forward_pcd.rotate(self.calibrated_rsTopdown2ego, center=self.rsTopdown2ego_Rcenter)

        if False and align: #self.with_forward_alignment and self.frame_count % 30 ==15:
            # try to align with icp. also calculates normals for the points
            self.calibrateForward2Ego(forward_pcd, topdown_pcd)

        forward_pcd.transform(self.calibrated_rsForward2ego)

        return forward_pcd,rgbd,intrinsic
    
    
    def calibrateTopdown2Ego(self, topdown_pcd):
        try:
            self.calibrated_rsTopdown2ego, self.rsTopdown2ego_Rcenter, floor = util3d.calc_unrotate_plane(topdown_pcd)

            if self.isFloorCalibrationPointsStored:
                self.floorCalibrationPoints.points = floor.points
                self.floorCalibrationPoints.colors = floor.colors
                self.floorCalibrationPoints.normals = floor.normals
        except:
            print("cameras: failed to unrotate floor", flush=True)
    

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

        fitness_thresh = .15
        if self.rsColorEnabled:
            fitness_thresh = .03
                    
        if  reg_p2p.fitness > fitness_thresh and len(reg_p2p.correspondence_set)>250:
            self.calibrated_rsForward2ego = reg_p2p.transformation
            return True
        else:
            print("bad fit")
            return False
            #forward_pcd.transform(last_forward_transform)


    def getForwardCamFrame(self):
        if self.webForwardCam.isOpened():
            try:
                ret, frame = self.webForwardCam.read()
                if ret:
                    frame = cv2.resize(frame, None,fx=.5,fy=.5, interpolation=cv2.INTER_AREA)
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    return frame
                
            except cv2.error as e:
    
                    # inspect error object
                    print(e)
                    for k in dir(e):
                        if k[0:2] != "__":
                            print("e.%s = %s" % (k, getattr(e, k)))

                        # handle error: empty frame
                        if e.err == "!_src.empty()":
                            return None # break the while loop
                        else:
                            print("cameras: unknown cv2 cam error reading frame", flush=True)
                            return None
            except:
                print("cameras: webForwardCam died on frame", flush=True)
                self.webForwardCam.release()
                self.webForwardCam = self.getNewWebForwardCam()
                return self.getForwardCamFrame()
        
        return None