import cv2
import numpy as np
import math
from dataclasses import dataclass
from typing import Any,Optional,List

@dataclass
class Motion2D:
    lin_vel:     Optional[float]
    ang_vel:     Optional[float]
    seconds:     Optional[float]
    img_tf:      Optional[np.array]
    img_tf_inv:  Optional[np.array] 
    dtheta:      Optional[float]
    dx:          Optional[float]
    dy:          Optional[float]
    euclid_dist: Optional[float]
    travel_dist: Optional[float]
    arc_radius:  Optional[float]



class TopdownVisualOdometry:
    def __init__(self, pixel_size=.01, center=None, show_matches=False):
        # meters per pixel
        self.pixel_size = pixel_size
        
        # (x,y) center of rotation for motion (set to center of first image if not given)
        self.center = center
        
        # window with keypoint matches
        self.show_matches = show_matches

        # or feature params
        self.good_feat_percent = .4
        self.max_feat = 1000
        

    def find_transform(self, img_from, img_to):
    
        # Convert images to grayscale if needed
        if img_from.ndim > 2:
            img_from = cv2.cvtColor(img_from, cv2.COLOR_BGR2GRAY)
        if img_to.ndim > 2:
            img_to = cv2.cvtColor(img_to, cv2.COLOR_BGR2GRAY)
        
        # center of rotation. points are shifted here so theta, dx, dy will be relative to here
        if self.center is None:
            self.center = (img_to.shape[1] / 2, img_to.shape[0] / 2)
                    
        # Detect ORB features and compute descriptors.
        orb = cv2.ORB_create(self.max_feat)
        kp1, des1 = orb.detectAndCompute(img_from, None)
        kp2, des2 = orb.detectAndCompute(img_to, None)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key = lambda x:x.distance)
        
        # Remove not so good matches
        numGoodMatches = 3 + int(len(matches) * self.good_feat_percent)

        good = matches[:numGoodMatches]
        #good = matches[:12]

        # Draw top matches before we shift to (0,0)
        if self.show_matches:
            #print("matches/good:",len(matches),"/",numGoodMatches)
            #kpimg = cv2.drawKeypoints(img1, kp1[:numGoodMatches], None, color=(0,255,0), flags=0)
            #cv2.imshow("points", kpimg)        
            imMatches = cv2.drawMatches(img_from, kp1, img_to, kp2, good, None)
            print("not showing matches")
            #cv2.imshow("matches TopdownVisualOdometry.find_transform", imMatches)

        # Calculate the shifts to the center
        x_shift, y_shift = self.center

        # Shift the keypoints to be relative to the center
        for keypoint in kp1:
            keypoint.pt = (keypoint.pt[0] - x_shift, keypoint.pt[1] - y_shift)
        for keypoint in kp2:
            keypoint.pt = (keypoint.pt[0] - x_shift, keypoint.pt[1] - y_shift)

        # Extract location of good matches
        points1 = np.zeros((len(good), 2), dtype=np.float32)
        points2 = np.zeros((len(good), 2), dtype=np.float32)

        for i, match in enumerate(good):
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt
              
        mat, inliers = cv2.estimateAffinePartial2D(points1, points2, False)
        
        return mat


    def find_motion2d(self, img1, img2, seconds=1.0) -> Motion2D:
        # find transform matrix from img1 to img2
        h = self.find_transform(img1, img2)
        if h is None:
            return Motion2D() # empty. not same as 'no motion'

        # Extract the rotation angle (theta)
        theta = np.arctan2(-h[1, 0], h[0, 0])

        # Extract the translation values (dx and dy)
        tx = h[0, 2]
        ty = h[1, 2]

        # theta dx dy are in relation to 0,0 as the center (pixels shifted above)
        tf_img1_to_img2 = cv2.getRotationMatrix2D(center=self.center, angle=np.degrees(theta), scale=1)
        tf_img1_to_img2[0,2] += tx # ie center_x + translate x
        tf_img1_to_img2[1,2] += ty

        # reverse transform (useful for aligning image to map)
        tf_img2_to_img1 = np.linalg.inv(np.vstack((tf_img1_to_img2,np.array([[0,0,1]]))))[:2]
        
        # find distance traveled along arc 
        lin_dist = math.sqrt(tx**2 + ty**2)
        try:
            radius = lin_dist / (2 * math.sin(theta/2))
            arc_dist = radius * theta  # distance physically travelled since turned to get there
        except ZeroDivisionError:
            radius = 10000000000
            arc_dist = lin_dist

        # calculate actual linear velocity (along arc) and angular velocity
        try:
            lin_vel = arc_dist / seconds
            ang_vel = theta / seconds
        except ZeroDivisionError:
            lin_vel = 0
            ang_vel = 0
        
        # return as motion relative to bot (ie image left means bot right, img down means bot up)
        # also transform from px space to meters 
        # (local relative transform, x forward y up, right handed)
        # x,y are swapped because in image y is aligned with bot forward, but 
        # x is forward in local bot frame
        ps = self.pixel_size
        ego_m = Motion2D(
            lin_vel     = lin_vel * ps,    # meters per sec
            ang_vel     = ang_vel,         # rad per sec
            seconds     = seconds,         # elapsed from img a to b
            img_tf      = tf_img1_to_img2, # transform img1 to im2 (for showing calc of motion is correct)
            img_tf_inv  = tf_img2_to_img1, # transform im2 back onto img 1 (for alignment to map)
            dtheta      = -theta,          # rad bot rotated
            dx          = ty * ps,         # meters forward (local x) NOTICE SWAP img y up to bot x forward  
            dy          = tx * ps,         # meters moved in local y
            euclid_dist = lin_dist * ps,   # meters as bird flies
            travel_dist = arc_dist * ps,   # meters likely traveled (arc from a to b)
            arc_radius  = radius * ps)     # radius in meters of arc 
            

        return ego_m
    

if __name__ == '__main__':
    import time

    img1 = cv2.imread("anglerdroid/data/tests/floorexample.png")
    img2 = cv2.imread("anglerdroid/data/tests/floorexample30.png")

    cv2.namedWindow("Overlay")
    cv2.createTrackbar("match_percent", "Overlay", 0, 100, lambda x: cv2.setTrackbarPos("match_percent", "Overlay", x))
    cv2.setTrackbarPos("match_percent", "Overlay", 50)

    botxy = (120,212)
    vodom = TopdownVisualOdometry(pixel_size=.01, center=botxy, show_matches=True)

    try:
        while True:
            good_per = cv2.getTrackbarPos('match_percent','Overlay') / 100
            t = time.time()
            
            vodom.good_feat_percent = good_per
            m = vodom.find_motion2d(img1, img2)
            print(time.time()-t)
            print(m)
            print(f"Angle (theta): {np.degrees(m.dtheta):.2f} degrees. ", f"Translation (forward, left): ({m.dx:.2f}, {m.dy:.2f})")
            
            # Apply the affine transformation to the image
            warped_forward = cv2.warpAffine(img1, m.img_tf, (img2.shape[1], img2.shape[0]))
            warped_back = cv2.warpAffine(img2, m.img_tf_inv, (img2.shape[1], img2.shape[0]))

            #cv2.imshow('Image 1', img1)
            #cv2.imshow('Image 2', img2)
            cv2.imshow('Overlay', cv2.addWeighted(img2,.3,warped_forward,.3,0.0))
            cv2.imshow('Overlay2', cv2.addWeighted(img1,.3,warped_back,.3,0.0))

            cv2.waitKey(1)
    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()
