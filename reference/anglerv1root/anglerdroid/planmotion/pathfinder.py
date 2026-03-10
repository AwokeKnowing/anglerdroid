import numpy as np
import cv2
import math

class DynamicPathfinder:
    def __init__(self, 
                 botxy=(320,360), 
                 wheel_base_diameter=.34, 
                 wheel_diameter=.172,
                 pixel_size=.01,
                 max_rev_per_sec=2,
                 lookahead_sec=3,
                 path_margin_size=.02):
        
        # bot size/speed info
        self.wheel_base_diameter = wheel_base_diameter
        self.wheel_diameter = wheel_diameter     
        self.max_rev_per_sec = max_rev_per_sec
        self.max_dist_per_sec = self.wheel_diameter * np.pi * self.max_rev_per_sec
        
        # seconds to drive on curve path. 
        # affects which parths are chosen because 'collisions' happen on path
        self.path_lookahead_sec = lookahead_sec 
        self.path_steps_per_sec = 4

        # calculate values in pixel space
        self.pixel_size = pixel_size
        self.botxy = botxy
        self.max_px_per_sec = self.max_dist_per_sec / self.pixel_size
        self.max_distance_px = self.max_px_per_sec * self.path_lookahead_sec
        self.wheel_base_diameter_px = self.wheel_base_diameter / self.pixel_size
        
        # extra space beyond wheel base required for robot to spin around freely
        self.path_margin_size = path_margin_size 
        self.path_margin_size_px = self.path_margin_size / self.pixel_size
        self.bot_circle_radius_px = int(self.wheel_base_diameter_px/2)+int(self.path_margin_size_px)   
        
        # left/right wheel starting velocity (-1 to 1)
        self.current_vels_norm = (0,0)        

        


    def next_wheel_vels(self, image=None, goal=(0,0), from_vels=None, steps=4, botxy=None):
        if image     is not None: self.image = image
        if botxy     is not None: self.botxy = botxy
        if from_vels is not None: self.current_vels_norm = from_vels          

        vels = self.get_possible_wheel_vels(self.current_vels_norm, steps)
        paths = self.get_possible_paths(image, vels)
        path = self.get_best_path(paths, goal, self.botxy)

        if path is None:
            # phantom braking
            # need instead to realistically choose the slowes speed possible from here
            x,y = self.botxy
            path = {'vels':(0,0), 'points':np.array([[x,y]]), 'safe':True}
            paths.append(path)

        if True:
            imgview=self.image.copy()
            for p in paths:
                if p['safe']:
                    self.draw_path(imgview, p['points'],(255,255,255))
                else:
                    self.draw_path(imgview, p['points'],(64,64,64))
                    
                # show each path as it's generated
                #cv2.imshow("Paths", imgview)
                #cv2.waitKey(100)
            path['image'] = imgview
            path['all'] = paths
            
        return path


    def get_possible_wheel_vels(self,current_vels_norm=(0,0),steps=2,step_by=.04,max=1.0,min=-1.0):
        # bot can't reach all speeds in a short period. 
        # here we calculate which speeds are acheivable from current wheel speeds
        # if bot can change speed fast, may want to increas steps and decrease step_by

        #need to handle round low speed to zero eg < .01 = 0
        vels=[]
        step_max=(steps*step_by)
        current_v1,current_v2=current_vels_norm
        
        i = -(step_max)
        while not math.isclose(i,step_max):
            j = -steps*step_by
            while not math.isclose(j,step_max):
                vh1,vh2=current_v1+i,current_v2+j
                if min <= vh1 <= max and min <= vh2 <= max:
                    vels.append((vh1,vh2))
                j += step_by
            i += step_by

        #print(len(vels))
        #for v1,v2 in vels:
        #    print(v1,v2)

        return vels
    
    
    def get_possible_paths(self, image, possible_vels):
        """ given image of obstacles and possible wheel speeds, calculate 
        and find which ones collide with obstacle
        """
        image_sum=np.sum(image)
        paths=[]
        
        for vel in possible_vels:
            img=image.copy()
            
            points = self.get_path_points(vel)

            # Plot points on the image
            for px, py in points:
                # black filled circle traces path,
                cv2.circle(img, (px,py), self.bot_circle_radius_px, (0, 0,0), -1) 
            safe = np.sum(img) == image_sum
            path = {
                'vels': vel,
                'points': points,
                'safe': safe
            }
                
            paths.append(path)

        return paths
    
    
    def get_path_points(self, vels=(.5,.5)):
                    # we are scaling basically the 'zoom'/time of the pattern. 
            # should base on cm/px traveled during time represented by path
            px_traveled_left_wheel  = vels[0] * self.max_distance_px 
            px_traveled_right_wheel = vels[1] * self.max_distance_px
            curve_length_px = (px_traveled_left_wheel + px_traveled_right_wheel) / 2

            #r is the radius of circle defining the curvature of the path
            turn_radius_px = self.calculate_radius_px(vels[0]*self.max_px_per_sec, 
                                                      vels[1]*self.max_px_per_sec)
            turn_radius_px*=1.8 #!!!hack!!
            #the angle corresponding to how far the bot would travel on the circle (smaller for large circles)
            circumference = 2 * np.pi * turn_radius_px
            distance_factor = curve_length_px / circumference

            theta = np.linspace(0, 2 * np.pi * distance_factor, self.path_steps_per_sec*self.path_lookahead_sec)
            x = turn_radius_px * np.cos(theta) - turn_radius_px  # Offset the x-values by -r
            y = turn_radius_px * np.sin(theta)
            
            # Convert to pixel coordinates (shift to bot pos on image)
            pixel_x = (x + self.botxy[0]).astype(np.int32)
            pixel_y = (-y + self.botxy[1]).astype(np.int32)

            points = np.column_stack((pixel_x, pixel_y))

            return points
    

    def calculate_radius_px(self, vel_left, vel_right):     
        # Calculate the turning radius based on the velocity ratio
        if math.isclose(vel_right, vel_left):
            radius = 2000000000 #go straight
        else:
            radius = (self.wheel_base_diameter_px / 2) * ((vel_right + vel_left) / 
                                                 (vel_right - vel_left))
        if math.isclose(0.0,radius):
            radius = .0001 #spin in place
        
        return radius
    

    def draw_path(self, img, points=[], color=(255,),draw_bot=False, draw_bot_path=False):
        cv2.polylines(img, [points.reshape((-1, 1, 2))], False, color, 1, cv2.LINE_AA)
        #for px, py in points:
        #    cv2.circle(img, (px,py), 0, color, -1)
        if draw_bot_path:
            for px, py in points:
                cv2.circle(img, (px,py), self.bot_circle_radius_px, color, -1)
        if draw_bot:
            cv2.circle(img, (points[-1][0], points[-1][1]), self.bot_circle_radius_px, color, 1)
    
    
    def get_best_path(self, 
                      paths, 
                      goalxy=(0,0), 
                      botxy=(1,1), 
                      w_spin=.034, 
                      w_direction=.0034, 
                      w_speed=.18,
                      w_goal=.008):
        
        w_direction=.1
        w_goal=.01
        
        paths = sorted([path for path in paths if path['safe']], key=lambda p: ( 
            w_speed * ((1-p['vels'][0])/2 + (1-p['vels'][1])/2) / 2 + 
            w_spin * abs(p['vels'][0] - p['vels'][1])/2 +
            w_direction * abs(self.angle_between_points(p['points'][-1],goalxy,botxy)) / 180 +
            w_goal * min(math.sqrt((p['points'][-1][0]-goalxy[0])**2 + (p['points'][-1][1]-goalxy[1])**2) / (3.0/self.pixel_size), 1.0)
        ))

        if False:
            print("path scores:\n")
            for i,p in enumerate(paths):
                print("{:2}({:.2f}, {:.2f})".format(i,*p['vels']), 
                    "speed: {:.2f}".format(w_speed * ((1-p['vels'][0])/2 + (1-p['vels'][1])/2) / 2), 
                    "spin:  {:.2f}".format(abs(p['vels'][0] - p['vels'][1])/2),
                    "direc: {:.2f}".format(w_direction * abs(self.angle_between_points(p['points'][-1],goalxy,botxy)) / 180),
                    "goal: {:.2f}".format(min(math.sqrt((p['points'][-1][0]-goalxy[0])**2 + (p['points'][-1][1]-goalxy[1])**2) / (3/self.pixel_size), 1.0))
                )
        
        if len(paths) == 0:
            return None

        return paths[0]


    def angle_between_points(self,point=(1,1), target=(-1,1), origin=(0,0),rad=False):
        # Calculate the vector/angle from the origin to point
        vec_point = (point[0] - origin[0], point[1] - origin[1]) 
        rad_point = np.arctan2(-vec_point[1], vec_point[0]) - np.pi/2
        if rad_point < -np.pi:
            rad_point+=2*np.pi
        deg_point = np.degrees(rad_point)

        # Calculate the vector/angle from the origin to the target point
        vec_target = (target[0] - origin[0], target[1] - origin[1]) 
        rad_target = np.arctan2(-vec_target[1], vec_target[0]) - np.pi/2
        if rad_target < -np.pi:
            rad_target+=2*np.pi

        deg_target = np.degrees(rad_target)

        # Calculated difference
        rad_diff = rad_point - rad_target
        deg_diff = deg_point - deg_target
        
        #if deg_diff > 180:
        #    rad_diff = rad_diff - (2*np.pi)
        #    deg_diff = deg_diff - (360)

        #if deg_diff <= -180:
        #    rad_diff = rad_diff + (2*np.pi)
        #    deg_diff = deg_diff + (360)

        if False:
            print("Angle in radians to point: {:.2f}".format(rad_point))
            print("Angle in degrees to point: {:.2f}".format(deg_point))
            print("Angle in radians to target: {:.2f}".format(rad_target))
            print("Angle in degrees to target: {:.2f}".format(deg_target))
            print("Diff/Error in radians: {:.2f}".format(rad_diff))
            print("Diff/Error in degrees: {:.2f}".format(deg_diff))

        if rad:
            return rad_diff
        
        return deg_diff


if __name__ == '__main__':
    import time
    import random

    image = cv2.imread('anglerdroid/data/maps/world/world_fixed.png', cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, None,fx=6,fy=6, interpolation=cv2.INTER_NEAREST)
    generate = True

    if generate: #generate map
        # Create an OpenCV image of size 240x424
        # Define the dimensions of the image
        height,width = image.shape #424*2, 240*2

        # Create a black image
        #image = np.zeros((height, width), dtype=np.uint8)

        # Number of random blobs
        num_blobs = 10

        for _ in range(num_blobs):
            # Generate random coordinates for the center of the blob
            center_x = random.randint(0, width - 1)
            center_y = random.randint(0, height - 1)
            radius = random.randint(10, 50)

            # Draw a white blob on the black image
            cv2.circle(image, (center_x, center_y), radius, (255,), -1)

        # make sure no blobs over bot
        #cv2.circle(image, (width//2, int(height*.9)), int(height*.2), (0,), -1)

    print("shape", image.shape)
    width = image.shape[1]
    height = image.shape[0]

    goalx=width//2
    goaly=height//10
    
    botx = width//2
    boty = int(height*.8)
    wheel_space = .34  # in meters
    pixel_size = .02
    wheel_diameter=.172

    vel_l=.50
    vel_r=.55

    cv2.namedWindow("path")
    pathfinder = DynamicPathfinder((botx, boty), wheel_space, wheel_diameter, pixel_size)
  
    
    cv2.createTrackbar("vel L", "path", 0, 200, lambda x: cv2.setTrackbarPos("vel L", "path", x))
    cv2.createTrackbar("vel R", "path", 0, 200, lambda x: cv2.setTrackbarPos("vel R", "path", x))
    cv2.setTrackbarPos("vel L", "path", int(vel_l * 100)+100)
    cv2.setTrackbarPos("vel R", "path", int(vel_r * 100)+100)
    
    try:
        while True:
            vl = (cv2.getTrackbarPos('vel L','path')-100) / 100
            vr = (cv2.getTrackbarPos('vel R','path')-100) / 100
            
            t=time.time()
            path = pathfinder.next_wheel_vels(image, (goalx, goaly),(vl, vr)) 
            print("time: ",time.time()-t)

            show=image.copy()
            pathfinder.draw_path(show, path['points'],(255,))
            cv2.circle(show, (goalx, goaly), int((wheel_space/pixel_size)/2), (128,), -1)
            cv2.imshow("path", show)
            
            keyCode = cv2.waitKey(1) & 0xFF
            if keyCode == 27 or keyCode == ord('q'): break
            
            if keyCode == ord('w'): goaly = goaly - int(.05*height)
            if keyCode == ord('s'): goaly = goaly + int(.05*height)
            if keyCode == ord('a'): goalx = goalx - int(.05*width)
            if keyCode == ord('d'): goalx = goalx + int(.05*width)
                
    except Exception:
        import traceback
        traceback.print_exc()     
        
    cv2.destroyAllWindows()
