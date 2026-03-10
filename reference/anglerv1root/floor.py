import time
import numpy as np
import cv2


class DynamicPathfinder:
    def __init__(self, axle_x, axle_y, cm_between_wheels=10, px_per_cm=1):
        self.axle_x = axle_x
        self.axle_y = axle_y
        self.cm_between_wheels = cm_between_wheels
        self.px_per_cm = px_per_cm

        self.current_vels = (0,0)

    def get_distance_factor(self, desired_length, radius):
        circumference = 2 * np.pi * radius
        distance_factor = desired_length / circumference
        return distance_factor
    

    def calculate_radius(self, distance_between_wheels, velocity_left_wheel, velocity_right_wheel):
        # Calculate the difference in wheel velocities
        velocity_difference = velocity_right_wheel - velocity_left_wheel

        # Calculate the turning radius based on the velocity ratio
        if velocity_difference == 0:
            radius = 2000000000 #go straight
        else:
            radius = distance_between_wheels / (2 * velocity_difference)
        return radius
    

    def get_valid_paths(self, image, possible_vels, distance_between_wheels,dist_factor=4.5):
        #distance = 6  # in cm

        image_sum=np.sum(image)
        n=0
        paths=[]
        for vel in possible_vels:
            img=image.copy()
            path={'vels':vel,'points':[]}
            
            v1 = vel[0]/100 * dist_factor
            v2 = vel[1]/100 * dist_factor
            distance = v1+v2/2
            r = self.calculate_radius(distance_between_wheels, v1,v2)
            
            distance_factor = self.get_distance_factor(distance, r)
            theta = np.linspace(0, 2 * np.pi * distance_factor, 8)
            x = r * np.cos(theta) - r  # Offset the x-values by -r
            y = r * np.sin(theta)
            
            # Convert to pixel coordinates (scaling factor of 20 for visualization)
            pixel_x = (x * 20 + 124).astype(int)
            pixel_y = (-y * 20 + 209).astype(int)

            r=68
            points = list(zip(pixel_x, pixel_y))
            # Plot points on the image
            
            for px, py in points:
                # black filled circle traces path,
                cv2.circle(img, (px,py), r, (0, 0,0), -1) 

            #if num white points has changed, black circles 'passed through' obstacles
            
            if np.sum(img) != image_sum:
                pass
                #self.draw_path(img,points,(255,255,255))
                
                #print("hit")
            else:
                path['points'] = points
                #self.draw_path(img, path['points'],(0,255,0))
                
                paths.append(path)
                
                #print("clear")

            #cv2.imshow("Turning Radii "+str(0), img)
            n+=1

            #cv2.waitKey(100)
        #cv2.destroyAllWindows()

        #for p in paths:
        #    print(p)
        return paths
    
    def draw_path(self, img, points=[], color=(255,255,255), r=68):
        for px, py in points:
            cv2.circle(img, (px,py), 3, color, -1)
        cv2.circle(img, (points[-1][0], points[-1][1]), r, color, 1)


    def draw_bot2d(self, image, axle_x, axle_y, color=(255,255,255), show_circle=False):
        rx = axle_x
        ry = axle_y

        cv2.rectangle(image, (rx-64, ry-29),(rx+64,ry+26),  color, -1)
        cv2.rectangle(image, (rx-53, ry-32),(rx+53,ry+50),  color, -1)
        cv2.rectangle(image, (rx-45, ry-39),(rx+45,ry+73),  color, -1)

        if show_circle:
            cv2.circle(image, (rx,ry),68,  (255, 255, 255), 1)


    def get_possible_wheel_vels(self, current_v1,current_v2,steps=5,step_by=10,max=100,min=-100):

        vels=[(5,-5)]
        
        i = -steps*step_by
        while i <= steps*step_by:
            j = -steps*step_by
            while j <= steps*step_by:
                vh1,vh2=current_v1+i,current_v2+j
                if min <= vh1 <= max and min <= vh2 <= max:
                    vels.append((vh1,vh2))
                j += step_by
            i += step_by

        return vels
    
    def get_best_path(self, paths, goal=(0,0), w_spin=.7, w_goal=.1, w_speed=1.0):
        paths = sorted(paths, key=lambda p: -( 
            w_speed * (p['vels'][0] + p['vels'][1]) - 
            w_spin * abs(p['vels'][0] - p['vels'][1]) -
            w_goal * np.linalg.norm(np.array(goal) - np.array(p['points'][-1]))
        ))
        """
        for p in paths:
            print(p['vels'], 
                  "speed:", w_speed * (p['vels'][0] + p['vels'][1]), 
                  "spin:",-w_spin * abs(p['vels'][0] - p['vels'][1]),
                  "goal:", -w_goal * np.linalg.norm(np.array(goal) - np.array(p['points'][-1]))
            )
        """
        return paths[0]


    def next_wheel_vels(self, image=None, goal=(0,0), from_vels=None, steps=2, axle_x=None, axle_y=None):
        if image is not None:  self.image = image
        if axle_x is not None: self.axle_x = axle_x
        if axle_y is not None: self.axle_y = axle_y
        if from_vels is not None:  self.current_vels = from_vels

        self.draw_bot2d(image, self.axle_x, self.axle_y, (0,0,0))

        vels = self.get_possible_wheel_vels(self.current_vels[0], self.current_vels[1], steps)
        paths = self.get_valid_paths(image, vels, self.cm_between_wheels)
        #print("found paths:",len(paths))
        path = self.get_best_path(paths, goal)

        if True:
            show=image.copy()
            path['points']
            self.draw_path(show, path['points'],(255,255,255))

            cv2.imshow("DynamicPathFinder path", show)
            #cv2.waitKey(1)
            
        return path
        


if __name__ == '__main__':
    # Create an OpenCV image of size 424x240
    image = cv2.imread('floorexample.png',cv2.IMREAD_GRAYSCALE)

    print("shape", image.shape)
    print (image.shape[0]/2)
    print (image.shape[1]/2)
    rx = 124
    ry = 209
    wheel_space = 10  # in cm
    px_per_cm = 1

    pathfinder = DynamicPathfinder(rx, ry, wheel_space, px_per_cm)
    pathfinder.draw_bot2d(image, rx, ry, (255,255,255))

    cv2.imshow("mask",image)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()

    pathfinder.next_wheel_vels(image, (50,0),(30, 30)) 

    t=time.time()
    for i in range(60):
        path =  pathfinder.next_wheel_vels(image, (50,0),(30, 30+i))
        #if i%10==0:cv2.waitKey(1)
    for i in range(60):
        path =  pathfinder.next_wheel_vels(image, (50,0),(30, 30+i))
        #if i%10==0:cv2.waitKey(1)
    for i in range(30):
        path =  pathfinder.next_wheel_vels(image, (50,0),(30, 30+i))
        #if i%10==0:cv2.waitKey(1)
    print(time.time()-t)

    show=image.copy()
    path['points']
    pathfinder.draw_path(show, path['points'],(0,255,0))

    #cv2.imshow("path", show)
    #cv2.waitKey(2000)
    cv2.destroyAllWindows()