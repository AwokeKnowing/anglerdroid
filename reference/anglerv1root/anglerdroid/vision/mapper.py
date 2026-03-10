import cv2
import numpy as np
import math
from dataclasses import dataclass
from typing import Any,Optional,List

from .visodom import TopdownVisualOdometry

    
def load_images_from_folder(folder_path):
    # Get a list of all PNG files in the folder
    png_files = [file for file in os.listdir(folder_path) if file.endswith('.png')]
    
    # Sort the files based on their names
    png_files.sort()
    
    # Initialize an empty list to store images
    images = []
    
    # Loop through each PNG file
    for file_name in png_files:
        # Read the image using OpenCV
        image = cv2.imread(os.path.join(folder_path, file_name),cv2.IMREAD_GRAYSCALE)
        
        # Optionally, you can resize the image if needed
        # image = cv2.resize(image, (width, height))
        
        # Append the image to the list
        images.append(image)
    
    # Convert the list of images to a NumPy array
    images_array = np.array(images)
    
    return images_array


def draw_transform(image, m, center_x=500, center_y=50, bot_angle=0):
    # Calculate new position and angle
    new_x = center_x + m.dx
    new_y = center_y + m.dy
    new_angle = bot_angle +m.dtheta

    # Draw a line representing the movement
    cv2.line(image, (int(center_x), int(center_y)), (int(new_x), int(new_y)), (0, 255, 0), 2)

    # Draw a circle representing the bot
    cv2.circle(image, (int(new_x), int(new_y)), 2, (0, 0, 255), -1)

    # Calculate the endpoint of the orientation line
    end_x = int(new_x + 50 * np.cos(new_angle))
    end_y = int(new_y + 50 * np.sin(new_angle))

    # Draw a line representing the orientation
    #cv2.line(image, (int(new_x), int(new_y)), (end_x, end_y), (255, 0, 0), 2)

    return image

def rotateImage(image, angle):
    center=tuple(np.array(image.shape[0:2])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[0:2],flags=cv2.INTER_LINEAR)

def overlay_image(image_large, image_small, center_x, center_y, weight=1.0):
    # Get the dimensions of the small image
    small_height, small_width = image_small.shape

    # Calculate the top-left corner coordinates for placing the small image based on the center
    top_left_x = center_x - small_width // 2
    top_left_y = center_y - small_height // 2

    # Calculate the region of interest (ROI) on the large image
    roi = image_large[top_left_y:top_left_y + small_height, top_left_x:top_left_x + small_width]

    # Overlay the small image onto the ROI with the specified weight
    # Add 'weight' to ROI pixel values, but clip the result to the valid range for np.int8
    roi[np.where(image_small != 0)] = np.minimum(roi[np.where(image_small != 0)] + weight, 255)

    return image_large
    

if __name__ == '__main__':
    import time
    captures_path="captures/"
    img1 = cv2.imread(captures_path+"2024-02-05-214001_00127.png")
    img2 = cv2.imread(captures_path+"2024-02-05-214001_00132.png")

    cv2.namedWindow("Overlay")
    cv2.createTrackbar("match_percent", "Overlay", 0, 100, lambda x: cv2.setTrackbarPos("match_percent", "Overlay", x))
    cv2.setTrackbarPos("match_percent", "Overlay", 50)

    botxy = (120,212)
    vodom = TopdownVisualOdometry(pixel_size=1, center=botxy, show_matches=True)
    vodom.good_feat_percent=.2
    try:
        while False:
            good_per = cv2.getTrackbarPos('match_percent','Overlay') / 100
            t = time.time()
            
            vodom.good_feat_percent = good_per
            m = vodom.find_motion2d(img1, img2)
            print(time.time()-t)
            print(m.__dict__)
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

    import cv2
    import numpy as np
    import os
    import pickle 

    def skeletonize(img):
        size = np.size(img)
        skel = np.zeros(img.shape,np.uint8)
        
        ret,img = cv2.threshold(img,127,255,0)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        done = False
        img = cv2.erode(img,element)
        img = cv2.erode(img,element)
        while( not done):
            eroded = cv2.erode(img,element)
            temp = cv2.dilate(eroded,element)
            temp = cv2.subtract(img,temp)
            skel = cv2.bitwise_or(skel,temp)
            img = eroded.copy()
        
            zeros = size - cv2.countNonZero(img)
            if zeros==size:
                done = True

        #skel = cv2.dilate(skel,element)
        return skel        
        #cv2.imshow("skel",skel)

    folder_path = "captures/"
    pickle_file = "images.pkl"

    # Check if pickle file exists
    if os.path.exists(pickle_file):
        # Load images from pickle file
        with open(pickle_file, 'rb') as f:
            images_array = pickle.load(f)
    else:
        print("reloading images")
        # Load images from folder and sort
        images_array = load_images_from_folder(folder_path)
        for i in range(len(images_array)):
            images_array[i]=skeletonize(images_array[i])
        
        # Save images array to pickle file
        with open(pickle_file, 'wb') as f:
            pickle.dump(images_array, f)

    
    # Initialize a blank image for drawing transformations
    blank_image = np.zeros((2000, 2000, 3), dtype=np.uint8)
    # Initialize a blank image for drawing transformations
    map_image = np.zeros((2000, 2000, 1), dtype=np.uint8)
    # Set initial bot pose
    bot_x, bot_y, bot_angle = 1000, 1000, np.radians(-90)

    # Iterate over each image and display using OpenCV
    skip = 1
    start = 0
    end = min(400,len(images_array))
    for i in range(start+1, end, skip):
        # Display the image
        cv2.imshow('Image', images_array[i])
        print("image ",i)

        try:
            m = vodom.find_motion2d(images_array[i-1],images_array[i])
        except:
            print("nothing",i)
            continue

        if abs(m.dx)>(80 * skip) or abs(m.dy)>(80 * skip) or np.degrees(abs(m.dtheta))>20:
            print("skip",i,m.dx,m.dy,m.dtheta)
            continue
        m.dx=m.dx
        m.dy=m.dy

        

         # Draw the transformation on the blank image
        #blank_image = draw_transform(blank_image, m, center_x=bot_x, center_y=bot_y, bot_angle=bot_angle)

        # Update bot's pose
        print("rotated",np.degrees(abs(m.dtheta)))
        bot_angle += m.dtheta
        new_x = bot_x + m.dx * np.cos(bot_angle) - m.dy * np.sin(bot_angle)
        new_y = bot_y - m.dx * np.sin(bot_angle) + m.dy * np.cos(bot_angle)
        

        cv2.line(blank_image, (int(bot_x), int(bot_y)), (int(new_x), int(new_y)), (0, 255, 0), 1)
        bot_x=new_x
        bot_y=new_y


        # Display the transformed image
        bls = cv2.resize(blank_image, (800, 800))
        cv2.imshow('Transformed Image', bls)
        cv2.imshow('Image before', images_array[i])
        unrotated=rotateImage(images_array[i], np.degrees(bot_angle) )
        cv2.imshow('unrotated Image', unrotated)
        print(np.degrees(bot_angle))
        
        #map_image = np.zeros((2000, 2000, 1), dtype=np.uint8)
        
        map_image=overlay_image(map_image,unrotated,int(bot_x),int(-bot_y),255)
        mps = cv2.resize(map_image, (800, 800))
        cv2.imshow('map', mps)
        
        # Wait for a key press
        cv2.waitKey(3)  # Adjust the delay (in milliseconds) to control the frame rate
        
        # Close the window
    cv2.destroyAllWindows()