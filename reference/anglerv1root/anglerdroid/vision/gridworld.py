import numpy as np

class GridLayer:
    def __init__(self, world_size=(768,1536), center=None):
        
        # Initialize the grid with specified dimensions
        self.width = world_size[0]
        self.height = world_size[1]

        # center 'pixel' in array data space (writing to world 0,0 writes to this)
        if center is None:
            center = (self.width//2, self.height//2 - 1)

        self.center = center
        self.layer_data = np.zeros((self.height, self.width))
    
    def update_layer(self, x, y, value):
        # Update the world grid at the specified coordinates
        world_x = int(self.center[0] + x)
        world_y = int(self.center[1] - y)
        self.layer_data[world_y, world_x] = value


class GridCamera:
    def __init__(self, grid_layers, view_size=(128,128), world_xy=(0,0), visible_layers=None):
        self.grid_layers = grid_layers

        self.set_size(view_size)
        self.set_visible_layers(visible_layers)
        self.look_at(world_xy)

    def set_bounds(self):
        # technically wrong according to constructor 
        # of gridlayer which can have center manually set (eg to have world 0,0 be at corner)
        height, width = list(self.grid_layers.values())[0].layer_data.shape
        self.grid_width = width
        self.grid_height = height
        self.min_cam_x = -(width // 2)  + (self.view_size[0] // 2)
        self.min_cam_y = -(height // 2) + (self.view_size[0] // 2)
        self.max_cam_x = -self.min_cam_x - 1
        self.max_cam_y = -self.min_cam_y - 1

    def set_size(self, size=None):
        if size is None:
            y,x = list(self.grid_layers.values())[0].shape
            size = (x,y)

        self.view_size = size

        self.set_bounds()

    def set_visible_layers(self, visible_layers=None):
        if visible_layers is None:
            visible_layers = self.grid_layers.keys()
        self.visible_layers = {name: True if name in visible_layers else False 
                               for name in self.grid_layers.keys()}
        
    def toggle_layer(self, name, state=None):
        if state is None:
            state = self.visible_layers[name]
        else:
            state = not self.visible_layers[name]

        self.visible_layers[name] = state

    def get_view(self):
        # Get views from specified layers or all layers

        # Calculate the view bounds (use the first grid_layer since they have the same size)
        x, y = self.world_xy
        layer = list(self.grid_layers.values())[0]

        view_x_start = max(0, int(layer.center[0] + x) - self.view_size[0] // 2)
        view_x_end = min(layer.width, view_x_start + self.view_size[0])

        view_y_start = max(0, int(layer.center[1] - y) - self.view_size[1] // 2)
        view_y_end = min(layer.height, view_y_start + self.view_size[1])

        
        views = {name: layer.layer_data[view_y_start:view_y_end, view_x_start:view_x_end] 
                 for name, layer in self.grid_layers.items() if self.visible_layers[name]}

        return views  
    
    def shift(self, relative_xy=(0,0)):
        x, y = self.world_xy
        x += relative_xy[0]
        y += relative_xy[1]

        self.look_at((x,y))

    def look_at(self, world_xy=(0,0)):
        x, y = world_xy
        x = max(self.min_cam_x, min(x, self.max_cam_x))
        y = max(self.min_cam_y, min(y, self.max_cam_y))

        self.world_xy = (x,y)


class GridWorld:
    def __init__(self,world_size=(768,1536)):
        self.world_size = world_size

        # Initialize an empty dictionaries for layers and cameras
        self.grid_layers = {}
        self.cameras = {} 

    def add_layer(self, layer_name='layer1', enable_for_cameras=True):
        # Add a new layer (GridMap) to the dictionary
        self.grid_layers[layer_name] = GridLayer(self.world_size)
        
        for camera in self.cameras.values():
            camera.toggle_layer(layer_name, enable_for_cameras)

    def add_camera(self, size=(128, 128), cam_name="camera1", world_xy=(0,0), vis_layers=None):
        # Add a new camera to the dictionary
        self.cameras[cam_name] = GridCamera(self.grid_layers, size, world_xy, vis_layers)

        return self.cameras[cam_name].get_view()
    
    def layer_update(self, x, y, value, layer_name='layer1'):
        # Update the world grid through a specific layer
        self.grid_layers[layer_name].update_layer(x, y, value)

    def camera_look_at(self, world_xy=(0,0), cam_name='camera1'):
        self.cameras[cam_name].look_at(world_xy)

    def camera_shift(self, relative_xy=(0,0), cam_name='camera1',):
        self.cameras[cam_name].shift(relative_xy)
        return self.cameras[cam_name].get_view()

    def camera_get_view(self, cam_name="camera1"):
        # Get views from specified layers or all layers using the stored initial position
        return self.cameras[cam_name].get_view()
    
    def camera_set_size(self,size=None, cam_name="camera1"):
        self.cameras[cam_name].set_size(size)
        return self.cameras[cam_name].get_view()


if __name__ == "__main__":
    import cv2

    # demo 1
    world1 = GridWorld((400,400))
    world1.add_layer('layer')
    view = world1.add_camera(size=(100,50))
    
    cv2.circle(view['layer'], (50,25), 30, (255,), -1)
    
    world1.camera_look_at((-150,0))
    for i in range(250):
        view = world1.camera_shift((1,0))
        if i % 10 == 0:
            cv2.circle(view['layer'], (50,25), 1, (255,), -1)
        cv2.imshow("camera", view['layer'])
        cv2.waitKey(16)

    for i in range(40):
        view = world1.camera_shift((-3,1))
        cv2.imshow("camera", view['layer'])
        cv2.waitKey(16)
    
    view = world1.camera_set_size((400,400))
    cv2.imshow("full layer view", view['layer'])

    cv2.waitKey(2000)
        
    # demo 2
    
    grid_world = GridWorld()

    # Add layers to GridWorld
    grid_world.add_layer('layer_a')
    grid_world.add_layer('layer_b')
    grid_world.add_layer('layer_c')

    # Add cameras to GridWorld
    grid_world.add_camera(size=(128, 128), cam_name='camera1', world_xy=(150, 150), vis_layers=['layer_a'])
    grid_world.add_camera(size=(128, 128), cam_name='camera2', world_xy=(200, 200), vis_layers=['layer_b'])
    grid_world.add_camera(size=(528, 528), cam_name='camera3', world_xy=(175, 175))

    # Get specified layers from the camera
    v1 = grid_world.camera_get_view('camera1')
    v2 = grid_world.camera_get_view('camera2')
    v3 = grid_world.camera_get_view('camera3') # all layers

    # Use cv2.circle to draw a 40-pixel circle in the view
    # we draw a circle in the center of each camera, but they offset because cameras are offset
    center = (v1['layer_a'].shape[1] // 2, v1['layer_a'].shape[0] // 2)  # Center of the view
    color = 255  # White color
    thickness = -1  # Fill the circle
    cv2.circle(v1['layer_a'], center, 50, color, thickness)
    cv2.circle(v2['layer_b'], center, 40, color, thickness)
    
    merged = cv2.merge(list(v3.values()))
    cv2.circle(merged, (200,200), 20, (255,255,255), thickness)

    cv2.imshow("camera1 (layer a)", v1['layer_a'])
    cv2.waitKey(1000)
    cv2.imshow("camera2 (layer b)", v2['layer_b'])
    cv2.waitKey(1000)
    cv2.imshow("camera3 (layer a,b,c merged)", merged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()