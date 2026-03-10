
import open3d as o3d
import numpy as np

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



def mb_box(sx,sy,sz,t,c):
    #c=[1,1,1]
    #create a box and return mesh and bounding box
    rb1 = o3d.geometry.TriangleMesh.create_box(width=sx, height=sy, depth=sz)
    rb1.paint_uniform_color(c)
    rb1.translate(t)
    return rb1.get_axis_aligned_bounding_box(),rb1



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