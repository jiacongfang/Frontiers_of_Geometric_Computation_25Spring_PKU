# Copyright (c) 2018 Andy Zeng

import numpy as np
from skimage import measure
import trimesh

class TSDFVolume:
    """Volumetric TSDF Fusion of RGB-D Images.
    """
    def __init__(self, vol_bnds, voxel_size):
        """Constructor.

        Args:
            vol_bnds (ndarray): An ndarray of shape (3, 2). Specifies the
                xyz bounds (min/max) in meters.
            voxel_size (float): The volume discretization in meters.
        """
        vol_bnds = np.asarray(vol_bnds)
        assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."

        # Define voxel volume parameters
        self.vol_bnds = vol_bnds
        self.voxel_size = float(voxel_size)
        self.trunc_margin = 5 * self.voxel_size  # truncation on SDF

        
        #######################    Task 2    #######################
        # TODO: build voxel grid coordinates and initiailze volumn attributes
        # Initialize voxel volume
        self.tsdf_vol = None
        # for computing the cumulative moving average of weights per voxel
        self.weight_vol = None
        # Set voxel grid coordinates
        self.vox_coords = None
        ############################################################

    def integrate(self, depth_im, cam_intr, cam_pose, obs_weight=1.):
        """Integrate an RGB-D frame into the TSDF volume.

        Args:
            depth_im (ndarray): A depth image of shape (H, W).
            cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
            cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
            obs_weight (float): The weight to assign for the current observation. 
        """

        #######################    Task 2    #######################
        # TODO: Convert voxel grid coordinates to pixel coordinates
        # TODO: Eliminate pixels outside depth images
        # TODO: Sample depth values
        ############################################################
        
        #######################    Task 3    #######################
        # TODO: Compute TSDF for current frame
        ############################################################

        #######################    Task 4    #######################
        # TODO: Integrate TSDF into voxel volume
        ############################################################


def cam_to_world(depth_im, cam_intr, cam_pose, export_pc=False):
    """Get 3D point cloud from depth image and camera pose
    
    Args:
        depth_im (ndarray): Depth image of shape (H, W).
        cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
        cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
        export_pc (bool): Whether to export pointcloud to a PLY file.
        
    Returns:
        world_pts (ndarray): The 3D point cloud of shape (N, 3).
    """
    
    #######################    Task 1    #######################
    # TODO: Convert depth image to world coordinates
    world_pts = ...
    ############################################################
    
    if export_pc:
        pointcloud = trimesh.PointCloud(world_pts)
        pointcloud.export("pointcloud.ply")
    
    return world_pts
