# Copyright (c) 2018 Andy Zeng

import numpy as np
import trimesh


class TSDFVolume:
    """Volumetric TSDF Fusion of RGB-D Images."""

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
        # Compute voxel volume dimensions
        self.vol_dim = np.round((vol_bnds[:, 1] - vol_bnds[:, 0]) / voxel_size).astype(
            int
        )

        # Initialize voxel volume with 1.0 (truncation band) (NOT 0 !!)
        self.tsdf_vol = np.ones(
            self.vol_dim[0] * self.vol_dim[1] * self.vol_dim[2], dtype=np.float32
        )
        # for computing the cumulative moving average of weights per voxel
        self.weight_vol = np.zeros(
            self.vol_dim[0] * self.vol_dim[1] * self.vol_dim[2], dtype=np.float32
        )
        # Set voxel grid coordinates
        xv, yv, zv = np.meshgrid(
            np.arange(self.vol_dim[0]),
            np.arange(self.vol_dim[1]),
            np.arange(self.vol_dim[2]),
            indexing="ij",
        )
        self.vox_coords = np.stack([xv, yv, zv], axis=-1).reshape(-1, 3)
        ############################################################

    def integrate(self, depth_im, cam_intr, cam_pose, obs_weight=1.0):
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
        # Step I: Convert voxel grid coordinates to pixel coordinates
        # voxel grid coordinates => world coordinates
        world_coords = self.vox_coords * self.voxel_size + self.vol_bnds[:, 0]  # (N, 3)
        world_coords_hom = np.hstack(
            [world_coords, np.ones((world_coords.shape[0], 1))]
        )  # (N, 4)

        # world coordinates => camera coordinates
        cam_pose_inv = np.linalg.inv(cam_pose)
        cam_coords = (np.dot(cam_pose_inv, world_coords_hom.T).T)[:, :3]  # (N, 3)

        # camera coordinates => pixel coordinates
        pixel_coords = np.dot(cam_intr, cam_coords.T).T  # (N, 3)
        pixel_coords[:, 0] /= pixel_coords[:, 2]
        pixel_coords[:, 1] /= pixel_coords[:, 2]
        # round to nearest pixel
        pixel_coords = np.round(pixel_coords[:, :2]).astype(int)

        # Step II: Eliminate pixels outside depth images [H, W]
        H, W = depth_im.shape
        valid_pix = (
            (pixel_coords[:, 0] >= 0)
            & (pixel_coords[:, 0] < W)
            & (pixel_coords[:, 1] >= 0)
            & (pixel_coords[:, 1] < H)
        )

        # Step III: Sample depth values: (depth_im[H, W] - the depth of the voxel grid coordinates)
        depth_val = np.zeros(self.vox_coords.shape[0])
        depth_val[valid_pix] = depth_im[
            pixel_coords[valid_pix, 1].astype(int),
            pixel_coords[valid_pix, 0].astype(int),
        ]

        depth_sample = np.zeros(self.vox_coords.shape[0])
        depth_sample[valid_pix] = (
            depth_val[valid_pix]
            - cam_coords[valid_pix, 2]  # depth of the voxel grid coordinates
        )

        #######################    Task 3    #######################
        # TODO: Compute TSDF for current frame
        ############################################################
        valid_mask = (depth_val > 0) & (depth_sample >= -self.trunc_margin)
        tsdf = np.minimum(1.0, depth_sample / self.trunc_margin)

        #######################    Task 4    #######################
        # TODO: Integrate TSDF into voxel volume
        ############################################################
        self.tsdf_vol[valid_mask] = (
            self.tsdf_vol[valid_mask] * self.weight_vol[valid_mask]
            + tsdf[valid_mask] * obs_weight
        ) / (self.weight_vol[valid_mask] + obs_weight)
        self.weight_vol[valid_mask] += obs_weight


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
    # Step I: Convert the depth iamge to camera coordinates
    cam_intr_inv = np.linalg.inv(cam_intr)

    H, W = depth_im.shape

    # Create a grid of pixel coordinates
    i, j = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pixels = np.stack([j, i, np.ones_like(i)], axis=-1).reshape(-1, 3)

    # Convert pixel coordinates to camera coordinates
    cam_cor = np.dot(cam_intr_inv, pixels.T).T
    cam_cor *= depth_im.flatten()[:, None]

    # Step II: Convert the camera coordinates to world coordinates
    world_pts = np.dot(
        cam_pose, np.hstack([cam_cor, np.ones((cam_cor.shape[0], 1))]).T
    )[:3, :].T
    ############################################################

    if export_pc:
        pointcloud = trimesh.PointCloud(world_pts)
        pointcloud.export("pointcloud.ply")

    return world_pts
