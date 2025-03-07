"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution."""

import time
import cv2
import numpy as np
from tqdm import tqdm
import pickle

import fusion


if __name__ == "__main__":
    print("Estimating voxel volume bounds...")
    n_imgs = 1000

    cam_intr = np.loadtxt("data/camera-intrinsics.txt", delimiter=" ")

    # vol_bnds[:, 0]: minimum bounds of the voxel volume along x, y, z
    # vol_bnds[:, 1]: maximum bounds of the voxel volume along x, y, z
    vol_bnds = np.zeros((3, 2))

    for i in tqdm(range(n_imgs)):
        # Read depth image
        depth_im = cv2.imread("data/frame-%06d.depth.png" % (i), -1).astype(float)
        # depth is saved in 16-bit PNG in millimeters
        depth_im /= 1000.0
        # set invalid depth to 0 (specific to 7-scenes dataset)
        depth_im[depth_im == 65.535] = 0

        # Read camera pose, a 4x4 rigid transformation matrix
        cam_pose = np.loadtxt("data/frame-%06d.pose.txt" % (i))

        #######################    Task 1    #######################
        #  Convert depth image to world coordinates
        view_frust_pts = fusion.cam_to_world(
            depth_im,
            cam_intr,
            cam_pose,
            export_pc=(i == 0),  # export pointcloud only for the first frame
        )
        # TODO: Update voxel volume bounds `vol_bnds`
        max_xyz = np.max(view_frust_pts, axis=0)
        min_xyz = np.min(view_frust_pts, axis=0)
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], min_xyz)
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], max_xyz)
        ############################################################

    print("Volume bounds:", vol_bnds)

    # Initialize TSDF voxel volume
    print("Initializing voxel volume...")
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.02)

    # Loop through images and fuse them together
    t0_elapse = time.time()
    for i in tqdm(range(n_imgs)):
        # Read depth image and camera pose
        depth_im = cv2.imread("data/frame-%06d.depth.png" % (i), -1).astype(float)
        depth_im /= 1000.0
        depth_im[depth_im == 65.535] = 0
        cam_pose = np.loadtxt("data/frame-%06d.pose.txt" % (i))

        # Integrate observation into voxel volume
        tsdf_vol.integrate(depth_im, cam_intr, cam_pose, obs_weight=1.0)

    fps = n_imgs / (time.time() - t0_elapse)
    print("Average FPS: {:.2f}".format(fps))

    # save the TSDF volume to disk
    with open("tsdf_vol.pkl", "wb") as f:
        pickle.dump(tsdf_vol, f)

    #######################    Task 4    #######################
    # TODO: Extract mesh from voxel volume, save and visualize it
    ############################################################
    print("Extracting mesh...")
    tsdf_vol.save_mesh("mesh")
    print("Mesh saved to mesh.ply")
