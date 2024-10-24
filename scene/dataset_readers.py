#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import json
import os
from glob import glob
from typing import NamedTuple

import cv2
import imageio
import numpy as np
import skimage
from PIL import Image
from plyfile import PlyData, PlyElement
from skimage.util.dtype import img_as_float32
from tqdm import tqdm

from scene.colmap_loader import read_points3D_binary
from scene.gaussian_model import BasicPointCloud
from utils.graphics_utils import focal2fov, getWorld2View2


class CameraInfo(NamedTuple):
    uid: int
    R: np.array  # type: ignore
    T: np.array  # type: ignore
    FovY: np.array  # type: ignore
    FovX: np.array  # type: ignore
    prcppoint: np.array  # type: ignore
    image: np.array  # type: ignore
    image_path: str
    image_name: str
    width: int
    height: int
    mask: np.array  # type: ignore
    mono: np.array  # type: ignore


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


# load ego pose and camera calibration(extrinsic and intrinsic)
def load_camera_info(datadir):
    ego_pose_dir = os.path.join(datadir, "ego_pose")
    extrinsics_dir = os.path.join(datadir, "extrinsics")
    intrinsics_dir = os.path.join(datadir, "intrinsics")

    intrinsics = []
    extrinsics = []
    for i in range(5):
        intrinsic = np.loadtxt(os.path.join(intrinsics_dir, f"{i}.txt"))
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        intrinsics.append(intrinsic)

    for i in range(5):
        cam_to_ego = np.loadtxt(os.path.join(extrinsics_dir, f"{i}.txt"))
        extrinsics.append(cam_to_ego)

    ego_frame_poses = []
    ego_cam_poses = [[] for i in range(5)]
    ego_pose_paths = sorted(os.listdir(ego_pose_dir))
    for ego_pose_path in ego_pose_paths:
        # frame pose
        if "_" not in ego_pose_path:
            ego_frame_pose = np.loadtxt(os.path.join(ego_pose_dir, ego_pose_path))
            ego_frame_poses.append(ego_frame_pose)
        else:
            cam = image_filename_to_cam(ego_pose_path)
            ego_cam_pose = np.loadtxt(os.path.join(ego_pose_dir, ego_pose_path))
            ego_cam_poses[cam].append(ego_cam_pose)

    # center ego pose
    ego_frame_poses = np.array(ego_frame_poses)
    center_point = np.mean(ego_frame_poses[:, :3, 3], axis=0)
    ego_frame_poses[:, :3, 3] -= center_point  # [num_frames, 4, 4]

    ego_cam_poses = [np.array(ego_cam_poses[i]) for i in range(5)]
    ego_cam_poses = np.array(ego_cam_poses)
    ego_cam_poses[:, :, :3, 3] -= center_point  # [5, num_frames, 4, 4]
    return intrinsics, extrinsics, ego_frame_poses, ego_cam_poses


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


image_filename_to_cam = lambda x: int(x.split(".")[0][-1])
image_filename_to_frame = lambda x: int(x.split(".")[0][:6])


def generate_dataparser_outputs(
    datadir, selected_frames=None, build_pointcloud=True, cameras=[0, 1, 2, 3, 4]
):
    image_dir = os.path.join(datadir, "images")
    image_filenames_all = sorted(glob(os.path.join(image_dir, "*.png")))
    num_frames_all = len(image_filenames_all) // 5

    start_frame = 0
    end_frame = num_frames_all - 1

    # load calibration and ego pose
    intrinsics, extrinsics, ego_frame_poses, ego_cam_poses = load_camera_info(datadir)

    # load camera, frame, path
    frames = []
    frames_idx = []
    cams = []
    image_filenames = []

    ixts = []
    exts = []
    poses = []
    c2ws = []

    timestamp_path = os.path.join(datadir, "timestamps.json")
    with open(timestamp_path, "r") as f:
        timestamps = json.load(f)

    for image_filename in image_filenames_all:
        image_basename = os.path.basename(image_filename)
        frame = image_filename_to_frame(image_basename)
        cam = image_filename_to_cam(image_basename)
        if frame >= start_frame and frame <= end_frame and cam in cameras:
            ixt = intrinsics[cam]
            ext = extrinsics[cam]
            pose = ego_cam_poses[cam, frame]
            c2w = pose @ ext

            frames.append(frame)
            frames_idx.append(frame - start_frame)
            cams.append(cam)
            image_filenames.append(image_filename)

            ixts.append(ixt)
            exts.append(ext)
            poses.append(pose)
            c2ws.append(c2w)

    exts = np.stack(exts, axis=0)
    ixts = np.stack(ixts, axis=0)
    poses = np.stack(poses, axis=0)
    c2ws = np.stack(c2ws, axis=0)

    result = dict()
    result["exts"] = exts
    result["ixts"] = ixts
    result["poses"] = poses
    result["c2ws"] = c2ws
    result["image_filenames"] = image_filenames

    return result


def readWaymoFullInfo(path, cameras, **kwargs):
    # dynamic mask
    dynamic_mask_dir = os.path.join(path, "dynamic_mask")
    # sky mask
    sky_mask_dir = os.path.join(path, "sky_mask")
    # Optional: monocular normal cue
    mono_normal_dir = os.path.join(path, "normal")

    output = generate_dataparser_outputs(
        datadir=path,
        selected_frames=None,
        build_pointcloud=True,
        cameras=cameras,
    )

    exts = output["exts"]
    ixts = output["ixts"]
    poses = output["poses"]
    c2ws = output["c2ws"]
    image_filenames = output["image_filenames"]

    ########################################################################################################################
    cam_infos = []
    for i in tqdm(range(len(exts))):
        # generate pose and image
        ixt = ixts[i]
        c2w = c2ws[i]
        pose = poses[i]
        image_path = image_filenames[i]
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        width, height = image.size
        fx, fy = ixt[0, 0], ixt[1, 1]
        FovY = focal2fov(fx, height)
        FovX = focal2fov(fy, width)

        RT = np.linalg.inv(c2w)
        R = RT[:3, :3].T
        T = RT[:3, 3]
        K = ixt.copy()
        prcppoint = np.array([K[0][2] / width, K[1][2] / height])

        # load dynamic mask
        """
        如果都是只有一方是白色,即只有一个255,mask仍然是true,即依旧mask,这部分应当赋值为0
        只有均不是255,也就是均是黑色,才能赋值为false,这部分应当赋值为1
        """
        dy_mask = load_mask(os.path.join(dynamic_mask_dir, f"{image_name}.png"))[None]
        sky_mask = load_mask(os.path.join(sky_mask_dir, f"{image_name}.png"))[None]
        mask = np.logical_or(sky_mask, dy_mask)
        mask = np.where(mask, 0, 1).astype("float32")

        # Optional: load monocular normal
        mono_normal_image = cv2.imread(
            os.path.join(mono_normal_dir, f"{image_name}_normal.png")
        )
        mono_normal_array = np.array(mono_normal_image)
        mono_normal_array = mono_normal_array.astype("float32") / 255.0
        h, w, _ = mono_normal_array.shape
        mono_normal_array = 2 * mono_normal_array - 1
        mono_normal_array = np.transpose(mono_normal_array, (2, 0, 1))

        cam_info = CameraInfo(
            uid=i,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=image,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
            prcppoint=prcppoint,
            mask=mask,
            mono=mono_normal_array,
        )
        cam_infos.append(cam_info)

    cam_infos = sorted(cam_infos.copy(), key=lambda x: x.image_name)
    train_cam_infos = cam_infos
    test_cam_infos = []

    #######################################################################################################################3
    # Get point3d.ply and nerf_normalization
    # 1. Default nerf++ setting
    nerf_normalization = getNerfppNorm(train_cam_infos)

    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    if not os.path.exists(ply_path):
        print(
            "Converting point3d.bin to .ply, will happen only the first time you open the scene."
        )
        xyz, rgb, _ = read_points3D_binary(bin_path)
        storePly(ply_path, xyz, rgb)
    point_cloud = fetchPly(ply_path)

    scene_info = SceneInfo(
        point_cloud=point_cloud,  # type: ignore
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,  # type: ignore
    )
    # ipdb.set_trace()

    return scene_info

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb, normal=None):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    normals = np.zeros_like(xyz) if normal is None else normal

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def glob_imgs(path):
    imgs = []
    for ext in ["*.png", "*.jpg", "*.JPEG", "*.JPG"]:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs


def load_rgb(path):
    img = imageio.imread(path)
    img = img_as_float32(img)

    # pixel values between [-1,1]
    img -= 0.5
    img *= 2.0
    img = img.transpose(2, 0, 1)
    return img


def load_mask(path):
    alpha = imageio.imread(path, pilmode="F")
    alpha = img_as_float32(alpha) / 255
    return alpha


sceneLoadTypeCallbacks = {
    "waymo": readWaymoFullInfo,
}

# if __name__ == '__main__':
#     None