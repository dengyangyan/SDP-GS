import re
import numpy as np
from typing import Tuple
from utils.stepfun import sample_np
import copy
import torch
import cv2 
import random
from random import randint
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(lookdir, up, position, subtract_position=False):
  """Construct lookat view matrix."""
  vec2 = normalize((lookdir - position) if subtract_position else lookdir)
  vec0 = normalize(np.cross(up, vec2))
  vec1 = normalize(np.cross(vec2, vec0))
  m = np.stack([vec0, vec1, vec2, position], axis=1)
  return m


def poses_avg(poses):
  """New pose using average position, z-axis, and up vector of input poses."""
  position = poses[:, :3, 3].mean(0)
  z_axis = poses[:, :3, 2].mean(0)
  up = poses[:, :3, 1].mean(0)
  cam2world = viewmatrix(z_axis, up, position)
  return cam2world


def focus_point_fn(poses):
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt



def recenter_poses(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Recenter poses around the origin."""
  cam2world = poses_avg(poses)
  transform = np.linalg.inv(pad_poses(cam2world))
  poses = transform @ pad_poses(poses)
  return unpad_poses(poses), transform


def render_generate_spiral_path(poses,
                                bounds,
                                n_frames: int = 180,
                                n_rots: int = 2,
                                zrate: float = .5) -> np.ndarray:
  """Calculates a forward facing spiral path for rendering."""
  # Find a reasonable 'focus depth' for this dataset as a weighted average
  # of near and far bounds in disparity space.
  close_depth, inf_depth = bounds.min() * .9, bounds.max() * 5.
  dt = .75
  focal = 1 / (((1 - dt) / close_depth + dt / inf_depth))

  # Get radii for spiral path using 90th percentile of camera positions.
  positions = poses[:, :3, 3]
  radii = np.percentile(np.abs(positions), 90, 0)
  radii = np.concatenate([radii, [1.]])

  # Generate poses for spiral path.
  render_poses = []
  cam2world = poses_avg(poses)
  up = poses[:, :3, 1].mean(0)
  for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
    t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]
    position = cam2world @ t
    lookat = cam2world @ [0, 0, -focal, 1.]
    z_axis = position - lookat
    render_poses.append(viewmatrix(z_axis, up, position))
  render_poses = np.stack(render_poses, axis=0)
  return render_poses
def backcenter_poses(poses, pose_ref):
    """Recenter poses around the origin."""
    cam2world = poses_avg(pose_ref)
    poses = pad_poses(cam2world) @ pad_poses(poses)
    return unpad_poses(poses)
def convert_poses(poses):
    poses = np.concatenate([poses[:, 1:2], poses[:, 0:1], -poses[:, 2:3], poses[:, 3:4], poses[:, 4:5]], 1).transpose(2,0,1)
    bottom = np.tile(np.array([0,0,0,1.]).reshape([1,1,4]), (poses.shape[0], 1, 1))

    H, W, fl = poses[0, :, -1]

    poses = np.concatenate([poses[..., :4], bottom], 1)
    poses = np.linalg.inv(poses)
    Rs = poses[:, :3, :3]
    tvecs = poses[:, :3, -1]
    # print(Rs.shape, tvecs.shape, H, W, fl)
    return Rs, tvecs, H, W, fl
        


def generate_spiral_path(poses_arr,
                         n_frames: int = 180,
                         n_rots: int = 2,
                         zrate: float = .5) -> np.ndarray:
  """Calculates a forward facing spiral path for rendering."""
  poses = poses_arr[:, :-2].reshape([-1, 3, 5])
  bounds = poses_arr[:, -2:]
  fix_rotation = np.array([
      [0, -1, 0, 0],
      [1, 0, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
  ], dtype=np.float32)
  poses = poses[:, :3, :4] @ fix_rotation

  scale = 1. / (bounds.min() * .75)
  poses[:, :3, 3] *= scale
  bounds *= scale
  poses, transform = recenter_poses(poses)

  close_depth, inf_depth = bounds.min() * .9, bounds.max() * 5.
  dt = .75
  focal = 1 / (((1 - dt) / close_depth + dt / inf_depth))

  # Get radii for spiral path using 90th percentile of camera positions.
  positions = poses[:, :3, 3]
  radii = np.percentile(np.abs(positions), 90, 0)
  radii = np.concatenate([radii, [1.]])

  # Generate poses for spiral path.
  render_poses = []
  cam2world = poses_avg(poses)
  up = poses[:, :3, 1].mean(0)
  for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
    t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]
    position = cam2world @ t
    lookat = cam2world @ [0, 0, -focal, 1.]
    z_axis = position - lookat
    render_pose = np.eye(4)
    render_pose[:3] = viewmatrix(z_axis, up, position)
    render_pose = np.linalg.inv(transform) @ render_pose
    render_pose[:3, 1:3] *= -1
    render_pose[:3, 3] /= scale
    render_poses.append(np.linalg.inv(render_pose))
  render_poses = np.stack(render_poses, axis=0)
  return render_poses


def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)

def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]

def transform_poses_pca(poses):
    """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
  """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform
    return poses_recentered, transform

def generate_ellipse_path(views, n_frames=600, const_speed=True, z_variation=0., z_phase=0.):
    poses = []
    for view in views:
        tmp_view = np.eye(4)
        tmp_view[:3] = np.concatenate([view.R.T, view.T[:, None]], 1)
        tmp_view = np.linalg.inv(tmp_view)
        tmp_view[:, 1:3] *= -1
        poses.append(tmp_view)
    poses = np.stack(poses, 0)
    poses, transform = transform_poses_pca(poses)


    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    # Path height sits at z=0 (in middle of zero-mean capture pattern).
    offset = np.array([center[0] , center[1],  0 ])
    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)

    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset
    # Optional height variation need not be symmetric
    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)


    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y.
        # Optionally also interpolate in z to change camera height along path.
        return np.stack([
            (low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5)),
            (low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5)),
            z_variation * (z_low[2] + (z_high - z_low)[2] *
                           (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
        ], -1)

    theta = np.linspace(0, 2. * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)

    if const_speed:
        # Resample theta angles so that the velocity is closer to constant.
        lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
        theta = sample_np(None, theta, np.log(lengths), n_frames + 1)
        positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])
    # up = normalize(poses[:, :3, 1].sum(0))

    render_poses = []
    for p in positions:
        render_pose = np.eye(4)
        render_pose[:3] = viewmatrix(p - center, up, p)
        render_pose = np.linalg.inv(transform) @ render_pose
        render_pose[:3, 1:3] *= -1
        render_poses.append(np.linalg.inv(render_pose))
    return render_poses




def generate_random_poses_llff(views):
    """Generates random poses."""
    n_poses = 10000 # 10000 # 1000 for try
    poses, bounds = [], []
    for view in views:
        tmp_view = np.eye(4)
        tmp_view[:3] = np.concatenate([view.R.T, view.T[:, None]], 1)
        tmp_view = np.linalg.inv(tmp_view)
        tmp_view[:, 1:3] *= -1
        poses.append(tmp_view)
        bounds.append(view.bounds)
    poses = np.stack(poses, 0)
    bounds = np.stack(bounds) # np.array([[ 16.21311152, 153.86329729]])

    scale = 1. / (bounds.min() * .75)
    poses[:, :3, 3] *= scale
    bounds *= scale
    poses, transform = recenter_poses(poses)

    # Find a reasonable 'focus depth' for this dataset as a weighted average
    # of near and far bounds in disparity space.
    close_depth, inf_depth = bounds.min() * .9, bounds.max() * 5.
    dt = .75
    focal = 1 / (((1 - dt) / close_depth + dt / inf_depth))

    # Get radii for spiral path using 90th percentile of camera positions.
    positions = poses[:, :3, 3]
    radii = np.percentile(np.abs(positions), 100, 0)
    radii = np.concatenate([radii, [1.]])

    # Generate random poses.
    random_poses = []
    cam2world = poses_avg(poses)
    up = poses[:, :3, 1].mean(0)
    for _ in range(n_poses):
      t = radii * np.concatenate([2 * np.random.rand(3) - 1., [1,]])
      position = cam2world @ t
      lookat = cam2world @ [0, 0, -focal, 1.]
      z_axis = position - lookat
      random_pose = np.eye(4)
      random_pose[:3] = viewmatrix(z_axis, up, position)
      random_pose = np.linalg.inv(transform) @ random_pose
      random_pose[:3, 1:3] *= -1
      random_pose[:3, 3] /= scale
      random_poses.append(np.linalg.inv(random_pose))
    render_poses = np.stack(random_poses, axis=0)
    return render_poses

def generate_random_poses_blender(views):
    """Generates random poses."""
    n_poses = 10000 # args.n_random_poses
    render_poses = []
    
    for poses_idx in range(n_poses):
        # if not pseudo_stack:
        pesudo_idx1, pesudo_idx2=random.sample(range(len(views)), 2)
        pesudocam = disturb_cameras_novel(views[pesudo_idx1], views[pesudo_idx2])
        render_poses.append(pesudocam)

    render_poses = np.stack(render_poses, axis=0)
    return render_poses


def slerp(val, low, high):
    """Spherical interpolation. val has a range of 0 to 1."""
    if val <= 0:
        return low
    elif val >= 1:
        return high
    elif torch.allclose(low, high):
        return low
    omega = torch.acos(torch.dot(low/ torch.norm(low), high / torch.norm(high)))
    so = torch.sin(omega)
    return torch.sin((1.0-val)*omega) / so * low + torch.sin(val*omega)/so * high

def lerp(val, low, high):
    """Linear interpolation."""
    return low + (high - low) * val

def interpolate_pose(slerp_val, lerp_val, rot_mat1, trans_vec1, rot_mat2, trans_vec2):
    """Interpolate between two camera poses."""
    assert rot_mat1.shape == rot_mat2.shape == (3, 3)
    assert trans_vec1.shape == trans_vec2.shape == (3,)

    # Convert rotation matrices to quaternions
    quat1 = torch.tensor(cv2.Rodrigues(rot_mat1)[0]).flatten()
    quat2 = torch.tensor(cv2.Rodrigues(rot_mat2)[0]).flatten()

    # Slerp rotation and lerp translation
    interp_quat = slerp(slerp_val, quat1, quat2)
    interp_trans = lerp(lerp_val, trans_vec1, trans_vec2)

    # Convert quaternion back to rotation matrix
    interp_rot_mat = cv2.Rodrigues(interp_quat.numpy())[0]

    return interp_rot_mat, interp_trans

def disturb_cameras_novel(viewpoint_cam_pre, viewpoint_cam):
    # disturb camera position
    R = viewpoint_cam.R
    T = viewpoint_cam.T
    # trans = viewpoint_cam.trans
    # scale = viewpoint_cam.scale
    # camera_center = viewpoint_cam.camera_center.cpu().numpy()
    # projection_matrix = viewpoint_cam.projection_matrix
    # viewpoint_cam_novel = copy.deepcopy(viewpoint_cam)

    slerp_val = random.random() * 0.5
    lerp_val = random.random() * 0.1
    rot_mat1 = viewpoint_cam_pre.R
    rot_mat2 = viewpoint_cam.R
    trans_vec1 = viewpoint_cam_pre.T
    trans_vec2 = viewpoint_cam.T
    

    R, T = interpolate_pose(slerp_val, lerp_val, rot_mat1, trans_vec1, rot_mat2, trans_vec2)
    pose = np.eye(4)
    pose[:3, :3] = R.T
    pose[:3, 3] = T


    return pose


def generate_random_poses_llff_ours(views):
    """Generates random poses."""
    n_poses = 10000 # args.n_random_poses
    poses, bounds = [], []
    render_poses = []
    
    for poses_idx in range(n_poses):
        # if not pseudo_stack:
        pesudo_idx1, pesudo_idx2=random.sample(range(len(views)), 2)
        # pesudo_idx = randint(0, len(views) - 1)
        # pseudo_stack = views.copy()
        # pseudo_cam_1 = pseudo_stack.pop(randint(0, len(pseudo_stack) - 1))
        # pseudo_cam_2 = pseudo_stack.pop(randint(0, len(pseudo_stack) - 1))
    #     poses_curr = 
    # for view in views:
        pesudocam = disturb_cameras_novel(views[pesudo_idx1], views[pesudo_idx2])
        render_poses.append(pesudocam)
        # tmp_view = np.eye(4)
    #     tmp_view[:3] = np.concatenate([view.R.T, view.T[:, None]], 1)
    #     tmp_view = np.linalg.inv(tmp_view)
    #     tmp_view[:, 1:3] *= -/
    #     poses.append(tmp_view)
    #     bounds.append(view.bounds)
    # poses = np.stack(poses, 0)
    # bounds = np.stack(bounds) # np.array([[ 16.21311152, 153.86329729]])

    # scale = 1. / (bounds.min() * .75)
    # poses[:, :3, 3] *= scale
    # bounds *= scale
    # poses, transform = recenter_poses(poses)

    # # Find a reasonable 'focus depth' for this dataset as a weighted average
    # # of near and far bounds in disparity space.
    # close_depth, inf_depth = bounds.min() * .9, bounds.max() * 5.
    # dt = .75
    # focal = 1 / (((1 - dt) / close_depth + dt / inf_depth))

    # # Get radii for spiral path using 90th percentile of camera positions.
    # positions = poses[:, :3, 3]
    # radii = np.percentile(np.abs(positions), 100, 0)
    # radii = np.concatenate([radii, [1.]])

    # # Generate random poses.
    # random_poses = []
    # cam2world = poses_avg(poses)
    # up = poses[:, :3, 1].mean(0)
    # for _ in range(n_poses):
    #   t = radii * np.concatenate([2 * np.random.rand(3) - 1., [1,]])
    #   position = cam2world @ t
    #   lookat = cam2world @ [0, 0, -focal, 1.]
    #   z_axis = position - lookat
    #   random_pose = np.eye(4)
    #   random_pose[:3] = viewmatrix(z_axis, up, position)
    #   random_pose = np.linalg.inv(transform) @ random_pose
    #   random_pose[:3, 1:3] *= -1
    #   random_pose[:3, 3] /= scale
    #   random_poses.append(np.linalg.inv(random_pose))
    render_poses = np.stack(render_poses, axis=0)
    return render_poses

def generate_random_poses_360(views, n_frames=10000, z_variation=0.1, z_phase=0):
    poses = []
    for view in views:
        tmp_view = np.eye(4)
        tmp_view[:3] = np.concatenate([view.R.T, view.T[:, None]], 1)
        tmp_view = np.linalg.inv(tmp_view)
        tmp_view[:, 1:3] *= -1
        poses.append(tmp_view)
    poses = np.stack(poses, 0)
    poses, transform = transform_poses_pca(poses)


    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    # Path height sits at z=0 (in middle of zero-mean capture pattern).
    offset = np.array([center[0] , center[1],  0 ])
    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)

    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset
    # Optional height variation need not be symmetric
    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)


    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y.
        # Optionally also interpolate in z to change camera height along path.
        return np.stack([
            (low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5)),
            (low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5)),
            z_variation * (z_low[2] + (z_high - z_low)[2] *
                           (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
        ], -1)

    theta = np.random.rand(n_frames) * 2. * np.pi
    positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])
    # up = normalize(poses[:, :3, 1].sum(0))

    render_poses = []
    for p in positions:
        render_pose = np.eye(4)
        render_pose[:3] = viewmatrix(p - center, up, p)
        render_pose = np.linalg.inv(transform) @ render_pose
        render_pose[:3, 1:3] *= -1
        render_poses.append(np.linalg.inv(render_pose))
    return render_poses