# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from Utils import *
from datareader import *
import itertools
from learning.training.predict_score import *
from learning.training.predict_pose_refine import *
import yaml


class FoundationPose:
  """  
  A class for estimating the pose of a 3D object in a scene.
  """
  def __init__(self, 
               model_pts=None, 
               model_normals=None, 
               symmetry_tfs=None, 
               mesh=None, 
               scorer:ScorePredictor=None, 
               refiner:PoseRefinePredictor=None, 
               glctx=None, 
               debug=0, debug_dir='/home/bowen/debug/novel_pose_debug/'):
    """I

    Args:
      model_pts (np.ndarray): The 3D points of the object model.
      model_normals (np.ndarray): The normals of the object model.
      symmetry_tfs (np.ndarray, optional): The symmetry transformations of the object model. Defaults to None.
      mesh (trimesh.Trimesh, optional): The object mesh. Defaults to None.
      scorer (ScorePredictor, optional): The score predictor for pose refinement. Defaults to None.
      refiner (PoseRefinePredictor, optional): The pose refiner for pose refinement. Defaults to None.
      glctx (dr.RasterizeCudaContext, optional): The rendering context for rendering synthetic views. Defaults to None.
      debug (int, optional): The debug level. Defaults to 0.
      debug_dir (str, optional): The directory for saving debug information. Defaults to '/home/bowen/debug/novel_pose_debug/'.
  """

    self.gt_pose = None
    self.ignore_normal_flip = True
    self.debug = debug
    self.debug_dir = debug_dir
    os.makedirs(debug_dir, exist_ok=True)

    if not( model_pts is None or model_normals is None and mesh is None ):
      self.reset_object(model_pts, model_normals, symmetry_tfs=symmetry_tfs, mesh=mesh)

    # Pose Initialization: Rotations uniformly sampled from isosphere centered on object with camera facing center
    

    self.glctx = glctx

    if scorer is not None:
      self.scorer = scorer
    else:
      self.scorer = ScorePredictor()

    if refiner is not None:
      self.refiner = refiner
    else:
      self.refiner = PoseRefinePredictor()

    self.pose_last = None   # Used for tracking; per the centered mesh


  def reset_object(
      self, 
      model_pts: np.ndarray, 
      model_normals: np.ndarray, 
      symmetry_tfs: np.ndarray = None, 
      mesh: trimesh.Trimesh = None):

    """
    Resets the object model and its properties.

    Args:
        model_pts (np.ndarray): The 3D points of the object model. (i.e. mesh.vertices.copy())
        model_normals (np.ndarray): The normals of the object model. (i.e. mesh.vertex_normals.copy())
        symmetry_tfs (np.ndarray, optional): The symmetry transformations of the object model. Defaults to None.
        mesh (trimesh.Trimesh, optional): The object mesh. Defaults to None.
    """
    # find geometric center of object
    max_xyz = mesh.vertices.max(axis=0)
    min_xyz = mesh.vertices.min(axis=0)
    self.model_center = (min_xyz+max_xyz)/2
    
    # center the mesh origin to its geometric center
    if mesh is not None:
      self.mesh_ori = mesh.copy()
      mesh = mesh.copy()
      mesh.vertices = mesh.vertices - self.model_center.reshape(1,3)

    model_pts = mesh.vertices

    # Determine the voxel size for downsampling the object point cloud
    # Based on the object's "size" (diameter)
    self.diameter = compute_mesh_diameter(model_pts=mesh.vertices, n_sample=10000)
    self.vox_size = max(self.diameter/20.0, 0.003)
    logging.info(f'self.diameter:{self.diameter}, vox_size:{self.vox_size}')
    self.dist_bin = self.vox_size/2
    self.angle_bin = 20  # Deg
    pcd = toOpen3dCloud(model_pts, normals=model_normals)
    pcd = pcd.voxel_down_sample(self.vox_size)

    # Set object bounds, points, and normals after centering and downsampling
    self.max_xyz = np.asarray(pcd.points).max(axis=0)
    self.min_xyz = np.asarray(pcd.points).min(axis=0)
    self.pts = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device='cuda')
    self.normals = F.normalize(torch.tensor(np.asarray(pcd.normals), dtype=torch.float32, device='cuda'), dim=-1)
    logging.info(f'self.pts:{self.pts.shape}')
    
    # Save the centered mesh and make tensors
    self.mesh_path = None
    self.mesh = mesh
    if self.mesh is not None:
      self.mesh_path = f'/tmp/{uuid.uuid4()}.obj'
      self.mesh.export(self.mesh_path)
    self.mesh_tensors = make_mesh_tensors(self.mesh)

    if symmetry_tfs is None:
      self.symmetry_tfs = torch.eye(4).float().cuda()[None]
    else:
      self.symmetry_tfs = torch.as_tensor(symmetry_tfs, device='cuda', dtype=torch.float)

    self.make_rotation_grid(min_n_views=40, inplane_step=60)
    logging.info("reset done")



  def get_tf_to_centered_mesh(self):
    """
    Returns the transformation matrix to center the object mesh.

    Returns:
        torch.Tensor: The transformation matrix.
    """
    tf_to_center = torch.eye(4, dtype=torch.float, device='cuda')
    tf_to_center[:3,3] = -torch.as_tensor(self.model_center, device='cuda', dtype=torch.float)
    return tf_to_center


  def to_device(self, s: str = 'cuda:0'):
    """
    Moves all tensors and modules to the specified device.

    Args:
        s (str, optional): The device to move to. Defaults to 'cuda:0'.
    """
    for k in self.__dict__:
      self.__dict__[k] = self.__dict__[k]
      if torch.is_tensor(self.__dict__[k]) or isinstance(self.__dict__[k], nn.Module):
        logging.info(f"Moving {k} to device {s}")
        self.__dict__[k] = self.__dict__[k].to(s)
    for k in self.mesh_tensors:
      logging.info(f"Moving {k} to device {s}")
      self.mesh_tensors[k] = self.mesh_tensors[k].to(s)
    if self.refiner is not None:
      self.refiner.model.to(s)
    if self.scorer is not None:
      self.scorer.model.to(s)
    if self.glctx is not None:
      self.glctx = dr.RasterizeCudaContext(s)


  def make_rotation_grid(
      self, 
      min_n_views: int = 40, 
      inplane_step: int = 60):
    """
    Creates a grid of rotation matrices for sampling object poses.
    Uniformly sample N_s viewpoints from an isosphere centered on the object with the camera facing the center.

    Args:
        min_n_views (int, optional): The minimum number of views to sample. Defaults to 40.
        inplane_step (int, optional): The step size for in-plane rotations. Defaults to 60.
    """
    cam_in_obs = sample_views_icosphere(n_views=min_n_views)
    logging.info(f'cam_in_obs:{cam_in_obs.shape}')
    rot_grid = []
    for i in range(len(cam_in_obs)):
      for inplane_rot in np.deg2rad(np.arange(0, 360, inplane_step)):
        cam_in_ob = cam_in_obs[i]
        R_inplane = euler_matrix(0,0,inplane_rot)
        cam_in_ob = cam_in_ob@R_inplane
        ob_in_cam = np.linalg.inv(cam_in_ob)
        rot_grid.append(ob_in_cam)

    rot_grid = np.asarray(rot_grid)
    logging.info(f"rot_grid:{rot_grid.shape}")
    rot_grid = mycpp.cluster_poses(30, 99999, rot_grid, self.symmetry_tfs.data.cpu().numpy())
    rot_grid = np.asarray(rot_grid)
    logging.info(f"after cluster, rot_grid:{rot_grid.shape}")
    self.rot_grid = torch.as_tensor(rot_grid, device='cuda', dtype=torch.float)
    logging.info(f"self.rot_grid: {self.rot_grid.shape}")


  def generate_random_pose_hypo(
      self, 
      K: np.ndarray, 
      rgb: np.ndarray, 
      depth: np.ndarray, 
      mask: np.ndarray, 
      scene_pts: torch.Tensor = None) -> torch.Tensor:
    """
    Generates random pose hypotheses for the object.

    Args:
        K (np.ndarray): The camera intrinsics matrix.
        rgb (np.ndarray): The RGB image.
        depth (np.ndarray): The depth image.
        mask (np.ndarray): The object mask.
        scene_pts (torch.Tensor[N,3], optional): The 3D points of the scene. Defaults to None.

    Returns:
        torch.Tensor: The pose hypotheses.
    """
    ob_in_cams = self.rot_grid.clone()
    center = self.guess_translation(depth=depth, mask=mask, K=K)
    ob_in_cams[:,:3,3] = torch.tensor(center, device='cuda', dtype=torch.float).reshape(1,3)
    return ob_in_cams


  def guess_translation(
      self, 
      depth: np.ndarray, 
      mask: np.ndarray, 
      K: np.ndarray) -> np.ndarray:

    """
    Guesses the translation of the object based on the depth and mask.
    Initialize the translation using the 3d point located at the median depth with the detected 2D bounding box.

    Args:
        depth (np.ndarray): The depth image.
        mask (np.ndarray): The object mask.
        K (np.ndarray): The camera intrinsics matrix.

    Returns:
        np.ndarray: The guessed translation.
    """
    vs,us = np.where(mask>0)
    if len(us)==0:
      logging.info(f'mask is all zero')
      return np.zeros((3))
    uc = (us.min()+us.max())/2.0
    vc = (vs.min()+vs.max())/2.0
    valid = mask.astype(bool) & (depth>=0.1)
    if not valid.any():
      logging.info(f"valid is empty")
      return np.zeros((3))

    zc = np.median(depth[valid])
    center = (np.linalg.inv(K)@np.asarray([uc,vc,1]).reshape(3,1))*zc

    if self.debug>=2:
      pcd = toOpen3dCloud(center.reshape(1,3))
      o3d.io.write_point_cloud(f'{self.debug_dir}/init_center.ply', pcd)

    return center.reshape(3)


  def register(
      self, 
      K: np.ndarray, 
      rgb: np.ndarray, 
      depth: np.ndarray, 
      ob_mask: np.ndarray, 
      ob_id: tuple = None, 
      glctx: dr.RasterizeCudaContext = None, 
      iteration: int = 5) -> np.ndarray:

    """
    Registers the object pose in the scene. Compute pose from given pts to self.pcd

    Note: @pts: (N,3) np array, downsampled scene points

    Args:
        K (np.ndarray): The camera intrinsics matrix.
        rgb (np.ndarray): The RGB image.
        depth (np.ndarray): The depth image.
        ob_mask (np.ndarray): The object mask.
        ob_id (tuple, optional): The object ID. Defaults to None.
        glctx (dr.RasterizeCudaContext, optional): The rendering context. Defaults to None.
        iteration (int, optional): The number of refinement iterations. Defaults to 5.

    Returns:
        np.ndarray: The estimated object pose.
    """
    set_seed(0)
    logging.info('Welcome')
    
    # Get or create the rendering context
    if self.glctx is None:
      if glctx is None:
        self.glctx = dr.RasterizeCudaContext()
        # self.glctx = dr.RasterizeGLContext()
      else:
        self.glctx = glctx

    # Process the depth map
    depth = erode_depth(depth, radius=2, device='cuda')
    depth = bilateral_filter_depth(depth, radius=2, device='cuda')
    if self.debug>=2:
      xyz_map = depth2xyzmap(depth, K)
      valid = xyz_map[...,2]>=0.1
      pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
      o3d.io.write_point_cloud(f'{self.debug_dir}/scene_raw.ply',pcd)
      cv2.imwrite(f'{self.debug_dir}/ob_mask.png', (ob_mask*255.0).clip(0,255))

    # Check that there is depth detected where the object is detected
    # Return dummy pose if not enough valid values
    normal_map = None
    valid = (depth>=0.1) & (ob_mask>0) 
    if valid.sum()<4: 
      logging.info(f'valid too small, return')
      pose = np.eye(4)
      pose[:3,3] = self.guess_translation(depth=depth, mask=ob_mask, K=K)
      return pose
    
    if self.debug>=2:
      imageio.imwrite(f'{self.debug_dir}/color.png', rgb)
      cv2.imwrite(f'{self.debug_dir}/depth.png', (depth*1000).astype(np.uint16))
      valid = xyz_map[...,2]>=0.1
      pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
      o3d.io.write_point_cloud(f'{self.debug_dir}/scene_complete.ply',pcd)
    
    self.H, self.W = depth.shape[:2]
    self.K = K
    self.ob_id = ob_id
    self.ob_mask = ob_mask

    # Generate pose initializations
    poses = self.generate_random_pose_hypo(K=K, rgb=rgb, depth=depth, mask=ob_mask, scene_pts=None)
    poses = poses.data.cpu().numpy()
    logging.info(f'poses:{poses.shape}')
    #below is weird... isn't this already done in generate_random_pose_hypo?
    center = self.guess_translation(depth=depth, mask=ob_mask, K=K)
    poses = torch.as_tensor(poses, device='cuda', dtype=torch.float)
    poses[:,:3,3] = torch.as_tensor(center.reshape(1,3), device='cuda') 
    
    add_errs = self.compute_add_err_to_gt_pose(poses)
    logging.info(f"after viewpoint, add_errs min:{add_errs.min()}")

    xyz_map = depth2xyzmap(depth, K)

    #Pose Refinement
    poses, vis = self.refiner.predict(mesh=self.mesh, 
                                      mesh_tensors=self.mesh_tensors, 
                                      rgb=rgb, 
                                      depth=depth, 
                                      K=K, 
                                      ob_in_cams=poses.data.cpu().numpy(),
                                      normal_map=normal_map, 
                                      xyz_map=xyz_map, #difference!
                                      glctx=self.glctx, 
                                      mesh_diameter=self.diameter, 
                                      iteration=iteration, 
                                      get_vis=self.debug>=2
                                      )
    if vis is not None: #save debug image
      imageio.imwrite(f'{self.debug_dir}/vis_refiner.png', vis)
    
    #Pose Ranking
    scores, vis = self.scorer.predict(mesh=self.mesh, 
                                      rgb=rgb, 
                                      depth=depth, 
                                      K=K, 
                                      ob_in_cams=poses.data.cpu().numpy(), 
                                      normal_map=normal_map, 
                                      xyz_map=xyz_map, #ADDED
                                      mesh_tensors=self.mesh_tensors, 
                                      glctx=self.glctx, 
                                      mesh_diameter=self.diameter, 
                                      get_vis=self.debug>=2)
    
    if vis is not None: #save debug image
      imageio.imwrite(f'{self.debug_dir}/vis_score.png', vis)

    add_errs = self.compute_add_err_to_gt_pose(poses)
    logging.info(f"final, add_errs min:{add_errs.min()}")

    ids = torch.as_tensor(scores).argsort(descending=True)
    logging.info(f'sort ids:{ids}')
    scores = scores[ids]
    poses = poses[ids]

    logging.info(f'sorted scores:{scores}')

    best_pose = poses[0]@self.get_tf_to_centered_mesh()
    self.pose_last = poses[0]
    self.best_id = ids[0]

    self.poses = poses
    self.scores = scores

    return best_pose.data.cpu().numpy()


  def compute_add_err_to_gt_pose(self, poses):
    '''
    @poses: wrt. the centered mesh
    '''
    return -torch.ones(len(poses), device='cuda', dtype=torch.float)


  def track_one(self, rgb, depth, K, iteration, extra={}):
    if self.pose_last is None:
      logging.info("Please init pose by register first")
      raise RuntimeError
    logging.info("Welcome")

    depth = torch.as_tensor(depth, device='cuda', dtype=torch.float)
    depth = erode_depth(depth, radius=2, device='cuda')
    depth = bilateral_filter_depth(depth, radius=2, device='cuda')
    logging.info("depth processing done")

    xyz_map = depth2xyzmap_batch(depth[None], torch.as_tensor(K, dtype=torch.float, device='cuda')[None], zfar=np.inf)[0]

    pose, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K, ob_in_cams=self.pose_last.reshape(1,4,4).data.cpu().numpy(), normal_map=None, xyz_map=xyz_map, mesh_diameter=self.diameter, glctx=self.glctx, iteration=iteration, get_vis=self.debug>=2)
    logging.info("pose done")
    if self.debug>=2:
      extra['vis'] = vis
    self.pose_last = pose
    return (pose@self.get_tf_to_centered_mesh()).data.cpu().numpy().reshape(4,4)

  def refine_only(self, pose, rgb, depth, K, iterations=10):
    pose = pose@np.linalg.inv(self.get_tf_to_centered_mesh().data.cpu().numpy())
    pose = torch.as_tensor(pose, device='cuda', dtype=torch.float)
    logging.info("Welcome")

    
    depth = erode_depth(depth, radius=2, device='cuda')
    depth = bilateral_filter_depth(depth, radius=2, device='cuda')
    logging.info("depth processing done")

    xyz_map = depth2xyzmap(depth, K)
    depth = torch.as_tensor(depth, device='cuda', dtype=torch.float)

    pose, vis, runtimes = self.refiner.predict(
      mesh=self.mesh, 
      mesh_tensors=self.mesh_tensors, 
      rgb=rgb, 
      depth=depth, 
      K=K, 
      ob_in_cams=pose.reshape(1,4,4), 
      normal_map=None, 
      xyz_map=xyz_map, 
      mesh_diameter=self.diameter, 
      glctx=self.glctx, 
      iteration=iterations, 
      get_vis=self.debug>=2)
    
    if vis is not None: #save debug image
      imageio.imwrite(f'{self.debug_dir}/vis_refiner.png', vis)
    
    logging.info("pose done")
    return (pose@self.get_tf_to_centered_mesh()).data.cpu().numpy().reshape(4,4), runtimes
