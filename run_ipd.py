# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from __future__ import annotations

from Utils import *
import json,uuid,joblib,os,sys
import scipy.spatial as spatial

from multiprocessing import Pool
import multiprocessing
from functools import partial
from itertools import repeat
import itertools
from datareader import *
from estimater import *
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/mycpp/build')


import intrinsic_ipd as ipd
from intrinsic_ipd import IPDataset, IPDCamera, IPDImage
from intrinsic_ipd.constants import CameraFramework, IPDLightCondition

from functools import lru_cache

RESCALE = 1

def visualize(reader, res):
  for scene_id in res:

    if reader.camera == IPDCamera.PHOTONEO:
      color = reader.get_img(scene_id, image_type=IPDImage.PHOTONEO_HDR)
    else:
      color = reader.get_img(scene_id, image_type=IPDImage.EXPOSURE_200)

    for object_name in res[scene_id]: 
      # add code here, to get mesh
      mesh_file = reader.get_mesh_file(object_name)
      mesh = trimesh.load(mesh_file)
      mesh.vertices = mesh.vertices * 1e-3 
      to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
      bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
  
      for object_id in res[scene_id][object_name]:
        pose = res[scene_id][object_name][object_id]
        center_pose = pose@np.linalg.inv(to_origin)
  
        color = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
        color = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=2, transparency=0, is_input_rgb=True)
    
    # cv2.imshow('1', color)
    # cv2.waitKey(1)
    cv2.imwrite(f'{opt.debug_dir}/renders/{scene_id}.png', color)
    # cv2.destroyAllWindows()



def run_pose_estimation_worker(
    reader : IPDataset, 
    scene_ids : list[int], 
    object_name:str, 
    est:FoundationPose, 
    debug=0, 
    device='cuda:0'
  ):

  torch.cuda.set_device(device)
  est.to_device(device)
  est.glctx = dr.RasterizeCudaContext(device=device)

  result = NestDict()

  for i, scene_id in enumerate(scene_ids):

    if reader.camera == IPDCamera.PHOTONEO:
      depth = reader.get_depth(scene_id) * 1e-3
      color = reader.get_img(scene_id, image_type=IPDImage.PHOTONEO_HDR)
    else:
      raise Exception(f"No depth file available for {reader.camera}")
      # color = reader.get_img(scene_id, image_type=IPDImage.EXPOSURE_200).convert('RGB')
    
    H,W = color.shape[:2]

    debug_dir =est.debug_dir

    part_object_ids = [id for part, id in reader.objects if part==object_name]
    for object_id in part_object_ids:
      logging.info(f"{i}/{len(scene_ids)}, scene_id:{scene_id}, object_name:{object_name}, object_id:{object_id}")

      mask = reader.get_mask(scene_id, object_name, object_id, detect_bounding_box=False)

      if mask is None:
        logging.info("ob_mask not found, skip")
        result[reader.dataset_id][reader.camera.name][scene_id][object_name][object_id] = np.eye(4)
        return result

      _ , o2c_by_object = reader.get_scene_labels(scene_id)
      est.gt_pose = o2c_by_object[(object_name, object_id)]

      pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, ob_id=(object_name,object_id))
    

      result[reader.dataset_id][reader.camera.name][scene_id][object_name][object_id] = pose

      if debug>=1:
        logging.info(f"pose:\n{pose}")
        mesh_file = reader.get_mesh_file(object_name)
        mesh = trimesh.load(mesh_file)
        mesh.vertices = mesh.vertices * 1e-3 
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
        center_pose = pose@np.linalg.inv(to_origin)
        vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
        cv2.imwrite(f'{opt.debug_dir}/renders/object/{object_name}_{object_id}_{scene_id}.png', color)
      
      # if debug >= 2:
        # cv2.imshow('1', vis)
        # cv2.waitKey(1)
      
      # cv2.destroyAllWindows()

      if debug>=3:
        m = est.mesh_ori.copy()
        tmp = m.copy()
        tmp.apply_transform(pose)
        tmp.export(f'{debug_dir}/model_tf.obj')

  return result



def run_pose_estimation():
  wp.force_load(device='cuda')
  dataset_id = opt.dataset_id
  camera = opt.camera_id
  root_dir = opt.root_dir
  reader = IPDataset(root_dir, dataset_id, IPDCamera[camera], IPDLightCondition.ROOM, resize=RESCALE, download=True)

  debug = opt.debug
  debug_dir = opt.debug_dir

  res = NestDict()
  glctx = dr.RasterizeCudaContext()

  mesh_tmp = trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4)).to_mesh()
  est = FoundationPose(
    model_pts=mesh_tmp.vertices.copy(), 
    model_normals=mesh_tmp.vertex_normals.copy(), 
    symmetry_tfs=None, mesh=mesh_tmp, scorer=None, refiner=None, glctx=glctx, debug_dir=debug_dir, debug=debug)

  parts = list(set([part for part, _ in reader.objects]))
  if debug >= 2:
    parts = [parts[1]]
    logging.debug(f"parts: {parts}")

  for part in parts:
    mesh_file = reader.get_mesh_file(part)
    mesh = trimesh.load(mesh_file)
    mesh.vertices = mesh.vertices * 1e-3
    

    args = []
    est.reset_object(
      model_pts=mesh.vertices.copy(), 
      model_normals=mesh.vertex_normals.copy(), 
      mesh=mesh,
      symmetry_tfs=None)
    
    scenes = list(reader.scenes.keys())
    if debug >= 2:
      scenes = scenes[:1]
      logging.debug(f"scenes: {scenes}")

    for scene_id in scenes: 
      args.append({
          "reader": reader,
          "scene_ids": [scene_id], 
          "est": est, 
          "debug": debug, 
          "object_name": part, 
          "device": "cuda:0"
        })

    outs = []
    for arg in args:
      out = run_pose_estimation_worker(**arg)
      outs.append(out)

    
    # result[reader.dataset_id][reader.camera.name][scene_id][object_name][object_id]
    for out in outs:
      for scene in out[reader.dataset_id][reader.camera.name]:
        for object_name in out[reader.dataset_id][reader.camera.name][scene]:
          for object_id in out[reader.dataset_id][reader.camera.name][scene][object_name]:
            res[scene][object_name][object_id] = out[reader.dataset_id][reader.camera.name][scene][object_name][object_id]

  visualize(reader, res)

  with open(f'{opt.debug_dir}/{dataset_id}_{camera}.yml','w') as ff:
    yaml.safe_dump(make_yaml_dumpable(res), ff)


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--dataset_id', type=str, default="dataset_darkbg_0", help="dataset id")
  parser.add_argument('--camera_id', type=str, default="PHOTONEO")
  parser.add_argument('--root_dir', type=str, default="../ipd/datasets")
  parser.add_argument('--debug', type=int, default=0)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug/ipd')
  opt = parser.parse_args()
  set_seed(0)

  detect_type = 'mask'   # mask / box / detected

  run_pose_estimation()
