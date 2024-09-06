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
from intrinsic_ipd import IPDReader, IPDCamera, IPDImage
from intrinsic_ipd.constants import CameraFramework, IPDLightCondition
from intrinsic_ipd.constants import DATASET_IDS

from functools import lru_cache

# def visualize(reader, res):
#   for scene_id in res:

#     if reader.camera == IPDCamera.PHOTONEO:
#       color = reader.get_img(scene_id, image_type=IPDImage.PHOTONEO_HDR)
#     else:
#       color = reader.get_img(scene_id, image_type=IPDImage.EXPOSURE_200)

#     for part in res[scene_id]: 
#       # add code here, to get mesh
#       mesh = reader.get_mesh(part)
#       mesh.vertices = mesh.vertices * 1e-3 
#       to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
#       bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
  
#       for instance in res[scene_id][part]:
#         pose = res[scene_id][part][instance]
#         center_pose = pose@np.linalg.inv(to_origin)
  
#         color = draw_posed_3d_box(reader.cam_K, img=color, ob_in_cam=center_pose, bbox=bbox)
#         color = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.cam_K, thickness=2, transparency=0, is_input_rgb=True)
    
#     # cv2.imshow('1', color)
#     # cv2.waitKey(1)
#     cv2.imwrite(f'{opt.debug_dir}/renders/{scene_id}.png', color)
#     # cv2.destroyAllWindows()


def run_pose_estimation_worker(
    reader : IPDReader, 
    part:str, 
    est:FoundationPose,
    debug:int,
    device='cuda:0'
  ):

  torch.cuda.set_device(device)
  est.to_device(device)
  est.glctx = dr.RasterizeCudaContext(device=device)

  result = NestDict()

  scenes = list(reader.scenes.keys())
  if debug >= 2:
    scenes = scenes[:1]
    logging.debug(f"scenes: {scenes}")

  for i, scene_id in enumerate(scenes):

    if reader.camera == IPDCamera.PHOTONEO:
      depth = reader.get_depth(scene_id) * 1e-3
      color = reader.get_img(scene_id, image_type=IPDImage.PHOTONEO_HDR)
    else:
      raise Exception(f"No depth file available for {reader.camera}")
      # color = reader.get_img(scene_id, image_type=IPDImage.EXPOSURE_200).convert('RGB')

    instances = [object[1] for object in reader.objects if object[0]==part]
    for instance in instances:
      
      logging.info(f"{i}/{len(scenes)}, scene_id:{scene_id}, part:{part}, instance:{instance}")

      try:
        mask = reader.get_mask(scene_id, part, instance, detect_bounding_box=False)
      except:
        logging.info("ob_mask not found, skip")
        result[reader.dataset_id][reader.camera.name][scene_id][part][instance] = np.eye(4)
        return result

      est.gt_pose = reader.o2c.sel(scene=scene_id, part=part, instance=instance)

      pose = est.register(K=reader.cam_K, rgb=color, depth=depth, ob_mask=mask, ob_id=(part,instance))

      result[reader.dataset_id][reader.camera.name][scene_id][part][instance] = pose

  return result



def run_pose_estimation(dataset_ids:list[int]):

  wp.force_load(device='cuda')
  camera = opt.camera_id
  root_dir = opt.root_dir
  debug = opt.debug
  debug_dir = opt.debug_dir
  glctx = dr.RasterizeCudaContext()

  mesh_tmp = trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4)).to_mesh()
  est = FoundationPose(
    model_pts=mesh_tmp.vertices.copy(), 
    model_normals=mesh_tmp.vertex_normals.copy(), 
    symmetry_tfs=None, mesh=mesh_tmp, scorer=None, refiner=None, glctx=glctx, debug_dir=debug_dir, debug=debug)
  
  for dataset_id in dataset_ids:

    if os.path.exists(f'{opt.root_dir}/../results/{dataset_id}_{camera}.yml'):
      #skip if already exists
      logging.critical(f"Results for {dataset_id} already exist. Skipping.")
      continue
    
    res = NestDict()
    reader = IPDReader(root_dir, dataset_id, IPDCamera[camera], IPDLightCondition.ROOM, download=False)

    parts = reader.parts
    if debug >= 2:
      parts = [parts[1]]
      logging.debug(f"parts: {parts}")

    outs = []
    for part in parts:
      logging.info(f"Pose estimation for part:{part}")
      
      try:
        mesh = reader.get_mesh(part)
      except:
        continue

      mesh.vertices = mesh.vertices * 1e-3
    
      est.reset_object(
        model_pts=mesh.vertices.copy(), 
        model_normals=mesh.vertex_normals.copy(), 
        mesh=mesh,
        symmetry_tfs=None)

      out = run_pose_estimation_worker(
            reader = reader,
            est = est,
            part = part, 
            debug = debug,
            device ="cuda:0")
      
      outs.append(out)

    
      # result[reader.dataset_id][reader.camera.name][scene_id][part][part_instance]
    for out in outs:
      for scene in out[reader.dataset_id][reader.camera.name]:
        for part in out[reader.dataset_id][reader.camera.name][scene]:
          for instance in out[reader.dataset_id][reader.camera.name][scene][part]:
            res[scene][part][instance] = out[reader.dataset_id][reader.camera.name][scene][part][instance]

    # visualize(reader, res)

    with open(f'{opt.root_dir}/../results/{dataset_id}_{camera}.yml','w') as ff:
      yaml.safe_dump(make_yaml_dumpable(res), ff)


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--dataset_id', type=str, default="ALL", help="dataset id")
  parser.add_argument('--camera_id', type=str, default="PHOTONEO")
  parser.add_argument('--root_dir', type=str, default="../ipd/datasets")
  parser.add_argument('--debug', type=int, default=0)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug/ipd')
  opt = parser.parse_args()
  set_seed(0)

  detect_type = 'mask'   # mask / box / detected
  if opt.dataset_id == "ALL":
    run_pose_estimation(DATASET_IDS)
  else:
    run_pose_estimation([opt.dataset_id])

