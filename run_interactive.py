import argparse
import pyrealsense2 as rs
from estimater import *
from make_mask import create_mask
import tkinter as tk
from tkinter import filedialog
from offscreen_renderer import ModelRendererOffscreen

parser = argparse.ArgumentParser()
code_dir = os.path.dirname(os.path.realpath(__file__))
parser.add_argument('--est_refine_iter', type=int, default=5)
parser.add_argument('--track_refine_iter', type=int, default=2)
parser.add_argument('--debug', type=int, default=0)
args = parser.parse_args()

set_logging_format()
set_seed(0)

root = tk.Tk()
root.withdraw()



mesh_path = filedialog.askopenfilename()
if not mesh_path:
    print("No mesh file selected")
    exit(0)
mask_file_path = create_mask()

mesh = trimesh.load(mesh_path)
mesh.vertices = mesh.vertices * 1e-3
to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

# fun_mesh_path = filedialog.askopenfilename()
# if not fun_mesh_path:
#     fun_mesh_path = "./pikachu/object/model.obj"
fun_mesh_path = "./pikachu/object/model.obj"

fun_mesh = trimesh.load(fun_mesh_path, force='mesh')
fun_mesh.apply_transform(trimesh.transformations.scale_matrix(.25))
R = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0], [0, 0, 0])
fun_mesh.apply_transform(R)
R = trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0], [0, 0, 0])
fun_mesh.apply_transform(R)
if fun_mesh_path.endswith("gltf"):
    fun_mesh = fun_mesh.geometry[list(fun_mesh.geometry.keys())[0]]  
# fun_mesh.vertices = fun_mesh.vertices * 1e-3

cam_K = np.array([[381.81698608, 0.0, 322.36413574],
                  [0.0,381.50231934, 239.3505249 ],
                  [0.0,0.0,1.0]])
renderer = ModelRendererOffscreen(cam_K, 480, 640)
renderer.add_point_light(50)

glctx = dr.RasterizeCudaContext()
est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, 
                     scorer=None, refiner=None, 
                     glctx=glctx,
                     debug_dir="./debug",
                     debug=args.debug
                     )
pipeline = rs.pipeline()
config = rs.config()
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale
align_to = rs.stream.color
align = rs.align(align_to)

i = 0

mask = cv2.imread(mask_file_path, cv2.IMREAD_UNCHANGED)
print("MASK SHAPE", mask.shape)
# cam_K = np.array( [[615.37701416, 0., 313.68743896],
#                    [0., 615.37701416, 259.01800537],
#                    [0., 0., 1.]])

cam_K = None
Estimating = True
time.sleep(3)
# Streaming loop
window_name = "Display1"
# cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
# cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
fun = False
try:
    while Estimating:

        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not aligned_depth_frame or not color_frame:
            continue
        
        # if cam_K is None:
        intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        cam_K = np.array( [
                [intrinsics.fx, 0., intrinsics.ppx],
                [0., intrinsics.fy, intrinsics.ppy],
                [0., 0., 1.]])
        
        print(cam_K)

        depth_image = np.asanyarray(aligned_depth_frame.get_data()).astype(np.float32) #/1e3
        color_image = np.asanyarray(color_frame.get_data())
        depth_image_scaled = (depth_image * depth_scale).astype(np.float32)

        # Remove background
        grey = 153
        depth_image_3d = np.dstack(
            (depth_image_scaled, depth_image_scaled, depth_image_scaled)
        )
        bg_removed = np.where(
            (depth_image_3d > clipping_distance) | (depth_image_3d <= 0),
            grey,
            color_image
        )
        
        # if cv2.waitKey(1) == 13:
        #     Estimating = False
        #     break       

        # H, W = color_image.shape[:2]
        # color = cv2.resize(color_image, (W,H), interpolation=cv2.INTER_NEAREST)
        # depth = cv2.resize(depth_image, (W,H), interpolation=cv2.INTER_NEAREST)
        color = color_image
        depth = depth_image_scaled
        depth[(depth<0.1) | (depth>=np.inf)] = 0
        
        if i==0:
            if len(mask.shape)==3:
                for c in range(3):
                    if mask[...,c].sum()>0:
                        mask = mask[...,c]
                        break
            # mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)
            mask = mask.astype(bool).astype(np.uint8)
        
            pose = est.register(K=cam_K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
        else:
            pose = est.track_one(rgb=color, depth=depth, K=cam_K, iteration=args.track_refine_iter)

        
        
        if not fun:
            render, _ = renderer.render(mesh, pose)  
        else:
            render, _ = renderer.render(fun_mesh, pose)

        render = cv2.cvtColor(render, cv2.COLOR_BGRA2RGBA)  
        # center_pose = pose@np.linalg.inv(to_origin)
        # vis = draw_posed_3d_box(cam_K, img=color, ob_in_cam=center_pose, bbox=bbox)
        # vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=cam_K, thickness=3, transparency=0, is_input_rgb=True)
        # cv2.imshow(window_name, vis[...,::-1])\

        # color = cv2.cvtColor(color, cv2.COLOR_RGB2RGBA)
        # alpha = 1
        # overlay = cv2.addWeighted(color, 1-alpha, render, alpha, 0)
        
        alpha = render[:,:,3]
        alpha = cv2.merge([alpha, alpha, alpha])
        front = render[:,:,:3]

        overlay = np.where(alpha==(0,0,0), color, front)

        
        cv2.imshow(window_name, overlay)
        # cv2.imshow(window_name, render)
        # cv2.waitKey(1)  
        
        # Switch rendering mesh with space bar
        key = cv2.waitKey(1)
        if key & 0xFF == ord(" "):
            fun = not fun
            print("FUN MODE=", fun)

        if key & 0xFF == ord("q") or key == 27:
            Estimating = False
            cv2.destroyAllWindows()
            

        i += 1
        
finally:
    pipeline.stop()