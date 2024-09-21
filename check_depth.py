import pyrealsense2 as rs
import numpy as np
import cv2

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
# different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
print(device_product_line)
found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
if device_product_line == "L500":
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

clipping_distance = 1 / depth_scale
print("Clipping distance is: ", clipping_distance)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)
cam_K = None

# Skip first frames for syncer and autoexposure
for _ in range(10): 
    frames = pipeline.wait_for_frames()

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        

        # Get aligned frames
        aligned_depth_frame = (
            aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        )
        color_frame = aligned_frames.get_color_frame()

        if cam_K is None:
            intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
            cam_K = np.array( [
                   [intrinsics.fx, 0., intrinsics.ppx],
                   [0., intrinsics.fy, intrinsics.ppy],
                   [0., 0., 1.]])
            print(cam_K)

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue
        
        colorizer = rs.colorizer()
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_color = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

        # Remove background
        grey = 153
        depth_image_3d = np.dstack(
            (depth_image, depth_image, depth_image)
        )
        bg_removed = np.where(
            (depth_image_3d > clipping_distance) | (depth_image_3d <= 0),
            grey,
            color_image
        )

        # Scale depth image to mm
        depth_image_scaled = (depth_image * depth_scale * 1000).astype(np.uint16)
        # depth_dist = cv2.applyColorMap(depth_image_scaled, cv2.COLORMAP_HSV)
        cv2.imshow('color', color_image)
        cv2.imshow('depth', depth_image)
        cv2.imshow('depth_color', depth_color)
        # cv2.imshow('depth', depth_image_scaled)
        cv2.waitKey(1)

        
finally:
    pipeline.stop()