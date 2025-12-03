# Libraries
import cv2
import os
import numpy as np
import pyrealsense2
from ultralytics import YOLO

# Define some constants 
RESOLUTION_WIDTH = 640 # pixels
RESOLUTION_HEIGHT = 480 # pixels
FRAME_RATE = 30  # fps

# file locations
os.chdir("../")
ROOT_DIRECTORY = os.getcwd()
WEIGHTS_LOCATION = ROOT_DIRECTORY + '/Transfer to ORIN/train4/weights/best.pt'
YOLO_TEST = 'yolov8n.pt'
# The below code will be added once we switch to our own weights from a custom data
#MODEL = YOLO(os.path.join(ROOT_DIRECTORY,WEIGHTS_LOCATION))
MODEL = YOLO(YOLO_TEST)
def findDevices():
    contexts = pyrealsense2.context() # Create librealsense context for managing devices
    serials = []
    cameras = contexts.query_devices()
    print("resetting devices")
    for cam in cameras:
        cam.hardware_reset()
        print("reset device: " + cam.get_info(pyrealsense2.camera_info.name))
    print("reset done, enabling stream")
    if (len(contexts.devices) > 0):
        for dev in contexts.devices:
            print ('Found device: ', \
                    dev.get_info(pyrealsense2.camera_info.name), ' ', \
                    dev.get_info(pyrealsense2.camera_info.serial_number))
            serials.append(dev.get_info(pyrealsense2.camera_info.serial_number))
    else:
        print("No Intel Device connected")
        
    return serials, contexts

def enableDevices(serials, contexts, RESOLUTION_WIDTH,RESOLUTION_HEIGHT, FRAME_RATE):

    pipelines = []
    for serial in serials:
        pipe = pyrealsense2.pipeline(contexts)
        configs = pyrealsense2.config()
        configs.enable_device(serial)
        configs.enable_stream(pyrealsense2.stream.depth, RESOLUTION_WIDTH, RESOLUTION_HEIGHT, pyrealsense2.format.z16, FRAME_RATE)
        configs.enable_stream(pyrealsense2.stream.color, RESOLUTION_WIDTH, RESOLUTION_HEIGHT, pyrealsense2.format.bgr8, FRAME_RATE)
        pipe.start(configs)
        pipelines.append([serial,pipe])
        
    return pipelines

def Visualize(pipelines):
    MODEL.to('cuda')
    print(MODEL.device.type)
    align_to = pyrealsense2.stream.color
    align = pyrealsense2.align(align_to)

    for (device,pipe) in pipelines:
        # Get frameset of color and depth
        frames = pipe.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        if color_frame:

	    # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue    

            color_image = np.asanyarray(color_frame.get_data())
            results = MODEL.predict(source=color_image)
            boxes = None

            for result in results:
                boxes = result.boxes

            for box in boxes:
                box_copy = box.xyxy[0]
                confidence = box.cls
                top_left_coordinates = (int(box_copy[0]), int(box_copy[1]))
                bottom_right_coordinates = (int(box_copy[2]), int(box_copy[3]))
                color_red = (0, 0, 255)
                font_face = cv2.FONT_HERSHEY_SIMPLEX
                line_type = cv2.LINE_4

                cv2.rectangle(color_image, top_left_coordinates, bottom_right_coordinates, color_red, 2, line_type)
                cv2.putText(color_image, MODEL.names[int(confidence)], top_left_coordinates,
                            font_face, 0.7, color_red, 2, line_type)

                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                annotated_frame = results.plot()

                cv2.imshow(f"RealSense {device}", annotated_frame)
                key = cv2.waitKey(1)

                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    return True

                if key == 115:  # 's'
                    cv2.imwrite(f"{device}_aligned_depth.png", depth_image)
                    cv2.imwrite(f"{device}_aligned_color.png", color_image)
                    print("Save")

def pipelineStop(pipelines):
    for (device,pipe) in pipelines:
        # Stop streaming
        pipe.stop() 
        
# -------Main program--------

def main():
    serials, contexts = findDevices()
    pipelines = enableDevices(serials, contexts, RESOLUTION_WIDTH, RESOLUTION_HEIGHT, FRAME_RATE)

    try:
        while True:
            exit = Visualize(pipelines)
            if exit == True:
                print('Program closing...')
                break
    finally:
        pipelineStop(pipelines)

if __name__ == '__main__':
    main()
