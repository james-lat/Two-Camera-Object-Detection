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
WEIGHTS_LOCATION = ROOT_DIRECTORY + '/Two-camera-object-detection/tomato_segmentation_data/runs/segment/train7/weights/best.pt '
YOLO_TEST = 'yolov8n-seg.pt'
# The below code will be added once we switch to our own weights from a custom data
MODEL = YOLO(os.path.join(ROOT_DIRECTORY,WEIGHTS_LOCATION))
#MODEL = YOLO(YOLO_TEST)
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
        configs.enable_stream(pyrealsense2.stream.color, RESOLUTION_WIDTH, RESOLUTION_HEIGHT, pyrealsense2.format.bgr8, FRAME_RATE)
        pipe.start(configs)
        pipelines.append([serial,pipe])
        
    return pipelines

def Visualize(pipelines):
    MODEL.to('cuda')
    # print(MODEL.device.type) # Optional: Commented out to reduce clutter
    align_to = pyrealsense2.stream.color
    align = pyrealsense2.align(align_to)

    for (device,pipe) in pipelines:
        # Get frameset of color
        frames = pipe.wait_for_frames()
        aligned_frames = align.process(frames)

        # Get aligned frames
        color_frame = aligned_frames.get_color_frame()
        
        if color_frame:
            color_image = np.asanyarray(color_frame.get_data())
            
            results = MODEL.predict(source=color_image, verbose=False)
            
            annotated_frame = color_image 
            
            for result in results:
                boxes = result.boxes
                
                if boxes is not None:
                    for box in boxes:
                        box_copy = box.xyxy[0]
                        confidence = box.cls
                        top_left_coordinates = (int(box_copy[0]), int(box_copy[1]))
                        bottom_right_coordinates = (int(box_copy[2]), int(box_copy[3]))
                        color_red = (0,0,255)
                        font_face = cv2.FONT_HERSHEY_SIMPLEX
                        line_type = cv2.LINE_4

                        cv2.rectangle(color_image, top_left_coordinates, bottom_right_coordinates, color_red, thickness=2, lineType=line_type)
                        cv2.putText(color_image, text=MODEL.names[int(confidence)], org=top_left_coordinates, fontFace=font_face, fontScale=0.7, color=color_red, thickness=2,
                                    lineType=line_type)
                    
                    # Update the frame with the plot if boxes exist
                    annotated_frame = results[0].plot()
            
            cv2.imshow(f'RealSense {device}', annotated_frame)
            key = cv2.waitKey(1)
            
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                return True
                
            # Save images by pressing 's'
            if key == 115:
                cv2.imwrite( str(device) + '_aligned_color.png', color_image)
                print('Save')
    return False            
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
