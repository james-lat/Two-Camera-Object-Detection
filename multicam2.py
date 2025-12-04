# Libraries
import cv2
import os
import numpy as np
import pyrealsense2
from ultralytics import YOLO
import threading 
import time 
RESOLUTION_WIDTH = 640 # pixels
RESOLUTION_HEIGHT = 480 # pixels
FRAME_RATE = 30  # fps

os.chdir("../")
ROOT_DIRECTORY = os.getcwd()
WEIGHTS_LOCATION =  '/home/nvidia/tomatoe_rec/best.pt'
MODEL = YOLO(WEIGHTS_LOCATION)
MODEL.to('cuda')
print(MODEL.task)
class CameraThread(threading.Thread):
    def __init__(self, serial, pipe, align, model, depth_scale, intrinsics): 
        super().__init__()
        self.serial = serial
        self.pipe = pipe
        self.align = align
        self.model = model
        self.depth_scale = depth_scale 
        self.intrinsics = intrinsics 
        
        self.lock = threading.Lock() 
        self.latest_frame = None  
        self.latest_depth = None  
        self.latest_3d_positions = {} 
        self.running = True
    
    def stop(self):
        self.running = False
        
        time.sleep(0.05)
        
        try:
            self.pipe.stop()
        except:
            pass    
    
    def run(self):
        while self.running:
            try:
                frames = self.pipe.wait_for_frames()
                aligned_frames = self.align.process(frames)

                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not aligned_depth_frame or not color_frame:
                    continue    

                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                
                results = self.model.predict(source=color_image, verbose=False)
                
                boxes = None
                for result in results:
                    boxes = result.boxes
                    masks = result.masks
                
                annotated_frame = color_image.copy() 
                current_3d_positions = {} 

                if masks is not None:
                    mask_array = masks.data.cpu().numpy()  

                    for mask in mask_array:
                        m = (mask * 255).astype(np.uint8)

                        colored_mask = np.zeros_like(color_image)
                        colored_mask[:, :, 1] = m  

                        annotated_frame = cv2.addWeighted(
                            annotated_frame, 1.0,
                            colored_mask, 0.5,
                            0
                        )


                if boxes is not None:
                    for box in boxes:
                        box_copy = box.xyxy[0].cpu().numpy() 
                        class_id = int(box.cls[0].item())
                        class_name = self.model.names[class_id]

                        u = int((box_copy[0] + box_copy[2]) / 2) 
                        v = int((box_copy[1] + box_copy[3]) / 2) 

                        
                        
                        if 0 <= u < RESOLUTION_WIDTH and 0 <= v < RESOLUTION_HEIGHT:
                            depth_value = aligned_depth_frame.get_distance(u, v)

                            if depth_value > 0:
                                point_3d = pyrealsense2.rs2_deproject_pixel_to_point(
                                    self.intrinsics, [u, v], depth_value)
                                
                                x_meters = point_3d[0]
                                y_meters = point_3d[1]
                                z_meters = point_3d[2]

                                current_3d_positions[class_name] = (x_meters, y_meters, z_meters)
                                
                                text_3d = f"X: {x_meters:.2f}m, Y: {y_meters:.2f}m, Z: {z_meters:.2f}m"
                                text_pos = (int(box_copy[0]), int(box_copy[3] + 25)) 
                                cv2.putText(annotated_frame, text_3d, text_pos, 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                                cv2.circle(annotated_frame, (u, v), 5, (0, 255, 0), -1) 
                            else:
                                current_3d_positions[class_name] = (0.0, 0.0, 0.0) 
                        
                        top_left_coordinates = (int(box_copy[0]), int(box_copy[1]))
                        bottom_right_coordinates = (int(box_copy[2]), int(box_copy[3]))
                        color_red = (0, 0, 255)
                        font_face = cv2.FONT_HERSHEY_SIMPLEX
                        line_type = cv2.LINE_4

                        cv2.rectangle(annotated_frame, top_left_coordinates, bottom_right_coordinates, color_red, 2, line_type)
                        cv2.putText(annotated_frame, class_name, (int(box_copy[0]), int(box_copy[1] - 10)),
                                    font_face, 0.7, color_red, 2, line_type)
                        
                with self.lock:
                    self.latest_frame = annotated_frame
                    self.latest_depth = depth_image
                    self.latest_3d_positions = current_3d_positions # Store 3D data

            except Exception as e:
                # print(f"Error in camera {self.serial} thread: {e}")
                pass


def findDevices():
    contexts = pyrealsense2.context() # Create librealsense context for managing devices
    serials = []
    cameras = contexts.query_devices()
    print("resetting devices")
    for cam in cameras:
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

def Visualize(camera_threads):
    frames_to_save = {}

    for thread in camera_threads:
        device = thread.serial
        
        with thread.lock:
            annotated_frame = thread.latest_frame
            depth_image = thread.latest_depth

        if annotated_frame is not None:
            cv2.imshow(f"RealSense {device}", annotated_frame)
            
            frames_to_save[device] = {
                'annotated': annotated_frame, 
                'depth': depth_image
            }
        
    key = cv2.waitKey(1) 

    # Check for quit
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        return True

    # Check for save
    if key == 115:  # 's'
        if frames_to_save:
            for device, data in frames_to_save.items():
                cv2.imwrite(f"{device}_annotated_color.png", data['annotated'])
                cv2.imwrite(f"{device}_aligned_depth.png", data['depth'])
            print(f"Saved frames for {len(frames_to_save)} device(s)")
        else:
            print("No valid frames to save.")

    return False

def pipelineStop(pipelines):
    pass 

def main():
    serials, contexts = findDevices()
    
    pipelines = enableDevices(serials, contexts, RESOLUTION_WIDTH, RESOLUTION_HEIGHT, FRAME_RATE)

    camera_threads = []
    align = pyrealsense2.align(pyrealsense2.stream.color) 
    MODEL.to('cuda') 

    print("Starting camera threads for parallel processing...")
    
    for serial, pipe in pipelines:
        profile = pipe.get_active_profile()
        
        device = profile.get_device() 
        depth_scale = device.first_depth_sensor().get_depth_scale()

        color_profile = pyrealsense2.video_stream_profile(profile.get_stream(pyrealsense2.stream.color))
        color_intrinsics = color_profile.get_intrinsics() ection!

        thread = CameraThread(serial, pipe, align, MODEL, depth_scale, color_intrinsics) 
        thread.start()
        camera_threads.append(thread)
    
    try:
        while True:
            exit = Visualize(camera_threads) 
            if exit == True:
                print('Program closing...')
                break
    finally:
        print("Stopping camera threads and pipelines...")
        for thread in camera_threads:
            thread.stop()
            thread.join()
        print("All resources released.")

if __name__ == '__main__':
    main()

