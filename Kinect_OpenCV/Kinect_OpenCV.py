
from pykinect2 import PyKinectRuntime as pkr
from pykinect2 import PyKinectV2 as pk2
import numpy as np
import cv2

# res1 should always be greater than res2, temporarily
def resolutionRatio(kinect, res1, res2):
    if (res1 < res2):
        return (None)
    remainder = res1 % res2
    min_stretch = (res1 - remainder) / res2
    stretch = np.full(res2, min_stretch)
    offset_points = np.rint(np.linspace(0, res2, num = remainder, endpoint = False))
    for x in offset_points:
        stretch[int(x)] += 1

    return (stretch.astype(int))

if __name__ == "__main__":
    kinect = pkr.PyKinectRuntime(pk2.FrameSourceTypes_Color | pk2.FrameSourceTypes_Depth | pk2.FrameSourceTypes_Infrared)

    print("initializing...")

    # assuming color resolution is always larger than depth
    print("height")
    color_depth_height_ratio = resolutionRatio(kinect, kinect.color_frame_desc.Height, kinect.depth_frame_desc.Height)
    print("width")
    color_depth_width_ratio = resolutionRatio(kinect, kinect.color_frame_desc.Width, kinect.depth_frame_desc.Width)
    
    if ((color_depth_height_ratio is not None) and (color_depth_width_ratio is not None)):
        while True:
            if kinect.has_new_color_frame():
                frame1D = kinect.get_last_color_frame()
                frameRGBA = frame1D.reshape((kinect.color_frame_desc.Height, kinect.color_frame_desc.Width, 4))
                frame = cv2.cvtColor(frameRGBA, cv2.COLOR_RGBA2RGB)
                cv2.imshow('colour', frame)
                print("colour") 
                print(frame.shape)
            
            if kinect.has_new_depth_frame():
                depth1D = kinect.get_last_depth_frame()
                depth = depth1D.reshape((kinect.depth_frame_desc.Height, kinect.depth_frame_desc.Width))
                depth = depth/8000
                depth = np.repeat(np.repeat(depth, color_depth_width_ratio, axis=1), color_depth_height_ratio, axis=0)
                cv2.imshow('depth', depth)
                print("depth")
                print(depth.shape)

            if kinect.has_new_infrared_frame():
                infrared1D = kinect.get_last_infrared_frame()
                infrared = infrared1D.reshape((kinect.infrared_frame_desc.Height, kinect.infrared_frame_desc.Width))
                infrared = infrared/65535
                cv2.imshow('infrared', infrared)

            key = cv2.waitKey(1)
            if key == 27: # exit on ESC
                break

        cv2.destroyAllWindows()

    