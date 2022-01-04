import os
from datetime import datetime, date
import seaborn_image as isns
import matplotlib.pyplot as plt
import json
import pyrealsense2 as rs
import numpy as np
import cv2
from IPython.display import display

class Raptor:
    def __init__(self, width=640, height=480):
        try:
            with open('raptor_config.json', 'r') as f:
                self.config = json.loads(f.read())
        except:
            self.config = {'counter_name':0, 'counter_imagespfile':0}
        self.counter_name = self.config['counter_name']
        self.counter_imagespfile = self.config['counter_imagespfile']
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, 30)

        # Start streaming
        self.pipeline.start(config)

    def run(self, d_function=None, rgb_function=None, make_output_json=False, filepath='./data/', frame_catch=30):
        try:
            if make_output_json:
                today = date.today().strftime('%d-%m-%Y')
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                counter = 0
                
            while True:
                # Wait for a coherent pair of frames: depth and color
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                infrared_frame = frames.get_infrared_frame(1)
                if not depth_frame or not color_frame:
                    continue

                # Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                infrared_image = np.asanyarray(infrared_frame.get_data())
                

                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

                if d_function:
                    d_function(depth_image)
                    
                if rgb_function:
                    d_function(depth_image)
                
                depth_colormap_dim = depth_colormap.shape
                color_colormap_dim = color_image.shape
                infrared_colormap_dim = infrared_image.shape
                
                if make_output_json:
                    if counter % frame_catch == 0:
                        if self.counter_imagespfile > 1000:
                            self.counter_imagespfile = 0
                            self.counter_name += 1
                        with open(filepath+today+'_'+str(self.counter_name)+'_rgb.data', 'a') as f:
                            f.write(json.dumps({'image':color_image.tolist(), 'dim':color_colormap_dim, 'timestamp':datetime.now().strftime('%d-%m-%Y %H:%M:%S')})+', ')
                        with open(filepath+today+'_'+str(self.counter_name)+'_depth.data', 'a') as f:
                            f.write(json.dumps({'image':depth_image.tolist(), 'dim':depth_colormap_dim, 'timestamp':datetime.now().strftime('%d-%m-%Y %H:%M:%S')})+', ')
                        with open(filepath+today+'_'+str(self.counter_name)+'_infrared.data', 'a') as f:
                            f.write(json.dumps({'image':infrared_image.tolist(), 'dim':infrared_colormap_dim, 'timestamp':datetime.now().strftime('%d-%m-%Y %H:%M:%S')})+', ')
                        self.counter_imagespfile+=1    
                    counter += 1
                    
                

                # If depth and color resolutions are different, resize color image to match depth image for display
                if depth_colormap_dim != color_colormap_dim:
                    resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                    images = np.hstack((resized_color_image, depth_colormap))
                else:
                    images = np.hstack((color_image, depth_colormap))

                # Show images
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', images)
                k = cv2.waitKey(1)
                
                if k == 27:
                    print('Salvando configurações...')
                    with open('raptor_config.json', 'w') as f:
                        self.config['counter_name'] = self.counter_name
                        self.config['counter_imagespfile'] = self.counter_imagespfile
                        f.write(json.dumps(self.config))
                    print('Finalizando...')    
                    cv2.destroyAllWindows()
                    break  

        finally:
            # Stop streaming
            self.pipeline.stop()
            
class Analysis:
    
    def __init__(self, filename, path='./data/'):
        self.data = self._get_json(filename, path)
        """Global settings for images"""
        isns.set_context("notebook")
        isns.set_image(cmap="deep", despine=True)
        isns.set_scalebar(color="red")
        
    def show_image(self, i):
        """Image with a scalebar"""
        ax = isns.imgplot(np.flipud(np.array(self.data['data'][i]['image'])))

    def show_video(self):    
        fig = plt.figure(1)
        x = fig.add_subplot( 111 )
        ax.set_title("My Title")
        for f in self.data['data']:
            im = f['image']
            ax = isns.imgplot(np.flipud(im))
            plt.pause(1)
  
    def _get_json(self, file, path):
        with open(path+file, 'r') as f:
            return json.loads('{"data":['+f.read()[:-2]+']}')