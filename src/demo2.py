#!/usr/bin/env python3

import rospy
from std_msgs.msg import Int32MultiArray, MultiArrayDimension 
import shutil
import argparse
import os, sys
from cv_bridge import CvBridge, CvBridgeError
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from sensor_msgs.msg import Image, CompressedImage
print(sys.path)
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import torchvision.transforms as transforms

from lib.config import cfg
from lib.config import update_config
from lib.utils.utils import create_logger, select_device
from lib.models import get_net
from lib.core.general import non_max_suppression, scale_coords
from lib.utils.plot2 import plot_one_box,show_seg_result
from lib.utils.augmentations import letterbox_for_img


class Yolo_P:
    def __init__(self,cfg,opt):
        self.cfg = cfg
        self.opt = opt
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.transform=transforms.Compose([
                transforms.ToTensor(),
                self.normalize,
            ])
    
        self.init_detector()
        self.bridge = CvBridge()     
        self.ros_topic_func()        
        
        
        
    def ros_topic_func(self):
        image_topic = rospy.get_param('~image_topic')
        self.img_sub = rospy.Subscriber(image_topic, CompressedImage, self.img_callback, queue_size=1)
        self.ll_pub = rospy.Publisher('/line_line', Int32MultiArray, queue_size = 1)
        self.da_pub = rospy.Publisher('/driv_eable_area', Int32MultiArray, queue_size = 1)    
        
        
        
    def img_callback(self, data):
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))    
        
     
        #print('data:', len(dataset))
        processed_image = self.detector(image)
        
        # Display the processed image (Optional)
        cv2.imshow('Processed Image', processed_image)
        cv2.waitKey(1)
        torch.cuda.empty_cache()    
        


    @torch.no_grad()
    def init_detector(self):
        logger, _, _ = create_logger(
            self.cfg, self.cfg.LOG_DIR, 'demo')
        self.device = select_device(logger,self.opt.device)
        if os.path.exists(self.opt.save_dir):  # output dir
            shutil.rmtree(self.opt.save_dir)  # delete dir
        os.makedirs(self.opt.save_dir)  # make new dir
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = get_net(self.cfg)
        checkpoint = torch.load(self.opt.weights, map_location= self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to(self.device)
        
        if self.half:
            self.model.half()  # to FP16
        
        img = torch.zeros((1, 3, self.opt.img_size, self.opt.img_size), device=self.device)  # init img
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once  
        self.model.eval()
                
        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        


    @torch.no_grad()
    def detector(self,data):
        path, img, img_det, vid_cap, shapes = LoadROSStream(data).__next__()
        img = self.transform(img).to(self.device)
        img = img.half() if self.half else img.float()
        
          # Convert to float tensor
        if len(img.shape) == 3:  # If it's C x H x W, add a batch dimension
            img = img.unsqueeze(0)  # B x C x H x W
        
        det_out, da_seg_out, ll_seg_out = self.model(img)
        inf_out, _ = det_out
        det_pred = non_max_suppression(inf_out, conf_thres=self.opt.conf_thres, iou_thres=self.opt.iou_thres, classes=None, agnostic=False)
        
        det = det_pred[0]
       
        _, _, height, width = img.shape
        h,w,_=img_det.shape
        #shapes = ((height, width),((height/h, width/w), (dw, dh)))
        pad_w, pad_h = shapes[1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        ratio = shapes[1][0][1]

        da_predict = da_seg_out[:, :, pad_h:(height-pad_h),pad_w:(width-pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
        # da_seg_mask = morphological_process(da_seg_mask, kernel_size=7)
        #print(da_seg_mask)
        
        ll_predict = ll_seg_out[:, :,pad_h:(height-pad_h),pad_w:(width-pad_w)]
        ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
        # Lane line post-processing
        #ll_seg_mask = morphological_process(ll_seg_mask, kernel_size=7, func_type=cv2.MORPH_OPEN)
        #ll_seg_mask = connect_lane(ll_seg_mask)

        img_det, da_coords, ll_coords = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)
        
        self.publish_multiarray(self.ll_pub, ll_coords)
        self.publish_multiarray(self.da_pub, da_coords)
        
        if len(det):
            det[:,:4] = scale_coords(img.shape[2:],det[:,:4],img_det.shape).round()
            for *xyxy,conf,cls in reversed(det):
                label_det_pred = f'{self.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img_det , label=label_det_pred, color=self.colors[int(cls)], line_thickness=2)
                xyxy_numbers = [coord.item() for coord in xyxy]
                #print(f"Bounding Box (Class: {self.names[int(cls)]}, Confidence: {conf:.2f}): {xyxy_numbers}")

        return img_det



    def publish_multiarray(self,publisher, numpy_array):
    
        # Convert the numpy array to a list
        array_list = numpy_array.ravel().tolist()
        # Create the message
        msg = Int32MultiArray()
        # Specify the layout for the data
        msg.layout.dim.append(MultiArrayDimension())
        msg.layout.dim[0].size = numpy_array.shape[0]
        msg.layout.dim[0].stride = numpy_array.size
        msg.layout.dim[0].label = "height"
        msg.layout.dim.append(MultiArrayDimension())
        msg.layout.dim[1].size = numpy_array.shape[1]
        msg.layout.dim[1].stride = numpy_array.shape[1]
        msg.layout.dim[1].label = "width"
        msg.layout.data_offset = 0
        # Assign the data
        msg.data = array_list
        # Publish the message
        publisher.publish(msg)



class LoadROSStream:  # Accepts ROS image directly
    def __init__(self, img0, img_size=640):
        self.mode = 'stream'
        self.img_size = img_size
        
        self.bridge = CvBridge()
        self.img0 = img0
        
    def __iter__(self):
        return self

    def __next__(self):
        if self.img0 is None:
            raise StopIteration

        h0, w0 = self.img0.shape[:2]
        img, _, pad = letterbox_for_img(self.img0, self.img_size, auto=True)
        h, w = img.shape[:2]
        self.shapes = (h0, w0), ((h / h0, w / w0), pad)
        
        # Convert
        # img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        self.img = np.ascontiguousarray(img)
        #print(self.shapes)
        return "ROS_Image", self.img, self.img0, None, self.shapes

    def __len__(self):
        return 1  # Only one source, the direct image






if __name__ == '__main__':
    try:
        rospy.init_node('yolo_p')
        args = rospy.myargv(argv=sys.argv) 
        parser = argparse.ArgumentParser()
        weights = str(rospy.get_param('~weights'))
        
        parser.add_argument('--weights', nargs='+', type=str, default= weights, help='model.pth path(s)')
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.7, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.7, help='IOU threshold for NMS')
        parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        opt = parser.parse_args(args[1:])
      
        yolo_p = Yolo_P(cfg,opt)
        rospy.spin()   
       
    except rospy.ROSInterruptException:
        pass

