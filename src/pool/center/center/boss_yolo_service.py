import rclpy
from rclpy.node import Node
from yolo_strategy_interfaces.srv import YoloStrategy

import math
import cv2
import numpy as np
import py_pubsub.darknet as darknet
import py_pubsub.pool_strategy as ps
import py_pubsub.pool_strat_v2 as ps2
import py_pubsub.verify_poolball_pose as vpp
import pyrealsense2 as rs

fix_abs_cam = [48.809, 310.996, 383.971, -179.499, -3.45, 89.95] # 固定相機的絕對位置
tool_to_cam = [-36.862, 128.23, -66.272] # 工具到相機的位置向量
CAM_TO_TABLE = 481 # 相機到桌面的距離

"""
影像檢測 image_detection
座標轉換 pixel_mm_convert
計算擊球路徑 375
# 根據識別出的球的位置來計算擊球路徑和角度
ValidRoute, bestrouteindex, obstacle_flag = ps.main(cue_x, cue_y, objectballs_x, objectballs_y, number_of_balls)
hitpoint_x, hitpoint_y = ps.find_hit_point(cue_x, cue_y, target_x, target_y)
yaw_angle = calculate_yaw_angle(vector_x, vector_y)

"""
# 神經網絡的配置文件和權重路徑
ALL_cfg_path ="/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/cfg/yolov4-obj.cfg"      #'./cfg/yolov4-obj.cfg'
ALL_weights_path = '/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/cfg/weights/ALL/yolov4-obj_best.weights'
ALL_data_path = '/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/cfg/hiwin_C_WDA_v4.data'

# 加載YOLO網絡模型
"""
載入神經網路
"""
ALL_network, ALL_class_names, ALL_class_colors = darknet.load_network(
        ALL_cfg_path,
        ALL_data_path,
        ALL_weights_path,
        batch_size=1
)

"""
影像檢測
    輸入:(影像位置,神經網路,物件名稱集,信心值閥值(0.0~1.0))
    輸出:(檢測後影像,檢測結果)
    註記:
"""
#YOLO網絡接受的格式，進行物體檢測，並將檢測到的物體畫在圖像上
def image_detection(image, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections



"""
座標轉換
    輸入:(YOLO座標,原圖寬度,原圖高度)
    輸出:(框的左上座標,框的右下座標)
    註記:
"""
#將YOLO的輸出座標（中心點和寬高比）轉換為對應圖像上的矩形框的四個角點座標。
def bbox2points(bbox,W,H):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """ 
    width = darknet.network_width(ALL_network)      # YOLO壓縮圖片大小(寬)
    height = darknet.network_height(ALL_network)    # YOLO壓縮圖片大小(高)

    x, y, w, h = bbox                           # (座標中心x,座標中心y,寬度比值,高度比值)
    x = x*W/width
    y = y*H/height
    w = w*W/width
    h = h*H/height
    # 輸出框座標_YOLO格式
    # print("     (left_x: {:.0f}   top_y:  {:.0f}   width:   {:.0f}   height:  {:.0f})".format(x, y, w, h))
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    
    return xmin, ymin, xmax, ymax



"""
原圖繪製檢測框線
    輸入:(檢測結果,原圖位置,框線顏色集)
    輸出:(影像結果)
    註記:
"""
#根據檢測結果在原始圖像上畫出框線，並為每個檢測到的物體添加標籤和信心指數。
def draw_boxes(detections, image, colors):
    ball_imformation = [[-999 for i in range(4)] for j in range(20)]
    i = 0

    H,W,_ = image.shape                      # 獲得原圖長寬

    # cv2.line(image,(640,0),(640,720),(0,0,255),5)

    for label, confidence, bbox in detections:
        xmin, ymin, xmax, ymax = bbox2points(bbox,W,H)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), colors[label], 1)
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                    (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[label], 2)
        # 輸出框座標_加工格式座標(左上點座標,右上點座標)
        #print("\t{}\t: {:3.2f}%    (x1: {:4.0f}   y1: {:4.0f}   x2: {:4.0f}   y2: {:4.0f})".format(label, float(confidence), xmin, ymin, xmax, ymax))
        
        mx = float(xmax + xmin)/2
        my = float(ymax + ymin)/2

        # cv2.circle(image, (int(mx),int(my)), 33, (0,0,255), 3)
        if label == 'C':
            ball_imformation[i] = [0.0, float(confidence), mx, my]
        elif label == 'M':
            ball_imformation[i] = [1.0, float(confidence), mx, my]
        i+=1
        
    return image, ball_imformation
#整合前面的 image_detection 和 draw_boxes 函數來執行完整的檢測流程，包括影像讀取、物體檢測和畫框。
def detect_ALL(img,thresh=0.8):
    out,detections = image_detection(img,ALL_network, ALL_class_names, ALL_class_colors,thresh)
    out2, ball_imformation= draw_boxes(detections, img, ALL_class_colors)
    # cv2.imshow('out2', out2)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return out2, ball_imformation
#根據向量計算角度，返回角度（以度為單位）和弧度，主要用於計算球桿相對於球的角度。
def yaw_angle(vectorx, vectory):
    vectorlength = math.sqrt((vectorx**2)+(vectory**2))
    rad = math.acos((-1*vectory)/(vectorlength*1))
    theta = rad*180/math.pi
    if vectorx >= 0:
        return theta, rad
    elif vectorx < 0:
        return -theta, -rad
 #像素距離轉換為實際的毫米距離，基於實際的物理測量與相機規格。   

#座標轉換
def pixel_mm_convert(pixel):
    actuallengh = 626
    pixellengh = 1920
    mm = actuallengh/pixellengh*pixel
    return mm
#基於RealSense相機的內參校正，使用校正後的內參來進行像素點到三維空間點的轉換。
def realsense_intrinsics(x, y):
    width = 1920
    height = 1080
    fps = 30
    depth = 1
    
    # calibrated_intrinsics_f = [1362.38, 1360.45]
    # calibrated_intrinsics_pp = [938.315, 552.935]

    calibrated_intrinsics_f = [1374.6184196461807, 1371.8167857594508]
    calibrated_intrinsics_pp = [962.554839837715, 555.7250519253395]

    # dis_coeffs = [0.0693826933, 0.445315521, 0.00291064076, -0.000845071017, -1.99098719]
    dis_coeffs = [0.1365981859476498, -0.34297522753848286, 0.0012525886922101891, -0.00011109914481045795, 0.15068607105066928]

    _intrinsics = rs.intrinsics()
    _intrinsics.width = width
    _intrinsics.height = height
    _intrinsics.ppx = calibrated_intrinsics_pp[0]
    _intrinsics.ppy = calibrated_intrinsics_pp[1]
    _intrinsics.fx = calibrated_intrinsics_f[0]
    _intrinsics.fy = calibrated_intrinsics_f[1]
    #_intrinsics.model = cameraInfo.distortion_model
    _intrinsics.model  = rs.distortion.none
    _intrinsics.coeffs = dis_coeffs

    # pipeline = rs.pipeline()
    # config = rs.config()
    # config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    # config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    # ストリーミング開始

    # profile = pipeline.start(config)
    # depth_intrinsics = rs.video_stream_profile(profile.get_stream(rs.stream.depth)).get_intrinsics()
    # color_intrinsics = rs.video_stream_profile(profile.get_stream(rs.stream.color)).get_intrinsics()

    pixel = [x, y]
    ca_point = rs.rs2_deproject_pixel_to_point(_intrinsics, pixel, depth)
    # print('calibrated point:',ca_point)

    x_ = int(ca_point[0] * calibrated_intrinsics_f[0] + calibrated_intrinsics_pp[0])
    y_ = int(ca_point[1] * calibrated_intrinsics_f[1] + calibrated_intrinsics_pp[1])
    # print('calibrated intrinsics x:',x_)
    # print('calibrated intrinsics y:',y_)

    # pipeline.stop()
    return float(x_), float(y_)

#相機角度校正函數
#這兩個函數用於校正影像中目標物體的位置，考慮相機與目標物之間的角度變化。
#第一個函數不考慮高度，第二個函數考慮了相機的高度變化。
def cam_angle_correction(camseex, camseey):
    width = 1920
    height = 1080
    if camseex <= width/2:
        midx = abs(width/2-camseex)
        if camseey <= height/2:
            midy = abs(height/2-camseey)
            devx = 16/455.417*midx 
            devy = 16/455.417*midy
            actx = camseex+devx
            acty = camseey+devy
        else:
            midy = abs(camseey-height/2)
            devx = 16/455.417*midx 
            devy = 16/455.417*midy
            actx = camseex+devx
            acty = camseey-devy
    
    else:
        midx = abs(camseex-width/2)
        if camseey <= height/2:
            midy = abs(height/2-camseey)
            devx = 16/455.417*midx 
            devy = 16/455.417*midy
            actx = camseex-devx
            acty = camseey+devy
        else:
            midy = abs(camseey-height/2)
            devx = 16/455.417*midx 
            devy = 16/455.417*midy
            actx = camseex-devx
            acty = camseey-devy

    return  actx, acty

def cam_angle_correction_2(camseex, camseey, camheight):
    width = 1920
    height = 1080
    if camseex <= width/2:
        midx = abs(width/2-camseex)
        if camseey <= height/2:
            midy = abs(height/2-camseey)
            devx = 16/camheight*midx 
            devy = 16/camheight*midy
            actx = camseex+devx
            acty = camseey+devy
        else:
            midy = abs(camseey-height/2)
            devx = 16/camheight*midx 
            devy = 16/camheight*midy
            actx = camseex+devx
            acty = camseey-devy
    
    else:
        midx = abs(camseex-width/2)
        if camseey <= height/2:
            midy = abs(height/2-camseey)
            devx = 16/camheight*midx 
            devy = 16/camheight*midy
            actx = camseex-devx
            acty = camseey+devy
        else:
            midy = abs(camseey-height/2)
            devx = 16/camheight*midx 
            devy = 16/camheight*midy
            actx = camseex-devx
            acty = camseey-devy

    return  actx, acty
#處理影像並進行決策。該服務基於影像輸入計算球的位置和擊球路線。
class YOLOService(Node):
    def __init__(self):
        super().__init__('yolo_service')
        # 在ROS 2中註冊一個服務，名為'yolo_strategy'，使用的是YoloStrategy服務類型
        self.yolo_service = self.create_service(YoloStrategy, 'yolo_strategy', self.service_callback)
        self.n = 2201 # 一個計數器或其他用途的變數（未在後續代碼中使用）
        
        # 初始化用於存儲識別結果的多個列表
        self.objectballx = []  # 物體x座標
        self.objectbally = []  # 物體y座標
        self.confidence = []  # 識別的置信度
        self.intrin_objx = []  # 校正後的x座標
        self.intrin_objy = []  # 校正後的y座標
        self.corrected_objx = []  # 更進一步校正後的x座標
        self.corrected_objy = []  # 更進一步校正後的y座標
        self.lucky_flag = 0  # 標誌位，用於標記某些特定狀態
    #這個回調函數是服務的核心，它根據客戶端請求的 send_position 參數執行不同的影像處理和數據提取任務。
    def service_callback(self, request, response):
        # detect image
        # clear array
        flat_list = [] # 初始化一個用於存儲扁平化數據的列表

        # 根據請求的類型進行不同的處理
        # request 1 for hitpoint position, yaw angle and route score
        # 處理第一類請求：從指定的影像文件中檢測並分析球的位置，計算擊球點、角度等
        if request.send_position == 1:
            flat_list = []
            self.objectballx = []
            self.objectbally = []
            self.confidence = []
            self.intrin_objx = []
            self.intrin_objy = []
            self.corrected_objx = []
            self.corrected_objy = []
            self.get_logger().info('culculating hitpoint yaw and pitch angle \n')
            img = cv2.imread('/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/testpics/detect_ball.jpg')

            out2, ballinfo = detect_ALL(img) # 假設detect_ALL是一個定義在外部的函數，用於執行物體識別
            # check for non float value in ballinfo since flat_list does not accept data other than float
            cnt = 0
            for i in range(len(ballinfo)):
                if ballinfo[i][1] != -999:
                    cnt += 1
                else:
                    break
            #flatten list of 2darray to 1darray
            flat_list = []
            for i in range(0,cnt):
                flat_list.extend(ballinfo[i])

            # response.current_position = flat_list
            # self.get_logger().info('current ball location sent !\n')
            # self.n += 1

            # convert flat array to usable array 
            # in this case objectballx(y)[], with cuex(y) in objectballx(y)[-1] and confidence[]
            # 處理獲得的球信息，存儲到各種列表中
            for i in range(0,len(flat_list),4):
                if flat_list[i] == 0:
                    self.confidence.append(flat_list[i+1])
                    self.objectballx.append(flat_list[i+2])
                    self.objectbally.append(flat_list[i+3]-83)
                    #################3
                else:
                    cueindex = i
            self.confidence.append(flat_list[cueindex+1])
            self.objectballx.append(flat_list[cueindex+2])
            self.objectbally.append(flat_list[cueindex+3]-83)
            #####################

            # print("input objectball x:\n",self.objectballx)
            # print("input objectball y:\n",self.objectbally)
            n = len(self.objectballx)
            camheight = request.update_position[0]
            print(camheight)
            for i in range(0,n):
                # realsense intrinsics calibration
                intrinx, intriny = realsense_intrinsics(self.objectballx[i], self.objectbally[i])
                self.intrin_objx.append(intrinx)
                self.intrin_objy.append(intriny)
                # cam angle correction
                realx, realy = cam_angle_correction_2(intrinx, intriny, camheight)
                self.corrected_objx.append(realx)
                self.corrected_objy.append(realy)
            print('intrinsics objectball x:\n', self.intrin_objx)
            print('intrinsics objectball y:\n', self.intrin_objy)
            print('real objectball x:\n',self.corrected_objx)
            print('real objectball y:\n',self.corrected_objy)

            # 計算最佳路徑、擊球點、角度等，並設置回應消息
            ValidRoute, bestrouteindex, obstacle_flag = ps.main(self.corrected_objx[-1],self.corrected_objy[-1], 
                                                self.corrected_objx[0:n-1], self.corrected_objy[0:n-1],n-1)
            ########################
            # ps.main need to change upper and lower bound 
            

            print('All valid route:\n',ValidRoute)
            print('Best route index:',bestrouteindex)
            print('Best Route:\n',ValidRoute[bestrouteindex])
            print('Score of best route:\n',ValidRoute[bestrouteindex][0])

            hitpointx, hitpointy = ps.findhitpoint(self.corrected_objx[-1],self.corrected_objy[-1],
                                                ValidRoute[bestrouteindex][3][0],ValidRoute[bestrouteindex][3][1])
            yaw, rad = yaw_angle(ValidRoute[bestrouteindex][3][0],ValidRoute[bestrouteindex][3][1])
            
            # check if any obstacle behind hitpoint,  if so ajust pitch angle
            # obstacle_flag = ps.check_obstacle(self.corrected_objx[-1],self.corrected_objy[-1],
            #                                     ValidRoute[bestrouteindex][3][0],ValidRoute[bestrouteindex][3][1],
            #                                     self.corrected_objx, self.corrected_objy)
            
            # print('Any obstacle:', obstacle_flag)

            hitpointmmx = pixel_mm_convert(hitpointx)
            hitpointmmy = pixel_mm_convert(hitpointy)
            score = ValidRoute[bestrouteindex][0]
            print('degree:',yaw)
            response.current_position = [hitpointmmx, hitpointmmy, yaw, score]
        
        # 處理第二類請求：從第二個影像中檢測球的位置
        elif request.send_position == 2: # detect second photo 
            flat_list = []
            self.objectballx = []
            self.objectbally = []
            self.confidence = []
            self.intrin_objx = []
            self.intrin_objy = []
            self.corrected_objx = []
            self.corrected_objy = []
            
            self.get_logger().info('detecting second photo ! \n')
            img2 = cv2.imread('/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/testpics/detect_second_ball.jpg')
            out2, ballinfo = detect_ALL(img2)
            cv2.imwrite('/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/testpics/detected_second_ball.jpg',out2)
            # check for non float value in ballinfo since flat_list does not accept data other than float
            cnt = 0
            for i in range(len(ballinfo)):
                if ballinfo[i][1] != -999:
                    cnt += 1
                else:
                    break
            #flatten list of 2darray to 1darray
            flat_list = []
            for i in range(0,cnt):
                flat_list.extend(ballinfo[i])

            # convert flat array to usable array 
            # usable array ==> in this case objectballx(y)[], with cuex(y) in objectballx(y)[-1] and confidence[]
            flag_cue = 0
            self.get_logger().info('culculating best route\n')
            for i in range(0,len(flat_list),4):
                if flat_list[i] == 0:
                    # self.confidence.append(flat_list[i+1])
                    self.objectballx.append(flat_list[i+2])
                    self.objectbally.append(flat_list[i+3])
                    #################################
                    print(i)
                elif flat_list[i] == 1:
                    cueindex = i
                    flag_cue = 1
            if flag_cue == 1:
                # self.confidence.append(flat_list[cueindex+1])
                self.objectballx.append(flat_list[cueindex+2])
                self.objectbally.append(flat_list[cueindex+3])
                ############################
            else:
                pass

            # print("input objectball x:\n",self.objectballx)
            # print("input objectball y:\n",self.objectbally)
            n = len(self.objectballx)

            # DO CAMERA CALIBRATION HERE
            for i in range(0,n):
                # realsense intrinsics calibration
                intrinx, intriny = realsense_intrinsics(self.objectballx[i], self.objectbally[i])
                self.intrin_objx.append(intrinx)
                self.intrin_objy.append(intriny)
            
            print('intrinsics objectball x:\n', self.intrin_objx)
            print('intrinsics objectball y:\n', self.intrin_objy)

            # return whole list, first half of list for x-axis and second half for y-axis
            wholelist = self.intrin_objx+self.intrin_objy
            print('1D ball position:\n', wholelist)
            response.current_position = wholelist
        # 處理第三類請求：返回所有球的位置
        elif request.send_position == 3: # send all ball in route
            flat_list = []
            self.objectballx = []
            self.objectbally = []
            self.confidence = []
            self.intrin_objx = []
            self.intrin_objy = []
            self.corrected_objx = []
            self.corrected_objy = []
            self.get_logger().info('request for current ball location\n')
            img = cv2.imread('/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/testpics/detect_ball.jpg')

            out2, ballinfo = detect_ALL(img)
            cv2.imwrite('/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/testpics/detected_ball.jpg',out2)
            print('ball info:', ballinfo)
            # check for non float value in ballinfo since flat_list does not accept data other than float
            cnt = 0
            for i in range(len(ballinfo)):
                if ballinfo[i][1] != -999:
                    cnt += 1
                else:
                    break
            #flatten list of 2darray to 1darray
            flat_list = []
            for i in range(0,cnt):
                flat_list.extend(ballinfo[i])

            # response.current_position = flat_list
            # self.get_logger().info('current ball location sent !\n')
            # self.n += 1

            # convert flat array to usable array 
            # in this case objectballx(y)[], with cuex(y) in objectballx(y)[-1] and confidence[]
            flag_cue = 0
            self.get_logger().info('culculating best route\n')
            for i in range(0,len(flat_list),4):
                if flat_list[i] == 0:
                    self.confidence.append(flat_list[i+1])
                    self.objectballx.append(flat_list[i+2])
                    self.objectbally.append(flat_list[i+3]-83)
                elif flat_list[i] == 1:
                    cueindex = i
                    flag_cue = 1
            if flag_cue == 1:
                self.confidence.append(flat_list[cueindex+1])
                self.objectballx.append(flat_list[cueindex+2])
                self.objectbally.append(flat_list[cueindex+3]-83)

            print("input obValidRoutejectballx")

            for i in range(0,n):
                # realsense intrinsics calibration
                intrinx, intriny = realsense_intrinsics(self.objectballx[i], self.objectbally[i])
                self.intrin_objx.append(intrinx)
                self.intrin_objy.append(intriny)
                # cam angle correction
                realx, realy = cam_angle_correction(intrinx, intriny)
                self.corrected_objx.append(realx)
                self.corrected_objy.append(realy)
            print('intrinsics objectball x:\n', self.intrin_objx)
            print('intrinsics objectball y:\n', self.intrin_objy)
            print('real objectball x:\n',self.corrected_objx)
            print('real objectball y:\n',self.corrected_objy)
            self.intrin_objx[0:n-1], self.intrin_objy[0:n-1],n-1
            
            score = ValidRoute[bestrouteindex][0]
            if score == -6000.0:
                self.lucky_flag = 1
            else:
                self.lucky_flag = 0
            #lcuky ball
            if NBallInRoute == 2:
                temp_flat = []
                temp_flat.append(float(ValidRoute[bestrouteindex][2][0]))
                temp_flat.append(float(ValidRoute[bestrouteindex][2][1]))
                temp_flat.append(float(ValidRoute[bestrouteindex][4][0]))
                temp_flat.append(float(ValidRoutYOLOService_2e[bestrouteindex][4][1]))
                temp_flat_mm = []
                for i in range(0,len(temp_flat)):
                    temp = pixel_mm_convert(tempYOLOService_2_flat[i])
                    temp_flat_mm.append(temp)
                response.current_position = temp_flat_mm
                response.obstacle_flag = ob_flagYOLOService_2
                temp_flat.append(float(ValidRoute[bestrouteindex][2][0]))
                temp_flat.append(float(ValidRoute[bestrouteindex][2][1]))
                temp_flat.append(float(ValidRoute[bestrouteindex][4][0]))
                temp_flat.append(float(ValidRoute[bestrouteindex][4][1]))
                temp_flat_mm = []
                response.current_position = temp_flat_mm
                response.obstacle_flag = ob_flag
            elif NBallInRoute == 3:
                temp_flat = []
                temp_flat.append(float(ValidRouYOLOService_2te[bestrouteindex][2][0]))
                temp_flat.append(float(ValidRoute[bestrouteindex][2][1]))
                temp_flat.append(float(ValidRouYOLOService_2te[bestrouteindex][4][0]))
                temp_flat.append(float(ValidRoute[bestrouteindex][4][1]))
                temp_flat.append(float(ValidRoute[bestrouteindex][6][0]))
                temp_flat.append(float(ValidRoute[bestrouteindex][6][1]))
                temp_flat_mm = []
                for i in range(0,len(temp_flat)):
                    temp = pixel_mm_convert(temp_flat[i])
                    temp_flat_mm.append(temp)
                response.current_position = temYOLOService_2g
            else:
                temp_flat = []
                temp_flat.append(float(ValidRoute[bestrouteinYOLOService_2dex][2][0]))
                temp_flat.append(float(ValidRoute[bestrouteindex][2][1]))
                temp_flat.append(float(ValidRoute[bestrouteindex][4][0]))
                temp_flat.append(float(ValidRoute[bestrouteindex][4][1]))
                temp_flat.append(float(ValidRoute[bestrouteindex][6][0]))
                temp_flat.append(float(ValidRoute[bestrouteindex][6][1]))
                temp_flat.append(float(ValidRoute[bestrouteindex][8][0]))
                temp_flat.append(float(ValidRoute[bestrouteindex][8][1]))
                # first index is cueball and the rest follow(s)
                temp_flat_mm = []
                for i in range(0,len(temp_flat)):
                    temp = pixel_mm_convert(temp_flat[i])
                    temp_flat_mm.append(temp)
                response.current_position = temp_flat_mm
                response.obstacle_flag = ob_flag

        # 處理第四類請求：根據更新的位置信息重新計算路徑
        elif request.send_position == 4:
            re_position = request.update_position
            cuex = re_position[0]
            cuey = re_position[1]
            objx = []
            objy = []
            l = len(re_position)
            for i in range(2,len(re_position),2):
                objx.append(re_position[i])
                objy.append(re_position[i+1])
            m = int(l/2)-1
            # objx.append(cuex)
            # objy.append(cuey)
            
            
            # ValidRoute, bestrouteindex, ob_flag = ps.main(self.corrected_objx[-1],self.corrected_objy[-1], 
            #                                     self.corrected_objx[0:n-1], self.corrected_objy[0:n-1],n-1)

            # score = ValidRoute[bestrouteindex][0]
            # if score == -6000.0:
            #     self.lucky_flag = 1
            # else:
            #     self.lucky_flag = 0
            print('-----------------------------------------------')
            print('lucky flag value:',self.lucky_flag)
            print('cue x:',cuex)
            print('cue y:',cuey)
            print('obj x:',objx)
            print('obj y:',objy)
            print('-----------------------------------------------')
            if self.lucky_flag == 0:
                re_Route, re_bestindex, _ = ps.main(cuex, cuey, objx, objy, m) 
                new_hitpointx, new_hitpointy = ps.findhitpoint(cuex, cuey,
                                                    re_Route[re_bestindex][3][0],re_Route[re_bestindex][3][1])
                new_yaw, rad = yaw_angle(re_Route[re_bestindex][3][0],re_Route[re_bestindex][3][1])
                new_hitpointmmx = pixel_mm_convert(new_hitpointx)
                new_hitpointmmy = pixel_mm_convert(new_hitpointy)
                score = re_Route[re_bestindex][0]

                response.current_position = [new_hitpointmmx, new_hitpointmmy, new_yaw, score]
                # response.current_position = [new_hitpointx, new_hitpointy, new_yaw, score]
            else:
                # # this is just for pausing, not lucky route
                # re_Route, re_bestindex, _ = ps.main(cuex, cuey, objx, objy, m) 
                new_hitpointx, new_hitpointy = ps.hitpoint(cuex, cuey, objx[0]-cuex, objy[0]-cuey)
                new_yaw, rad = yaw_angle(objx[0]-cuex, objy[0]-cuey)
                new_hitpointmmx = pixel_mm_convert(new_hitpointx)
                new_hitpointmmy = pixel_mm_convert(new_hitpointy)
                score = -6000.0
                response.current_position = [new_hitpointmmx, new_hitpointmmy, new_yaw, score]
        # 如果請求類型不符合預期，記錄一
        else:
            self.get_logger().info('waiting for proper request...')
            response.current_position = [-3.0]
            
        return response
    
class YOLOService_2(Node):
    def __init__(self):
        super().__init__('yolo_service_2')
        self.yolo_service = self.create_service(YoloStrategy, 'yolo_strategy', self.service_callback)
        self.n = 2201
        self.cameraMatrix = np.array([
                            [1363.8719422654856,  0, 938.9732418889218],
                            [ 0, 1362.7328132779226, 548.3344055737161],
                            [ 0,  0,  1]
                            ])
        self.distCoeffs = np.array([0.16291880696523953, -0.4619911499670495, 
                           0.00023885421077117, -0.0005976317960594, 
                           0.3376508830773949])
        
        self.objectballx = []
        self.objectbally = []
        self.confidence = []
        self.intrin_objx = []
        self.intrin_objy = []
        self.corrected_objx = []
        self.corrected_objy = []
        self.lucky_flag = 0

    def service_callback(self, request, response):
        # detect image
        # clear array
        flat_list = []

        # request 1 for hitpoint position, yaw angle and route score
        if request.send_position == 1:
            flat_list = []
            self.objectballx = []
            self.objectbally = []
            self.confidence = []
            self.intrin_objx = []
            self.intrin_objy = []
            self.corrected_objx = []
            self.corrected_objy = []
            self.get_logger().info('culculating hitpoint yaw and pitch angle \n')
            img = cv2.imread('/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/testpics/verification_use.jpg')
            undistorted_img = cv2.undistort(img, self.cameraMatrix, self.distCoeffs, None)
            out2, ballinfo = detect_ALL(undistorted_img)
            # check for non float value in ballinfo since flat_list does not accept data other than float
            cnt = 0
            for i in range(len(ballinfo)):
                if ballinfo[i][1] != -999:
                    cnt += 1
                else:
                    break
            #flatten list of 2darray to 1darray
            flat_list = []
            for i in range(0,cnt):
                flat_list.extend(ballinfo[i])

            # response.current_position = flat_list
            # self.get_logger().info('current ball location sent !\n')
            # self.n += 1

            # convert flat array to usable array 
            # in this case objectballx(y)[], with cuex(y) in objectballx(y)[-1] and confidence[]
            for i in range(0,len(flat_list),4):
                if flat_list[i] == 0:
                    self.confidence.append(flat_list[i+1])
                    self.objectballx.append(flat_list[i+2])
                    self.objectbally.append(flat_list[i+3]-83)
                    #################3
                else:
                    cueindex = i
            self.confidence.append(flat_list[cueindex+1])
            self.objectballx.append(flat_list[cueindex+2])
            self.objectbally.append(flat_list[cueindex+3]-83)
            #####################
            n = len(self.objectballx)
            for i in range(n):
                ball_pose_mm = vpp.pixel_mm_convert(CAM_TO_TABLE, [self.objectballx[i], self.objectbally[i]])
                ball_pose = vpp.convert_arm_pose(ball_pose_mm, fix_abs_cam)
                self.corrected_objx.append(ball_pose[0])
                self.corrected_objy.append(ball_pose[1])


            ValidRoute, bestrouteindex, obstacle_flag = ps2.main(self.corrected_objx[-1],self.corrected_objy[-1], 
                                                self.corrected_objx[0:n-1], self.corrected_objy[0:n-1],n-1)
            ########################
            # ps.main need to change upper and lower bound 
            

            print('All valid route:\n',ValidRoute)
            print('Best route index:',bestrouteindex)
            print('Best Route:\n',ValidRoute[bestrouteindex])
            print('Score of best route:\n',ValidRoute[bestrouteindex][0])

            hitpointx, hitpointy = ps2.findhitpoint(self.corrected_objx[-1],self.corrected_objy[-1],
                                                ValidRoute[bestrouteindex][3][0],ValidRoute[bestrouteindex][3][1])
            yaw, rad = yaw_angle(ValidRoute[bestrouteindex][3][0],ValidRoute[bestrouteindex][3][1])
            
            # check if any obstacle behind hitpoint,  if so ajust pitch angle
            # obstacle_flag = ps.check_obstacle(self.corrected_objx[-1],self.corrected_objy[-1],
            #                                     ValidRoute[bestrouteindex][3][0],ValidRoute[bestrouteindex][3][1],
            #                                     self.corrected_objx, self.corrected_objy)
            
            # print('Any obstacle:', obstacle_flag)
            score = ValidRoute[bestrouteindex][0]
            print('degree:',yaw)
            response.current_position = [hitpointx, hitpointy, yaw, score]
            
        elif request.send_position == 2: # detect second photo 
            flat_list = []
            self.objectballx = []
            self.objectbally = []
            self.confidence = []
            self.intrin_objx = []
            self.intrin_objy = []
            self.corrected_objx = []
            self.corrected_objy = []
            self.fix_z = 80.0
            self.table_z = fix_abs_cam[2] + tool_to_cam[2] - CAM_TO_TABLE
            
            self.get_logger().info('detecting second photo ! \n')
            img = cv2.imread('/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/testpics/verify_second_photo.jpg')
            undistorted_img = cv2.undistort(img, self.cameraMatrix, self.distCoeffs, None)
            out2, ballinfo = detect_ALL(undistorted_img)
            # cv2.imwrite('/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/testpics/detected_second_ball.jpg',out2)
            # check for non float value in ballinfo since flat_list does not accept data other than float
            cnt = 0
            for i in range(len(ballinfo)):
                if ballinfo[i][1] != -999:
                    cnt += 1
                else:
                    break
            #flatten list of 2darray to 1darray
            flat_list = []
            for i in range(0,cnt):
                flat_list.extend(ballinfo[i])

            # convert flat array to usable array 
            # usable array ==> in this case objectballx(y)[], with cuex(y) in objectballx(y)[-1] and confidence[]
            flag_cue = 0
            self.get_logger().info('culculating best route\n')
            for i in range(0,len(flat_list),4):
                if flat_list[i] == 0:
                    # self.confidence.append(flat_list[i+1])
                    self.objectballx.append(flat_list[i+2])
                    self.objectbally.append(flat_list[i+3])
                    #################################
                    print(i)
                elif flat_list[i] == 1:
                    cueindex = i
                    flag_cue = 1
            if flag_cue == 1:
                # self.confidence.append(flat_list[cueindex+1])
                self.objectballx.append(flat_list[cueindex+2])
                self.objectbally.append(flat_list[cueindex+3])
                ############################
            else:
                pass
            # return whole list, first half of list for x-axis and second half for y-axis
            mid_error = []
            for i in range(len(self.objectballx)):
                dev_x = self.objectballx[i] - 1920/2
                dev_y = self.objectbally[i] - 1080/2
                temp_error = math.sqrt((dev_x)**2+(dev_y)**2)
                mid_error.append(temp_error)
                min_error_index = mid_error.index(min(mid_error))
                midball_x = self.objectballx[min_error_index]
                midball_y = self.objectbally[min_error_index]
            ball_relative_cam_mm = vpp.pixel_mm_convert(self.fix_z - abs(tool_to_cam[2]) + abs(self.table_z), 
                                                 [midball_x, midball_y])

            # wholelist = self.objectballx+self.objectbally
            # print('1D ball position:\n', wholelist)
            response.current_position = ball_relative_cam_mm

        elif request.send_position == 3: # send all ball in route
            flat_list = []
            self.objectballx = []
            self.objectbally = []
            self.confidence = []
            self.intrin_objx = []
            self.intrin_objy = []
            self.corrected_objx = []
            self.corrected_objy = []
            self.get_logger().info('request for current ball location\n')
            img = cv2.imread('/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/testpics/detect_ball.jpg')
            undistorted_img = cv2.undistort(img, self.cameraMatrix, self.distCoeffs, None)
            out2, ballinfo = detect_ALL(undistorted_img)
            # cv2.imwrite('/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/testpics/detected_ball.jpg',out2)
            print('ball info:', ballinfo)
            # check for non float value in ballinfo since flat_list does not accept data other than float
            cnt = 0
            for i in range(len(ballinfo)):
                if ballinfo[i][1] != -999:
                    cnt += 1
                else:
                    break
            #flatten list of 2darray to 1darray
            flat_list = []
            for i in range(0,cnt):
                flat_list.extend(ballinfo[i])

            # response.current_position = flat_list
            # self.get_logger().info('current ball location sent !\n')
            # self.n += 1

            # convert flat array to usable array 
            # in this case objectballx(y)[], with cuex(y) in objectballx(y)[-1] and confidence[]
            flag_cue = 0
            self.get_logger().info('culculating best route\n')
            for i in range(0,len(flat_list),4):
                if flat_list[i] == 0:
                    self.confidence.append(flat_list[i+1])
                    self.objectballx.append(flat_list[i+2])
                    self.objectbally.append(flat_list[i+3])
                elif flat_list[i] == 1:
                    cueindex = i
                    flag_cue = 1
            if flag_cue == 1:
                self.confidence.append(flat_list[cueindex+1])
                self.objectballx.append(flat_list[cueindex+2])
                self.objectbally.append(flat_list[cueindex+3])

            print("input objectball x:\n",self.objectballx)
            print("input objectball y:\n",self.objectbally)

            n = len(self.objectballx)
            for i in range(n):
                relative_cam = vpp.pixel_mm_convert(CAM_TO_TABLE, [self.objectballx[i], self.objectbally[i]])
                relative_arm = vpp.convert_arm_pose(relative_cam, fix_abs_cam)
                self.intrin_objx.append(relative_arm[0])
                self.intrin_objy.append(relative_arm[1])
            ValidRoute, bestrouteindex, ob_flag = ps2.main(self.intrin_objx[-1],self.intrin_objy[-1], 
                                                self.intrin_objx[0:n-1], self.intrin_objy[0:n-1],n-1)
            
            score = ValidRoute[bestrouteindex][0]
            if score == -6000.0:
                self.lucky_flag = 1
            else:
                self.lucky_flag = 0
            
            # route() return this
            # score,cuefinalvector,cue,cuetoivector, objectballi, itok2vector, objectballk2 ,k2tok1vector, objectballk1, toholevector,n
            self.get_logger().info('sending best route ! \n')
            NBallInRoute = ValidRoute[bestrouteindex][-1] + 2 
            if ValidRoute[bestrouteindex][0] == -6000: #lcuky ball
                temp_flat = []
                temp_flat.append(float(ValidRoute[bestrouteindex][2][0])) # cuex
                temp_flat.append(float(ValidRoute[bestrouteindex][2][1])) # cuey
                temp_flat.append(float(ValidRoute[bestrouteindex][4][0])) # objectballix
                temp_flat.append(float(ValidRoute[bestrouteindex][4][1])) # objectballiy

                response.current_position = temp_flat
                response.obstacle_flag = ob_flag
            elif NBallInRoute == 2: # cue and objectballi
                temp_flat = []
                temp_flat.append(float(ValidRoute[bestrouteindex][2][0])) 
                temp_flat.append(float(ValidRoute[bestrouteindex][2][1]))
                temp_flat.append(float(ValidRoute[bestrouteindex][4][0]))
                temp_flat.append(float(ValidRoute[bestrouteindex][4][1]))
            
                response.current_position = temp_flat
                response.obstacle_flag = ob_flag
            elif NBallInRoute == 3: # cue, objectballi and objectballk2
                temp_flat = []
                temp_flat.append(float(ValidRoute[bestrouteindex][2][0]))
                temp_flat.append(float(ValidRoute[bestrouteindex][2][1]))
                temp_flat.append(float(ValidRoute[bestrouteindex][4][0]))
                temp_flat.append(float(ValidRoute[bestrouteindex][4][1]))
                temp_flat.append(float(ValidRoute[bestrouteindex][6][0]))
                temp_flat.append(float(ValidRoute[bestrouteindex][6][1]))
            
                response.current_position = temp_flat
                response.obstacle_flag = ob_flag
            else: # cue, objectballi, objectballk2 and objectballk1
                temp_flat = []
                temp_flat.append(float(ValidRoute[bestrouteindex][2][0]))
                temp_flat.append(float(ValidRoute[bestrouteindex][2][1]))
                temp_flat.append(float(ValidRoute[bestrouteindex][4][0]))
                temp_flat.append(float(ValidRoute[bestrouteindex][4][1]))
                temp_flat.append(float(ValidRoute[bestrouteindex][6][0]))
                temp_flat.append(float(ValidRoute[bestrouteindex][6][1]))
                temp_flat.append(float(ValidRoute[bestrouteindex][8][0]))
                temp_flat.append(float(ValidRoute[bestrouteindex][8][1]))
                # first index is cueball and the rest follow(s)
            
                response.current_position = temp_flat
                response.obstacle_flag = ob_flag
    
        elif request.send_position == 4:
            re_position = request.update_position
            cuex = re_position[0]
            cuey = re_position[1]
            objx = []
            objy = []
            l = len(re_position)
            for i in range(2,len(re_position),2):
                objx.append(re_position[i])
                objy.append(re_position[i+1])
            m = int(l/2)-1
        
            print('-----------------------------------------------')
            print('lucky flag value:',self.lucky_flag)
            print('cue x:',cuex)
            print('cue y:',cuey)
            print('obj x:',objx)
            print('obj y:',objy)
            print('-----------------------------------------------')
            if self.lucky_flag == 0:
                re_Route, re_bestindex, _ = ps2.main(cuex, cuey, objx, objy, m) 
                new_hitpointx, new_hitpointy = ps2.findhitpoint(cuex, cuey,
                                                    re_Route[re_bestindex][3][0],re_Route[re_bestindex][3][1])
                new_yaw, rad = yaw_angle(re_Route[re_bestindex][3][0],re_Route[re_bestindex][3][1])
                score = re_Route[re_bestindex][0]

                response.current_position = [new_hitpointx, new_hitpointy, new_yaw, score]
                # response.current_position = [new_hitpointx, new_hitpointy, new_yaw, score]
            else:
                # # this is just for pausing, not lucky route
                # re_Route, re_bestindex, _ = ps2.main(cuex, cuey, objx, objy, m) 
                new_hitpointx, new_hitpointy = ps2.hitpoint(cuex, cuey, objx[0]-cuex, objy[0]-cuey)
                new_yaw, rad = yaw_angle(objx[0]-cuex, objy[0]-cuey)
                score = -6000.0
                response.current_position = [new_hitpointx, new_hitpointy, new_yaw, score]

        else:
            self.get_logger().info('waiting for proper request...')
            response.current_position = [-3.0]
            
        return response
    
def main(args=None):
    rclpy.init(args=args)

    yolo_strategy = YOLOService_2()

    rclpy.spin(yolo_strategy)

    rclpy.shutdown()

if __name__=='__main__':
    main()