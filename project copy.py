from support.centroidtracker import CentroidTracker
from support.trackableobject import TrackableObject
import numpy as np
import dlib
import imutils
import time
from scipy import spatial
import cv2
import math
import random
from input_retrieval import *
import csv
import pandas as pd
import database
from array import array
from datetime import datetime
import track
import calc_speed
from sqlalchemy import create_engine


#All these classes will be counted as 'vehicles'
list_of_vehicles = ["car","motorbike","bus","truck"]
# Setting the threshold for the number of frames to search a vehicle for
FRAMES_BEFORE_CURRENT = 10  
inputWidth, inputHeight = 416, 416

real_distance = 20 # m

#Parse command line arguments and extract the values required
LABELS, weightsPath, configPath, inputVideoPath, outputVideoPath,\
	preDefinedConfidence, preDefinedThreshold, USE_GPU, skip_frames= parseCommandLineArguments()

# Initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")
# PURPOSE: Displays the vehicle count on the top-left corner of the frame
# PARAMETERS: Frame on which the count is displayed, the count number of vehicles 
# RETURN: N/A
def displayVehicleCount(frame, vehicle_count):
    luLine = cv2.line(img=frame, pt1=(495, 352), pt2=(820, 352), color=(255, 0, 0), thickness=2, lineType=5, shift=0)
    ldLine = cv2.line(img=frame, pt1=(350, 460), pt2=(980, 460), color=(255, 0, 0), thickness=2, lineType=5, shift=0)
    cv2.putText(
		frame, #Image
		'Detected Vehicles: ' + str(vehicle_count), #Label
		(30, 30), #Position
		cv2.FONT_HERSHEY_SIMPLEX, #Font
		0.8, #Size
		(0, 0, 255), #Color
		2, #Thickness
		cv2.FONT_HERSHEY_COMPLEX_SMALL,
		)
    # input(luLine)
    return luLine, ldLine



# PURPOSE: Displaying the FPS of the detected video
# PARAMETERS: Start time of the frame, number of frames within the same second
# RETURN: New start time, new number of frames 
def displayFPS(start_time, num_frames):
	current_time = int(time.time())
	if(current_time > start_time):
		os.system('clear') # Equivalent of CTRL+L on the terminal
		# print("FPS:", num_frames)
		num_frames = 0
		start_time = current_time
	return start_time, num_frames

# PURPOSE: Draw all the detection boxes with a green dot at the center
# RETURN: N/A
def drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame):
    # ensure at least one detection exists
    classIDsS = []
    confidencesS = []
    speeds = []
    if len(idxs) > 0:
        # loop over the indices we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            classIDsS.append(LABELS[classIDs[i]])
            confidencesS.append(confidences[i])
            # Draw a green dot in the middle of the box
            cv2.circle(frame, (x + (w//2), y+ (h//2)), 2, (0, 0xFF, 0), thickness=2)
            # , speeds
    return classIDsS, confidencesS, 1

# PURPOSE: Initializing the video writer with the output video path and the same number
# of fps, width and height as the source video 
# PARAMETERS: Width of the source video, Height of the source video, the video stream
# RETURN: The initialized video writer
def initializeVideoWriter(video_width, video_height, videoStream):
	# Getting the fps of the source video
	sourceVideofps = videoStream.get(cv2.CAP_PROP_FPS)
	# initialize our video writer
	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
	return cv2.VideoWriter(outputVideoPath, fourcc, sourceVideofps,
		(video_width, video_height), True)

WIDTH = 1280
HEIGHT = 720
def estimateSpeed(location1, location2):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    # ppm = location2[2] / carWidht
    ppm = 8.8
    d_meters = d_pixels / ppm
    #print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
    fps = 36
    speed = d_meters * fps * 3.6
    return speed

# PURPOSE: Identifying if the current box was present in the previous frames
# PARAMETERS: All the vehicular detections of the previous frames, 
#			the coordinates of the box of previous detections
# RETURN: True if the box was current box was present in the previous frames;
#		  False if the box was not present in the previous frames
def boxInPreviousFrames(previous_frame_detections, current_box, current_detections):
    centerX, centerY, width, height = current_box
    dist = np.inf #Initializing the minimum distance
    # Iterating through all the k-dimensional trees
    for i in range(FRAMES_BEFORE_CURRENT):
        coordinate_list = list(previous_frame_detections[i].keys())
        if len(coordinate_list) == 0: # When there are no detections in the previous frame
            continue
        # Finding the distance to the closest point and the index
        temp_dist, index = spatial.KDTree(coordinate_list).query([(centerX, centerY)])
        if (temp_dist < dist):
            dist = temp_dist
            frame_num = i
            coord = coordinate_list[index[0]]
    
    if (dist > (max(width, height)/2)):
        return False
    # Keeping the vehicle ID constant
    current_detections[(centerX, centerY)] = previous_frame_detections[frame_num][coord]
    return True


# to check if objects are close or not
angle_factor = 10.0
H_zoom_factor = 0.5

def dist(c1, c2):
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5

def T2S(T):
    S = abs(T/((1+T**2)**0.5))
    return S

def T2C(T):
    C = abs(1/((1+T**2)**0.5))
    return C

def isclose(p1,p2):
    
    c_d = dist(p1[2], p2[2])
    if(p1[1]<p2[1]):
        a_w = p1[0]
        a_h = p1[1]
    else:
        a_w = p2[0]
        a_h = p2[1]

    T = 0
    try:
        T=(p2[2][1]-p1[2][1])/(p2[2][0]-p1[2][0])
    except ZeroDivisionError:
        T = 1.633123935319537e+16
    S = T2S(T)
    C = T2C(T)
    d_hor = C*c_d
    d_ver = S*c_d
    vc_calib_hor = a_w*1.3
    vc_calib_ver = a_h*0.4*angle_factor
    c_calib_hor = a_w *1.7
    c_calib_ver = a_h*0.2*angle_factor
    # print(p1[2], p2[2],(vc_calib_hor,d_hor),(vc_calib_ver,d_ver))
    if (0<d_hor<vc_calib_hor and 0<d_ver<vc_calib_ver):
        return 1
    elif 0<d_hor<c_calib_hor and 0<d_ver<c_calib_ver:
        return 2
    else:
        return 0


# function to count, detect, track vahicles and calculate speed, distance between objects 
def count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame):
    current_detections = {}
    distances = []
    safes = []
    boxs = []
    distance = []
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indices we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            boxs.append((x, y, w, h))
            centerX = x + (w//2)
            centerY = y + (h//2)
            # When the detection is in the list of vehicles, AND
            # it crosses the line AND
            # the ID of the detection is not present in the vehicles
            if (LABELS[classIDs[i]] in list_of_vehicles):
                current_detections[(centerX, centerY)] = vehicle_count 
            if (not boxInPreviousFrames(previous_frame_detections, (centerX, centerY, w, h), current_detections)):
                vehicle_count += 1
                
                
            # Add the current detection mid-point of box to the list of detected items
            # Get the ID corresponding to the current detection
            ID = current_detections.get((centerX, centerY))
            # If there are two detections having the same ID due to being too close, 
            # then assign a new ID to current detection.
            if (list(current_detections.values()).count(ID) > 1):
                current_detections[(centerX, centerY)] = vehicle_count
                vehicle_count += 1
            
            
            for b in range(len(boxes)):
                for k in range(b+1, len(boxes)):
                    di = distObj(boxes[b][0], boxes[b][1], boxes[b][2], boxes[b][3], boxes[k][0], boxes[k][1], boxes[k][2], boxes[k][3])
                    distances.append(di)
            if len(idxs) > 0:
                status = []
                idf = idxs.flatten()
                close_pair = []
                S_close_pair = []
                center = []
                co_info = []
                for i in idf:
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    cen = [int(x + w / 2), int(y + h / 2)]
                    
                    center.append(cen)
                    cv2.circle(frame, tuple(cen), 1, (0, 0, 0), 1)
                    co_info.append([w, h, cen])
                    status.append(0)
                for i in range(len(center)):
                    for j in range(len(center)):                       
                        g = isclose(co_info[i], co_info[j])
                        if g == 1:
                            close_pair.append([center[i], center[j]])
                            centerS, Dists = DistCenter(center[i][0], center[i][1], center[j][0], center[j][1])
                            status[i] = 1
                            status[j] = 1
                        elif g == 2:
                            S_close_pair.append([center[i], center[j]])
                            if status[i] != 1:
                                status[i] = 2
                            if status[j] != 1:
                                status[j] = 2
                        else:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 120, 255), 2)
                            
                total_p = len(center)
                low_risk_p = status.count(2)
                high_risk_p = status.count(1)
                safe_p = status.count(0)
                kk = 0
                for i in idf:
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    cens = [int(x + w / 2), int(y + h / 2)]
                    if status[kk] == 1:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 150), 2)
                        if distances[i] < 35:
                            dis = "{:.2f} m".format(distances[i])
                            cv2.putText(frame,"dist: "+dis, (cens[0], cens[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 0, 0), 2)
                    elif status[kk] == 0:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    else:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 120, 255), 2)
                        if distances[i] < 35:
                            dis = "{:.2f} m".format(distances[i])
                            cv2.putText(frame,"dist: "+dis, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 0, 0), 2)
        
                    kk +=1
                
                    
                for h in close_pair:
                    cv2.line(frame, tuple(h[0]), tuple(h[1]), (0, 0, 255), 2)
                for b in S_close_pair:
                    cv2.line(frame, tuple(b[0]), tuple(b[1]), (0, 255, 255), 2)
            
            # Notifications about the safety of the each vehicle
            if di < 10:
                safe = "distance very small"
                safes.append(safe)
            elif di < 20:
                safe = "Good distance"
                safes.append(safe)
            else:
                safe = "SAFE"
                safes.append(safe)
            
            # Display the ID at the center of the box
            cv2.putText(frame, str(ID), (centerX+3, centerY+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,255], 2)
    
    return vehicle_count, current_detections, classIDs, distances, safes,boxs


def midpoint(x, y, w, h):
    	return ((x + w) * 0.5, (y + h) * 0.5)

def distObj(x1,y1,w1,h1,x2,y2,w2,h2):
    cx1, cy1 = midpoint(x1,y1,w1,h1)
    cx2, cy2 = midpoint(x2,y2,w2,h2)
    distance = math.sqrt((cx2 - cx1)  ** 2 + (cy2 - cy1) ** 2)
    return distance

def DistCenter(x1, y1, x2, y2):
    center = []
    dists = []
    centerX, centerY = ((x1 + x2) / 2), ((y1 + y2) / 2)
    center.append([centerX, centerY])
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    dists.append(distance)
    return center, dists
# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

#Using GPU if flag is passed
if USE_GPU:
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
 
videoStream = cv2.VideoCapture(inputVideoPath) # to use the laptop web Cam change to: videoStream = cv2.VideoCapture(0)
video_width = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))


#Initialization
previous_frame_detections = [{(0,0):0} for i in range(FRAMES_BEFORE_CURRENT)]
# previous_frame_detections = [spatial.KDTree([(0,0)])]*FRAMES_BEFORE_CURRENT 

# Initializing all trees
num_frames, vehicle_count = 0, 0
writer = initializeVideoWriter(video_width, video_height, videoStream)
start_time = int(time.time())
FR = 0
(W, H) = (None, None)


ids = []
safes = []
trackableOjects = {}
# Variables used to calculate the speed 
frameCounter = 0
currentCarID = 0
fps = 0
carTracker = {}
carNumbers = {}
carLocation1 = {}
carLocation2 = {}
speed = [None] * 1000

# init centroid tracker
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
# loop over frames from the video file stream
while True:
    print("================NEW FRAME================")
    
    rects = []
    # num_frames = frameCounter #######
    num_frames+= 1
    print("FRAME:\t", num_frames)
    # Initialization for each iteration
    boxes, confidences, classIDs = [], [], [] 
    cx, cy = {}, {}
    vehicle_crossed_line_flag = False 


	# Calculating fps each second
    s_time, numFrames = displayFPS(start_time, num_frames)
	# read the next frame from the file
    (grabbed, frame) = videoStream.read()
    

	# if the frame was not grabbed, then we have reached the end of the stream
    if not grabbed:
        break
    
    
    if W is None or H is None:
        (H, W) = frame.shape[:2]
        FW=W
        if(W<1075):
            FW = 1075
        FR = np.zeros((H+210,FW,3), np.uint8)

        col = (255,255,255)
        FH = H + 210
    FR[:] = col
    

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (inputWidth, inputHeight),swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    

	# loop over each of the layer outputs
    for output in layerOutputs:
		# loop over each of the detections
        for i, detection in enumerate(output):
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # to ensure vehicles detection 
            if LABELS[classID] in list_of_vehicles:
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > preDefinedConfidence:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([video_width, video_height, video_width, video_height])
                    (centerX, centerY, width, height) = box.astype("int")
                    boxsx = box.astype("int")
                    # print("box ", box[1])
                    # use the center (x, y)-coordinates to derive the top
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
                    # imgCropped = frame[x: x , y: y ]
                    # for i in range(len(boxes)):
                    #     # print(boxes[i][0])
                    #     imgCropped = frame[boxes[i][0]: int(boxes[i][2]), boxes[i][1]: int(boxes[i][3])]
                    #     print(imgCropped)
                    # Save detection time 
                    nowDate = str(datetime.now().date())
                    nowTime = str(datetime.now().time())
                    now = nowTime + ' / ' + nowDate
                    # current_time = now.strftime("%H:%M:%S:%N")
                    print("Current Time =", now)
                    
                        
                    # img_cropped = frame[x1:x1+w1, y1:y1+h1]
                    # for i in range(len(boxes)):
                    #     print(boxes[i], classIDs)
                    # for i, pos in enumerate(boxes):
                    #     print(i, pos)
                        # print(d)
                    
                    
    
    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, preDefinedConfidence,preDefinedThreshold)
    if num_frames % skip_frames == 0:
        for i in idxs:
            for j in i:
                print(j)
        # print(i)
        
    
    # Draw detection box 
    classIDsS, confidencesS, speeds = drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame)
    vehicle_count, current_detections, classIDs, distances, safe, boxs = count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame)
    
    # # calculate speed
    # objects = ct.update(rects)
    # speed = 0
    # dist = 0
    # for (objectID, centroid) in objects.items():
    #     # init speed array
    #     # speed = 0
    #     # print(len(centroid))
    #     # check to see if a tracktable object exists for the current objectID
    #     to = trackableOjects.get(objectID, None)

    #     # if there is no tracktable object, create one
    #     if to is None:
    #         to = TrackableObject(objectID, centroid)
    #         # print(to.centroids)
    #     # otherwise, use it for speed estimation
    #     else:
    #         to.centroids.append(centroid)
    #         # print(len(to.centroids))
    #         location1 = to.centroids[-2]
    #         location2 = to.centroids[-1]
    #         location3 = to.centroids[0]
    #         location4 = to.centroids[1]
    #         # location5 = to.centroids[-3]
    #         print(to.centroids)
    #         # print(to.centroids[-2], to.centroids[-1], to.centroids[0], to.centroids[1])
            
    #         # print("location1: ",location1, "location2", location2, "location3", location3, "location4", location4)
    #         speed = estimateSpeed(location1, location2, location3, location4)
    #                                                                                                                                                                                                     # speed = random.randint(80, 140)
    #         # print(objectID, speed)
    #     trackableOjects[objectID] = to
        
	# Display Vehicle Count if a vehicle has passed the line 
    luLine, ldLine = displayVehicleCount(frame, vehicle_count)
    end = time.time()
    
    timee = end - start
    # print("time",type(timee))
    
    
    # init the status for detecting or tracking
    for centerPos, id in current_detections.items():
        temp = [centerPos, id]
        ids.append(temp[1])    
    
    # Create a Dataframe with all returned Values 
    dfree = pd.DataFrame(columns=["vehicle_id", "vehicle_class", "accuracy", "speed", "position", "state"])
    dfree["vehicle_id"] = idxs.tolist()
    dfree["vehicle_class"] = classIDsS
    dfree["accuracy"] = confidencesS
    dfree["speed"] = boxs
    dfree["position"] = speeds
    dfree["state"] = safe
    dfree["time"] = now
    
    # Create connection to postgres database
    connect = "postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/postgres_db"
    engine = create_engine(connect)
    
    #write dataframe to database
    dfree.to_sql('api_vehicle', con=engine, index=False, if_exists='append')
    
    
    
    # write the output frame to disk
    writer.write(frame) 
    # cv2.imshow('imgCropped', imgCropped)
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break	
	
	# Updating with the current frame detections
    previous_frame_detections.pop(0) #Removing the first frame from the list
	# previous_frame_detections.append(spatial.KDTree(current_detections))
    previous_frame_detections.append(current_detections)
    

# connection to postgres database
# from sqlalchemy import create_engine
writer.release()
videoStream.release()
