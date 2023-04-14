import obd

# connect to the vehicle's BCM
connection = obd.Async("eth0")

# wait for the connection to be established
if not connection.is_connected():
    connection.wait()

# get the VIN from the vehicle's BCM
response = connection.query(obd.commands.VIN)

# print the VIN
print(response.value)

from doip import DoIPClient
from doip import VehicleIdentification

# Define the VIN query function
def query_vin(client):
    vin = None
    vehicle_identification = client.get_vehicle_identification()
    if vehicle_identification is not None:
        vin = vehicle_identification.vehicle_id_number
    return vin

# Connect to the vehicle
client = DoIPClient('192.168.0.10')  # Replace with your vehicle's IP address
client.connect()

# Check if the connection was successful
if client.is_connected():
    # Retrieve VIN number
    vin = query_vin(client)
    if vin is None:
        print("VIN number not found")
    else:
        print("VIN number: " + vin)
else:
    print("Unable to connect to vehicle")

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow.compat.v1 as tf
import matplotlib.patches as patches
from local_utils import reconstruct
import os
import argparse
import imutils
import time

# python main.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

#resize OPENCV window size
#cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

# load model architecture and weights
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize camera stream
print("[INFO] starting video stream...")

vid=cv2.VideoCapture(0)

# Set the width and height of the video capture
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Set the delay between frames (in milliseconds)

def preprocess_image(image_path,resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img
delay = 2000
def get_plate(image_path, sess, Dmax=608, Dmin = 608):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    processed_image, Iresized = prepare_image_for_tf_inferencing(vehicle, bound_dim)
    infrencing = sess.run(output_tensor, {'x:0': processed_image})
    np_inferencing = np.squeeze(infrencing)
    _ , LpImg, _, cor = detect_lp_tf(Iresized, np_inferencing, vehicle, lp_threshold=0.5)
    return vehicle, LpImg, cor

def prepare_image_for_tf_inferencing(vehicle, bound_dim):
    min_dim_img = min(vehicle.shape[:2])
    factor = float(bound_dim) / min_dim_img
    w, h = (np.array(vehicle.shape[1::-1], dtype=float) * factor).astype(int).tolist()
    Iresized = cv2.resize(vehicle, (w, h))
    T = Iresized.copy()
    T = T.reshape((1, T.shape[0], T.shape[1], T.shape[2]))
    return T, Iresized

def detect_lp_tf(Iresized, np_inferencing, I, lp_threshold):
    L, TLp, lp_type, Cor = reconstruct(I, Iresized, np_inferencing, lp_threshold)
    return L, TLp, lp_type, Cor


## Loading model
sess=tf.InteractiveSession()
frozen_graph="simple_frozen_graph.pb"
with tf.gfile.GFile(frozen_graph, "rb") as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
sess.graph.as_default()
tf.import_graph_def(graph_def)
# Frozen model inputs: 
# [<tf.Tensor 'x:0' shape=(None, None, None, 3) dtype=float32>]
# Frozen model outputs: 
# [<tf.Tensor 'Identity:0' shape=(None, None, None, 8) dtype=float32>]  
input_tensor = sess.graph.get_tensor_by_name("x:0") 
output_tensor = sess.graph.get_tensor_by_name("Identity:0")        
print("Tensor Input : ", input_tensor)
print("Tensor Output: ", output_tensor)
print("..... Extracing Number Plate .......")

image_dir = r'imgs'

# Get the list of all subfolders in the image directory
subfolders = [f.path for f in os.scandir(image_dir) if f.is_dir()]

# Initialize the index of the current subfolder and image
subfolder_index = 0
image_index = 0
while True:
    ret,frame = vid.read()
    try:# read frame from camer and resize to 400 pixels
        
        frame = imutils.resize(frame, width=400)
 
		# grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
 
		# pass the blob through the network and obtain the detections and
		# predictions
        net.setInput(blob)
        detections = net.forward()

		# loop over the detections
        for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
            confidence = detections[0, 0, i, 2]

		# filter out detections by confidence
            if confidence < args["confidence"]:
                continue

		# compute the (x, y)-coordinates of the bounding box for the
		# object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
 
		# draw the bounding box of the face along with the associated
		# probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10

			# Grab ROI with Numpy slicing and blur
            ROI = frame[startY:startY+(endY-startY), startX:startX+(endX-startX)]
            blur = cv2.GaussianBlur(ROI, (51,51), 0) 
			# Insert ROI back into image
            frame[startY:startY+(endY-startY), startX:startX+(endX-startX)] = blur 
            cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),	cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            
    except:
         print("plate")
    resized_frame = cv2.resize(frame, (960, 540))
    print(resized_frame.shape)
    cv2.imwrite("cam.jpg", resized_frame)

    images = os.listdir(subfolders[subfolder_index])
    test_image_path = os.path.join(subfolders[subfolder_index], images[image_index])
    print(test_image_path)
    #test_image_path = cv2.imread(image_path)
    try:
        vehicle, LpImg,cor = get_plate(test_image_path, sess)
    except AssertionError:
        print("No plates found")
    for i in cor:
        startX=int(i[0][0])
        startY=int(i[1][0])
        endX=int(i[0][2])
        endY=int(i[1][2])
        ROI = vehicle[startY:startY+(endY-startY), startX:startX+(endX-startX)]
        blur = cv2.GaussianBlur(ROI, (51,51), 0) 
	    # Insert ROI back into image
        vehicle[startY:startY+(endY-startY), startX:startX+(endX-startX)] = blur 
        #cv2.rectangle(vehicle, (int(i[0][0]), int(i[1][0])), (int(i[0][2]), int(i[1][2])),(0, 0, 255), 2)
    imS = cv2.resize(vehicle, (960, 540)) 
    print(imS.shape)

     # Combine the frame and image
    cv2.imwrite("car.jpg", imS)
    #combined_frame = cv2.hconcat([resized_frame, resized_frame])q
    # Add text to the final frame
    text = "Blurred faces and license plates"
    #cv2.putText(combined_frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the combined frame
    #cv2.imshow('Combined', combined_frame)
   
    #vis = np.concatenate((resized_frame, imS), axis=1)
    cv2.imshow('car', imS)
    cv2.imshow("cam",resized_frame)
    print(len(cor))
    image_index += 1

    # If the end of the current subfolder is reached, move to the next subfolder
    if image_index == len(images):
        subfolder_index += 1
        image_index = 0

    # If the end of the subfolders is reached, start from the beginning
    if subfolder_index == len(subfolders):
        subfolder_index = 0

    # Exit on 'q' key press
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Release video capture object and destroy all windows

cv2.destroyAllWindows()

from doip import DoIPClient, DiagnosticSessionControlType, EcuResetType

client = DoIPClient()
client.connect("127.0.0.1", 13400)

client.open_diagnostic_session(0x01, 0x10)

response = client.send_request(0x22, [0x90])

client.close_diagnostic_session()
client.reset_ecu(EcuResetType.hard_reset)

vin = response[3:].decode("ascii")
print(vin)

import obd

# Establish connection to OBD-II port
connection = obd.OBD()

# Check if the connection was successful
if connection.is_connected():
    # Retrieve VIN number
    response = connection.query(obd.commands.GET_VIN)
    if response.is_null():
        print("VIN number not found")
    else:
        print("VIN number: " + str(response.value))
else:
    print("Unable to connect to OBD-II port")

