import cv2
import numpy as np

#Loading the algorithm
nNetwork = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
#loading the classes
classes = []
with open("dataset.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
print("List Of Objects:",classes)

#config the nNetwork object
layer_names = nNetwork.getLayerNames()
output_layers = [layer_names[i[0]-1] for i in nNetwork.getUnconnectedOutLayers()]

#defining color for each detected object
colors = np.random.uniform(0,255,size=(len(classes),3))

#loading and resizing the images
img = cv2.imread("img/3d.png")
img = cv2.resize(img, None, fx=0.9, fy=0.9)
height, width, channels = img.shape

#detecting and extracting features from the image
blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, crop=False)

#processing images by the algorithm
nNetwork.setInput(blob)
out = nNetwork.forward(output_layers)
#showing informations
class_ids = []
confs = []
boxs = []
for outs in out:
	for detect in outs:
		score = detect[5:]
		class_id = np.argmax(score)
		conf = score[class_id]
		if conf > 0.5:
			#object detection and its cordenates
			center_x = int(detect[0] * width)
			center_y = int(detect[1] * height)
			w = int(detect[2] * width)
			h = int(detect[3] * height)

			#rec cordenates
			x = int(center_x - w / 2)
			y = int(center_y - h / 2)

			#all rec areas
			boxs.append([x,y,w,h])
			#how conf its of the detection
			confs.append(float(conf))
			#to know the neme of the object detcted
			class_ids.append(class_id)

#looking thru the objects
objects_detected = len(boxs)
print("Number of Objects detected:", objects_detected)
#removing the double boxes/noise
indexs = cv2.dnn.NMSBoxes(boxs, confs, 0.5, 0.4)
print("Number of indexs:",indexs)
#defining fonts
font = cv2.FONT_HERSHEY_PLAIN
text_thikness = int((w * h) / (1000 * 1000))
for i in range(len(boxs)):
	if i in indexs:
		x,y,w,h = boxs[i]
		label = classes[class_ids[i]]
		color = colors[i]
		cv2.rectangle(img, (x,y), (x+w, y+h), color, 1)
		#puting text to the image
		cv2.putText(img, label, (x+15,y+30), font, 2, (0,0,0), text_thikness)
		print("Object Detected:",label)

#displaying
cv2.imshow("Object Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()