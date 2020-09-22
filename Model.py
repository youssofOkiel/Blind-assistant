import cv2
import numpy as np

#Weight file =>  trained model
#Cfg file =>  configuration file
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

classes = []
with open("cocoMain.txt", "r") as file:
    classes = [line.strip() for line in file.readlines()]

print(classes)

#===================================
# it's A Fully Convolutional Neural Network 

# learned by a deep convolutional neural network to detect an object
layer_names = net.getLayerNames() # get some layers from Neural Network it based on them
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# print(output_layers)

#===============================================
# We then load the image where we want to perform the object detection and we also get its width and height.

img = cv2.imread("objects.png")
# img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape

# we need it to convert it to blob. 
#to extract feature from the image and to resize them. YOLO accepts three sizes 

blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)  #0.00392 => scaleFactor 

# for b in blob:
#     for n, img_blob in  enumerate(b):
#         cv2.imshow(str(n), img_blob)

        
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#=================================================
# then pathing blobs and layers to YOLO algorithm
net.setInput(blob)
outs = net.forward(output_layers)

#the detection is done :)


# We then loop trough the outs array to extract objests, we calculate the confidence and we choose a confidence threshold.

# some array to add rectangle , objects Names, confidence
class_ids = []
confidences = []
rects = []

for out in outs:
    for detection in out:
        scores = detection[5:]
#         print(detection)
        class_id = np.argmax(scores) #Returns the indices of the maximum values along an axis.
#         print(class_id)
        confidence = scores[class_id]
#         print(confidence)
        #here we filtering the object that's we couldn't sure if it's true object or not 
        if confidence > 0.5:
            
            #here we get the cordinate of this objet and w , h
            cen_x =int( detection[0] * width )
            cen_y = int(detection[1] * height)
            
            w =int (detection[2] * width)
            h = int(detection[3] * height)
            
            x = int( cen_x - w/2 )
            y = int( cen_y - h/2 )
            
#             cv2.circle(img, (cen_x,cen_y), 5, (0,0,255), 2)
#             print(cen_x, cen_y, w, h)
#             cv2.rectangle(img, (x,y),(x + w, y + h), (255,255,0), 1)
    
            class_ids.append(class_id)
            confidences.append(float(confidence))
            rects.append([x, y, w, h])

            
# print(class_ids, confidences, len(rects))

# to prevent dublicated 
indexes = cv2.dnn.NMSBoxes(rects, confidences, 0.5, 0.4)

font = cv2.FONT_HERSHEY_SIMPLEX
for i in range(len(rects)):
    if i in indexes:
        x, y, w, h = rects[i]
        label = classes[class_ids[i]]
        cv2.rectangle(img, (x,y), (x + w, y + h), (153,153,0), 3)
        cv2.putText(img, label+str( round(confidences[i], 2) * 100 ), (x, y), font, 0.5, (255,0,255), 2)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#==============================================================
cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)  #0.00392 => scaleFactor 
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    height, width, channels = img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores) #Returns the indices of the maximum values along an axis.
            confidence = scores[class_id]

            #here we filtering the object that's we couldn't sure if it's true object or not 
            if confidence > 0.8:

                #here we get the cordinate of this objet and w , h
                cen_x =int( detection[0] * width )
                cen_y = int(detection[1] * height)

                w =int (detection[2] * width)
                h = int(detection[3] * height)

                x = int( cen_x - w/2 )
                y = int( cen_y - h/2 )

                cv2.circle(img, (cen_x,cen_y), 5, (0,0,255), 2)
                cv2.rectangle(img, (x,y),(x + w, y + h), (0,200,150), 3)
                label = classes[class_id]
                cv2.putText(img, label+str( round(confidence, 2) * 100 ), (x, y), font, 0.5, (60,0,255), 2)


    cv2.imshow('frame', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cv2.destroyAllWindows()
cap.release()