
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
import imutils
import os
import urllib
# camera = cv2.VideoCapture(0)
# URL = "http://192.168.1.6:8080/video"
# cap = cv2.VideoCapture(URL)



h, w = None, None

with open('yolo-coco-data/coco.names') as f:
    labels = [line.strip() for line in f]


network = cv2.dnn.readNetFromDarknet('yolo-coco-data/yolov3.cfg',
                                     'yolo-coco-data/yolov3.weights')


layers_names_all = network.getLayerNames()


layers_names_output = \
    [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]


probability_minimum = 0.50
threshold = 0.30




while True:

        # ret, frame = cap.read()


        # _, frame = camera.read()
    value = "./input/deneme"

    image_list = [os.path.join(value, f) for f in os.listdir(value) if f.endswith('.jpeg')]
    for image in image_list:
        frame = np.uint8(cv2.imread(image))


        if w is None or h is None:
            h, w = frame.shape[:2]


        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (412, 412),
                                     swapRB=True, crop=False)



        network.setInput(blob)
        start = time.time()
        output_from_network = network.forward(layers_names_output)
        end = time.time()


        bounding_boxes = []
        confidences = []
        class_numbers = []


        for result in output_from_network:

            for detected_objects in result:

                scores = detected_objects[5:]
                class_current = np.argmax(scores)
                confidence_current = scores[class_current]


                if confidence_current > probability_minimum and class_current == 41 or class_current == 39  :


                    box_current = detected_objects[0:4] * np.array([w, h, w, h])


                    x_center, y_center, box_width, box_height = box_current
                    x_min = int(x_center - (box_width/2))
                    y_min = int(y_center - (box_height / 2))


                    bounding_boxes.append([x_min, y_min,int(box_width), int(box_height)])
                    confidences.append(float(confidence_current))
                    class_numbers.append(class_current)


        results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                                   probability_minimum, threshold)


        if len(results) > 0:


            for i in results.flatten():


                x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
                class_number = int(class_numbers[i])

                frame2 = frame[y_min:box_height+y_min , x_min:box_width+x_min]
                hsv_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

                font_size = 1

                low_green = np.array([36, 25, 25])
                high_green = np.array([70, 255, 255])

                low_blue = np.array([96, 60, 0])
                high_blue = np.array([131, 255, 255])

                low_kirmizi = np.array([0, 120, 70])
                high_kirmizi = np.array([10, 255, 255])

                low_orange = np.array([5, 50, 50])
                high_orange = np.array([15, 255, 255])

                low_neue = np.array([0, 0, 0])
                high_neue = np.array([0, 0, 0])


                if(class_number==39):



                    # Grün
                    mask_green = cv2.inRange(hsv_frame, low_green, high_green)
                    green  = cv2.bitwise_and(frame2, frame2, mask=mask_green)
                    # cv2.imshow("g", green )

                    # Blau

                    mask_blue = cv2.inRange(hsv_frame, low_blue, high_blue)
                    blue = cv2.bitwise_and(frame2, frame2, mask=mask_blue)
                    # cv2.imshow("b", blue)

                    # Rot

                    kirmizi_mask1 = cv2.inRange(hsv_frame, low_kirmizi, high_kirmizi)

                    low_red = np.array([170, 120, 70])
                    high_red = np.array([180, 255, 255])
                    mask_red = cv2.inRange(hsv_frame, low_red, high_red)

                    kirmizi_mask = kirmizi_mask1 + mask_red

                    # Orange
                    orange_mask = cv2.inRange(hsv_frame, low_orange, high_orange)
                    orange = cv2.bitwise_and(frame2, frame2, mask=orange_mask)
                    # cv2.imshow("o", orange)



                    cnts1 = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cnts1 = imutils.grab_contours(cnts1)

                    cnts2 = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cnts2 = imutils.grab_contours(cnts2)

                    cnts3 = cv2.findContours(kirmizi_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cnts3 = imutils.grab_contours(cnts3)

                    cnts4 = cv2.findContours(orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cnts4 = imutils.grab_contours(cnts4)

                    font = cv2.FONT_HERSHEY_PLAIN
                    farbe_wert=(0, 0, 0)

                    text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                           confidences[i])


                    cnts=[[cnts1,"Gruen"],[cnts2,"Blau"],[cnts3,"Rot"],[cnts4,"Orange"]]



                    for c in cnts:
                        if len(c[0])>0:
                            c[0] = max(c[0], key=cv2.contourArea)
                            a, b, y, d = cv2.boundingRect(c[0])
                            if y*d > box_width * box_height * 15 / 100:
                                fuellstand = float(d / box_height)
                                if(c[1]=="Gruen"):
                                    farbe_wert = (0, 128, 0)
                                    farbe_str="Gruen"
                                elif (c[1] == "Blau"):
                                    farbe_wert = (255, 0, 0)
                                    farbe_str = "Blau"
                                elif (c[1] == "Rot"):
                                    farbe_wert = (0,0,255)
                                    farbe_str = "Rot"
                                elif (c[1] == "Orange"):
                                    farbe_wert = (0,127,255)
                                    farbe_str = "Orange"
                                else:
                                    farbe_wert = (0, 0, 0)
                                    farbe_str = "Andere Farbe"
                                fuellstand_str = str(fuellstand)
                                fuellstand_str = "Fuellstandsrate: " + fuellstand_str[0:4]
                                cv2.putText(frame, text_box_current, (x_min, y_min - 55), cv2.FONT_HERSHEY_SIMPLEX,
                                            font_size,
                                            farbe_wert, 2)
                                cv2.putText(frame, fuellstand_str, (x_min, y_min - 25), cv2.FONT_HERSHEY_SIMPLEX, font_size,
                                            farbe_wert, 2)
                                cv2.putText(frame, "Farbe: "+farbe_str, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, font_size,
                                            farbe_wert, 2)





                    cv2.rectangle(frame, (x_min, y_min), (x_min + box_width, y_min + box_height), farbe_wert, 2)



                    print("Flasche")
                elif (class_number==41):


                    # Rot
                    kirmizi_mask = cv2.inRange(hsv_frame, low_kirmizi, high_kirmizi)
                    kirmizi = cv2.bitwise_and(frame2, frame2, mask=kirmizi_mask)
                    result_renk_kirmizi = np.count_nonzero(np.all(frame2 == kirmizi, axis=2))

                    # Orange
                    orange_mask = cv2.inRange(hsv_frame, low_orange, high_orange)
                    orange = cv2.bitwise_and(frame2, frame2, mask=orange_mask)
                    result_renk_turuncu = np.count_nonzero(np.all(frame2 == orange, axis=2))
                    result_normal = np.count_nonzero(np.all(frame2, axis=2))


                    # Green
                    green_mask = cv2.inRange(hsv_frame, low_green, high_green)
                    green = cv2.bitwise_and(frame2, frame2, mask=green_mask)
                    result_renk_green = np.count_nonzero(np.all(frame2 == green, axis=2))

                    # Blau

                    mask_blue = cv2.inRange(hsv_frame, low_blue, high_blue)
                    blue = cv2.bitwise_and(frame2, frame2, mask=mask_blue)
                    result_renk_blau = np.count_nonzero(np.all(frame2 == blue, axis=2))

                    if result_renk_turuncu > result_normal * 0.3:
                        farbe_str="Orange"
                        farbe_wert= (0, 127, 255)
                    elif result_renk_green > result_normal * 0.3:
                        farbe_str="Gruen"
                        farbe_wert=  (0, 128, 0)
                    elif result_renk_kirmizi > result_normal * 0.3:
                        farbe_str="Rot"
                        farbe_wert=  (0, 0, 255)
                    elif result_renk_blau > result_normal * 0.3:
                        farbe_str = "Blau"
                        farbe_wert = (255, 0, 0)
                    else:
                        farbe_str="Keine Farbe"
                        farbe_wert=  (0, 0, 0)


                    cv2.rectangle(frame, (x_min, y_min),
                                  (x_min + box_width, y_min + box_height),
                                  farbe_wert, 2)

                    text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                           confidences[i])



                    cv2.putText(frame, text_box_current, (x_min, y_min - 28),
                                cv2.FONT_HERSHEY_SIMPLEX, font_size, farbe_wert, 2)
                    cv2.putText(frame, "Farbe: "+farbe_str, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, font_size, farbe_wert,
                                2)
                    print("Tasse")




        cv2.namedWindow('Objekterkennung', cv2.WINDOW_NORMAL)

        cv2.imshow('Objekterkennung', frame)
        cv2.waitKey(0)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



# camera.release()
cv2.destroyAllWindows()