import cv2
while True:
    img=cv2.imread("x.jpeg")
    imgColor=cv2.cvtColor(img[0],cv2.COLOR_GRAY2BGR)
    cv2.imshow("color",imgColor)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break