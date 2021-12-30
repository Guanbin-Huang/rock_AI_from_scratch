import cv2
import numpy as np
cap = cv2.VideoCapture(0) # file_name, rtsp
# cap = cv2.VideoCapture("./a.mp4")

fourcc = cv2.VideoWriter.fourcc(*'XVID')

out = cv2.VideoWriter("output.avi", fourcc, 20.0, (640,480)) #!!!! (w, h)
print(cap.isOpened())

while (cap.isOpened()):
    ret, frame = cap.read()

    if ret:
        print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out.write(gray)
        cv2.imshow("frame", gray)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()