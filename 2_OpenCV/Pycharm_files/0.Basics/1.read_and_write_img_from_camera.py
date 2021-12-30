import cv2
cap = cv2.VideoCapture(0) # file_path, rtsp
fourcc = cv2.VideoWriter.fourcc(*'XVID') #
out = cv2.VideoWriter("output.avi", fourcc, 20.0, (640, 480)) # (w, h)
print(cap.isOpened())


while (cap.isOpened()):
    ret, frame = cap.read()

    if ret:
        print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # ref: https://docs.opencv.org/3.4.15/d4/d15/group__videoio__flags__base.html#ggaeb8dd9c89c10a5c63c139bf7c4f5704dab26d2ba37086662261148e9fe93eecad
        print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out.write(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("frame", gray)

        if cv2.waitKey(1) & 0xFF == ord("q"): # waitKey(0) --> display a still image.  waitKey(1) --> display an image for 1 ms, after which, a new one will be displayed. ref: https://stackoverflow.com/questions/51143458/difference-in-output-with-waitkey0-and-waitkey1/51143586
            break

    else:
        break


cap.release()
out.release()
cv2.destroyAllWindows()

