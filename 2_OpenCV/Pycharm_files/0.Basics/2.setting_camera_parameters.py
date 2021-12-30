import cv2
cap = cv2.VideoCapture(0)
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # ref: https://docs.opencv.org/3.4.15/d4/d15/group__videoio__flags__base.html#ggaeb8dd9c89c10a5c63c139bf7c4f5704dab26d2ba37086662261148e9fe93eecad
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.set(3, 2) # only the available resolution is supported.
cap.set(4, 2) # 3 and 4

print(cap.get(3))
print(cap.get(4))

while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            cv2.imshow("frame", gray)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break
cap.release()
cv2.destroyAllWindows()