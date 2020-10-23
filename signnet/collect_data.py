import cv2


def crop_center(frame):
    height = frame.shape[0]
    width = frame.shape[1]
    extra = (width - height) // 2
    return frame[:, extra:extra + width]


if __name__ == '__main__':

    cam = cv2.VideoCapture(0)

    img_counter = 0

    while True:
        ret, frame = cam.read()

        left, top, right, bottom = 100, 100, 600, 600

        if not ret:
            print("failed to grab frame")
            break
        else:
            # frame = crop_center(frame)
            frame = cv2.flip(frame, 180)
            roi = frame[top:bottom, left:right]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        gray = cv2.resize(gray, (100, 100))

        cv2.imshow("test", frame)
        cv2.imshow("Gray", gray)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, gray)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()

    cv2.destroyAllWindows()
