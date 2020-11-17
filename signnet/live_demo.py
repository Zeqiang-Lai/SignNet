# organize imports
import time

import cv2
import imutils
import torch
import argparse

from model.vgg import vgg11
from dlcommon.json import load_dict


class SignNet:
    def __init__(self, model_path, config_path):
        self.model = vgg11()
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict']
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        config = load_dict(config_path)
        self.classes = config['classes']
        self.class2idx = config['class2idx']
        self.idx2class = dict((v, k) for k, v in self.class2idx.items())

    def inference(self, img):
        """
        Perform inference on img

        :param img: opencv format [height, width, channel]
        :return:
        """
        torch_img = img[None, :]
        torch_img = torch.tensor([torch_img], dtype=torch.float32)
        out = self.model(torch_img)
        label = torch.argmax(out, 1)
        return label.item()


# global variables
bg = None


# --------------------------------------------------
# To find the running average over the background
# --------------------------------------------------
def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)


# ---------------------------------------------
# To segment the region of hand in the image
# ---------------------------------------------
def segment(image, threshold=15):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    cnts, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='experiments/lzq/checkpoint_latest.pth')
    parser.add_argument('--config', type=str, default='experiments/lzq/class_map.json')
    args = parser.parse_args()

    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 50, 50, 250, 250

    aWeight = 0.5  # initialize weight for running average
    num_frames = 0

    count = 0  # count of continuous identical sign
    last_label = None  # we need to track last label to accumulate 'count'

    # we use these two var to calculate fps
    prev_frame_time = 0
    new_frame_time = 0

    model = SignNet(args.model, args.config)

    while (True):
        (grabbed, frame) = camera.read()

        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)

        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray_origin = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray_origin, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            hand = segment(gray)

            if hand is not None:
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                # cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                # cv2.imshow("Thesholded", thresholded)

                mask = (thresholded / 255).astype("uint8")
                gray_mask = gray_origin * mask

                label = model.inference(cv2.resize(gray_mask, (64, 64)))

                if last_label == label:
                    count += 1
                else:
                    last_label = label
                    count = 0

                if count > 5:
                    print(model.idx2class[label] + str(count))

                new_frame_time = time.time()
                fps = 1 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time
                fps = int(fps)

                cv2.putText(gray_mask, str(fps), (5, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.imshow('gray', gray_mask)

        # draw the segmented hand
        # cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

        num_frames += 1
        cv2.imshow("Video Feed", clone)

        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
