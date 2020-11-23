# organize imports
import argparse
import sys
import time

import imutils
import numpy as np
import torch
import torch.nn.functional as F
from PyQt5.QtWidgets import QApplication, QWidget, QTextEdit, QVBoxLayout
import cv2

from dlcommon.json import load_dict

from model.vgg import vgg11

# work around for:
# OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class SignNet:
    def __init__(self, model_path, config_path):
        config = load_dict(config_path)
        self.classes = config['classes']
        self.class2idx = config['class2idx']
        self.idx2class = dict((v, k) for k, v in self.class2idx.items())

        self.model = vgg11(len(self.classes))
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict']
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def inference(self, img):
        """
        Perform inference on img

        :param img: opencv format [height, width, channel]
        :return: label and probabilities for each classes
        """
        torch_img = img[None, :]
        torch_img = torch.tensor([torch_img], dtype=torch.float32)
        logit = self.model(torch_img)
        prob = F.softmax(logit)
        label = torch.argmax(prob, 1)
        return label.item(), prob.detach().cpu().numpy()[0][:27]


class HandDetector:
    def __init__(self):
        self.bg = None
        self.aWeight = 0.5  # initialize weight for running average

    def init_bg(self, image):
        """
        To find the running average over the background
        """
        # initialize the background
        if self.bg is None:
            self.bg = image.copy().astype("float")
            return

        # compute weighted average, accumulate it and update the background
        cv2.accumulateWeighted(image, self.bg, self.aWeight)

    def segment(self, image, threshold=15):
        """
        To segment the region of hand in the image
        """
        # find the absolute difference between background and current frame
        diff = cv2.absdiff(self.bg.astype("uint8"), image)

        # threshold the diff image so that we get the foreground
        thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

        return thresholded


class FPSCounter:
    def __init__(self):
        self.prev_frame_time = 0
        self.count = 0
        self.acc = 0

    def get(self):
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - self.prev_frame_time)
        self.prev_frame_time = new_frame_time
        self.count += 1
        self.acc += fps
        return fps

    def avg_fps(self):
        return self.acc / self.count


class TextPreview(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Text Preview")
        self.resize(800, 400)

        self.textEdit = QTextEdit()
        self.textEdit.setFontPointSize(70)
        layout = QVBoxLayout()
        layout.addWidget(self.textEdit)
        self.setLayout(layout)

    def set_text(self, text):
        self.textEdit.setPlainText(text)


class TextVisualizer:
    def __init__(self):
        self.count = 0  # count of continuous identical sign
        self.last_label = None  # we need to track last label to accumulate 'count'
        self.count_threshold = 10

        self.text = ''
        self.last_add = ''

        self.NOTHING = 'NOTHING'
        self.DEL = 'DEL'
        self.SPACE = 'SPACE'

        self.win = TextPreview()
        self.win.show()

    def add(self, label, prob):
        if self.last_label == label:
            self.count += 1
        else:
            self.last_label = label
            self.count = 0

        current = model.idx2class[label]
        if self.count > self._get_threshold(label) and current != self.last_add:
            self.last_add = current
            if current == self.DEL:
                self.text = self.text[:-1]
                current = ''
            elif current == self.SPACE:
                current = ' '
            elif current == self.NOTHING:
                current = ''
            self.text += current
            print(model.idx2class[label] + str(self.count))

    def show(self):
        self.win.set_text(self.text)

    def _get_threshold(self, label):
        if label == self.SPACE:
            return self.count_threshold + 5
        else:
            return self.count_threshold


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='pretrained/custom/checkpoint_latest.pth')
    parser.add_argument('--config', type=str, default='pretrained/custom/class_map.json')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    camera = cv2.VideoCapture(0)
    model = SignNet(args.model, args.config)
    detector = HandDetector()
    fps_counter = FPSCounter()
    visualizer = TextVisualizer()

    # region of interest (ROI) coordinates
    top, right, bottom, left = 50, 50, 250, 250
    num_frames = 0

    while (True):
        (grabbed, frame) = camera.read()
        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)

        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray_origin = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray_origin, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 30:
            detector.init_bg(gray)
        else:
            thresholded = detector.segment(gray)

            mask = (thresholded / 255).astype("uint8")
            gray_mask = gray_origin * mask

            label, prob = model.inference(cv2.resize(gray_mask, (64, 64)))
            visualizer.add(label, prob)
            visualizer.show()

            fps = fps_counter.get()
            cv2.putText(gray_mask, str(int(fps)), (5, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.imshow('Gray', gray_mask)

        num_frames += 1
        # cv2.imshow("Video Feed", frame)

        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            break

    print('Average FPS: {}'.format(fps_counter.avg_fps()))
    camera.release()
    cv2.destroyAllWindows()
    sys.exit(app.exec_())
