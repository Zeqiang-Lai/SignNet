# organize imports
import cv2
import imutils
import torch

from model.vgg import vgg11

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
           'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

class2idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12,
             'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24,
             'Z': 25, 'del': 26, 'nothing': 27, 'space': 28}

idx2class = dict((v, k) for k, v in class2idx.items())


def inference(img):
    """
    Perform inference on img
    :param img: opencv format [height, width, channel]
    :return:
    """
    model = vgg11()
    checkpoint = torch.load('experiments/checkpoint_latest.pth', map_location=torch.device('cpu'))['model_state_dict']
    model.load_state_dict(checkpoint)
    model.eval()

    # Pytorch require B, C, H, W   transforms.RandomHorizontalFlip(),
    #         transforms.Grayscale(),
    #         transforms.Resize((64, 64)),
    #         transforms.ToTensor(),
    torch_img = img[None, :]
    torch_img = torch.tensor([torch_img], dtype=torch.float32)

    print(torch_img.shape)
    out = model(torch_img)
    print(out.shape)
    label = torch.argmax(out, 1)
    print(label.shape)
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


# -----------------
# MAIN FUNCTION
# -----------------
if __name__ == "__main__":
    # initialize weight for running average
    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 50, 50, 250, 250

    # initialize num of frames
    num_frames = 0

    # out = cv2.VideoWriter('filename.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (200, 200), 0)

    # keep looping, until interrupted
    while (True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
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
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                cv2.imshow("Thesholded", thresholded)

                mask = (thresholded / 255).astype("uint8")

                gray_mask = gray_origin * mask
                print(gray_mask.shape)
                cv2.imshow('gray', gray_mask)

                label = inference(cv2.resize(gray_mask, (64, 64)))
                print(label)
                print(idx2class[label])
                cv2.waitKey(100)
                # out.write(gray_mask)

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

    # free up memory
    camera.release()
    # out.release()
    cv2.destroyAllWindows()
