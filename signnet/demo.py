import torch

from model.vgg import vgg11
import cv2
import numpy as np

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
    model.load_state_dict(torch.load('./gesture_cnn_t2.pt', map_location=torch.device('cpu')))
    model.eval()

    # Pytorch require B, C, H, W
    torch_img = np.transpose(img, (2, 0, 1))
    torch_img = torch.tensor([torch_img], dtype=torch.float32)
    print(torch_img.shape)
    out = model(torch_img)
    print(out.shape)
    label = torch.argmax(out, 1)
    print(label.shape)
    return label.item()


if __name__ == '__main__':
    # path = 'data/asl_alphabet_train/asl_alphabet_train/C/C8.jpg'
    # path = 'opencv_frame_6.png'
    path = 'data/asl_alphabet_test/asl_alphabet_test/B/B_test.jpg'
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    label = inference(img)
    print(label)
    print(idx2class[label])
