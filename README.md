<img src='imgs/header.png'/>

# SignNet

### [Report](report/report.pdf) |[Pretrained Model](https://drive.google.com/drive/folders/1nvcGXNynlo0k259ACTKcOLiGnMVcgyyF?usp=sharing)| [Demo]()

**SignNet: Recognize Alphabets in the American Sign Language in Real Time**

[Zeqiang Lai](https://github.com/Zeqiang-Lai) , [Zhiyuan Liang](https://github.com/zhiyuan0112) , [Kexiang Huang](https://github.com/YellCanFly)

Beijing Institute of Technology

> Note that this is a simple course project rather than a serious research project.

## Getting Started

1. **Clone the repository**

```shell
git clone https://github.com/Zeqiang-Lai/SignNet.git
```

2. **Install the requirements**, see [requirement section](#requirement) for instruction.

3. **Download pretrained model**
   - [BaiduNetDisk](https://pan.baidu.com/s/1KenvrAqAWNd1d7zJ4wuA2Q), Code:  40hi
   - [Google Drive](https://drive.google.com/drive/folders/1nvcGXNynlo0k259ACTKcOLiGnMVcgyyF?usp=sharing)
4. **Unzip pretrained model and put it into `signnet/pretrained` directory**
5. **Run the live demo**

```shell
python live_demo.py
```

## Requirement

- OpenCV
- QT
- Python3

**Install the Python requirement using the following command.**

```shell
cd signnet
conda env create -f env.yaml
```

## Gestures

The gestures for the sign alphabets we used are a little **differnt** from what it is shown in the header image. Our version can be shown in the following pictures.

![alphabets](imgs/alphabets.png)

The total alphabets our model can recognitize includes **26 letters, space and delete** (control operator).

## Evaluation

To be done

## Datasets

The training and test dataset are collected on our own. 

- The training set contains 29 videos (26 letters, space, delete and nothing) and each lasts 30s. 
- The test set also contains 29 videos, but each  only last 10s.

We will open our dataset soon.

> We also made a attempt to train a model with [ASL Alphabet](https://www.kaggle.com/grassknoted/asl-alphabet) dataset, and test using  [ASL Alphabet Test](https://www.kaggle.com/danrasband/asl-alphabet-test/home) , but the result is **poor**.

## To Do

- [x] Record video demo
- [ ] Upload training and test data
- [ ] Finish report
- [ ] Integrate camera and text windows into a single one
- [x] Complete testing code (test.py)
- [ ] Complete demo with static image (demo.py)
- [ ] Tutorial for training

## Reference

This project is inspired by these related projects.

- [loicmarie](https://github.com/loicmarie)/**[sign-language-alphabet-recognizer](https://github.com/loicmarie/sign-language-alphabet-recognizer)**

