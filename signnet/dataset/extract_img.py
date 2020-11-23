import argparse
import os

import cv2


def save_image(name, num, image, output_dir):
    image_path = os.path.join(output_dir, str(name), '{}.jpg'.format(str(num)))
    cv2.imwrite(image_path, image)


def capture_video(file_path, output_dir, frame_interval):
    #  获得视频名称
    video_name = file_path.split('/')[-1].split('.')[0]

    #  创建每个视频的帧采样目录
    if not os.path.exists(os.path.join(output_dir, video_name)):
        os.mkdir(os.path.join(output_dir, video_name))

    vc = cv2.VideoCapture(file_path)

    count = 0  # 已保存的图片计数
    frame_count = 0  # 视频帧数记录

    while True:
        ret, frame = vc.read()
        if not ret:
            break
        #  每frame_interval帧存储一张图
        if frame_count % frame_interval == 0:
            save_image(video_name, count, frame, output_dir)
            print("num：" + str(count) + ", frame: " +
                  str(frame_count))
            count += 1
        frame_count += 1
        cv2.waitKey(1)

    vc.release()


def run(input_dir, output_dir, frame_interval):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        print('capturing ' + file)
        capture_video(os.path.join(input_dir, file), output_dir, frame_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--frame_interval', type=int, default=1)
    args = parser.parse_args()
    run(args.input, args.output, args.frame_interval)
