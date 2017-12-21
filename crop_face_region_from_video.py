import cv2
import os


def crop_face_region_from_video(video_path, save_image_path):
    cap = cv2.VideoCapture()
    if not cap.open(video_path):
        print("can not open the video: ", video_path)
        exit(-1)
    frame_num = 0
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    while True:
        ret, image = cap.read()
        frame_num += 1
        if ret == 0:
                break
        if frame_num % 5 == 0:
            # if frame_num < 400:
            #     continue
            # if frame_num > 2000:
            #     break
            label_path = "./res/{:05d}.txt".format(frame_num)
            if not os.path.exists(label_path):
                continue
            boxes = open(label_path, "r")
            for box in boxes.readlines():
                xmin,ymin,xmax,ymax = box.strip().split(',')
                xmin,ymin,xmax,ymax = float(xmin)*frame_width, float(ymin)*frame_height, \
                                      float(xmax)*frame_width, float(ymax)*frame_height
                xmin,ymin,xmax,ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                face_region = image[ymin:ymax, xmin:xmax, :]
                cv2.imwrite("{}/{:05d}.jpg".format(save_image_path, frame_num), face_region)


def main():
    video_path = r'D:/videos/test4.mp4'
    save_image_path = "./images/yang"
    if not os.path.exists(save_image_path):
        os.mkdir(save_image_path)
    crop_face_region_from_video(video_path, save_image_path)


if __name__ == '__main__':
    main()
