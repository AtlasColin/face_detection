import numpy as np
import sys
import tensorflow as tf
import time
import cv2
from utils import label_map_util
from utils import visualization_utils_color as vis_util

sys.path.append("..")


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './protos/face_label_map.pbtxt'

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    im_width, im_height = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

video_path = r'D:/videos/test4.mp4'
save_video_path = r'D:/videos/test_out.avi'
cap = cv2.VideoCapture()
if not cap.open(video_path):
    print("can not open the video: ", video_path)
    exit(-1)
out = cv2.VideoWriter(save_video_path, 0, 25.0, (1280, 720))


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

with detection_graph.as_default():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=detection_graph, config=config) as sess:
        # frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_num = 0
        while True:
            ret, image = cap.read()
            frame_num += 1
            if ret == 0:
                    break
            if frame_num % 5 == 0:
                # if frame_num < 2000:
                #     continue
                # if frame_num > 2000:
                #     break
                image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # boxes = tf.multiply(boxes, 1.2)
                # Actual detection.
                start_time = time.time()
                (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                elapsed_time = time.time() - start_time
                print('inference time cost: {}'.format(elapsed_time))
                # print(boxes.shape, boxes)
                # print(scores.shape,scores)
                # print(classes.shape,classes)
                # print(num_detections)
                # # Visualization of the results of a detection.
                # vis_util.visualize_boxes_and_labels_on_image_array(
                #         # image_np,
                #         image,
                #         np.squeeze(boxes),
                #         np.squeeze(classes).astype(np.int32),
                #         np.squeeze(scores),
                #         category_index,
                #         use_normalized_coordinates=True,
                #         line_thickness=4)
                filepath = "./res/{:05d}.txt".format(frame_num)
                vis_util.write_boxes_to_file(
                        filepath,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores))

                # out.write(image)
                # cv2.namedWindow("image", flags=cv2.WINDOW_AUTOSIZE)
                # cv2.resizeWindow("image", 320, 288)
                # cv2.imshow("image", image)
                # cv2.waitKey(1)
        cap.release()
        out.release()
