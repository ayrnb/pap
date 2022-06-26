import os
import re

import cv2

labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
          'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
          'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
          'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
          'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
          'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
          'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
          'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
          'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
          'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
          'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
          'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
          'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
          'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]
    # round函数确定(xmin, ymin, xmax, ymax)的小数位数
    x = round(x * dw, 6)
    w = round(w * dw, 6)
    y = round(y * dh, 6)
    h = round(h * dh, 6)
    return (x, y, w, h)


if __name__ == "__main__":
    path = "E:/data/diff/"
    result_path = "E:/data/output/"
    file = os.listdir(path)
    for f in file:
        diff_path = os.path.join(path, f)
        if os.path.isfile(diff_path):
            continue
        content_path = os.path.join(diff_path + "/", "difflist.txt")
        result = os.path.join(result_path, f + ".txt")
        diff_file = open(content_path, "r")
        diff_list = diff_file.readlines()
        result_list = open(result, "r").readlines()
        # E: / data / lightnessBackground / 000000000034.jpg: Predicted in 1688.531000 milli - seconds.
        # zebra: 100 % (left_x:  -10   top_y:  -52   width:  458   height:  521)
        #  Enter Image Path:  Detection layer: 139 - type = 28
        #  Detection layer: 150 - type = 28
        #  Detection layer: 161 - type = 28

        diff_list = [x.strip() for x in diff_list]
        filename = ""
        for content in result_list:
            flag = False
            if filename in diff_list:
                flag = True
            if content[-15:-1] == "milli-seconds.":
                list = content.split("/")
                filename = list[3][0:12]
                continue
            if content == "Enter Image Path:  Detection layer: 139 - type = 28 \n":
                continue
            if content == " Detection layer: 161 - type = 28 \n":
                if flag:
                    f_txt.close()
                continue
            if content == " Detection layer: 150 - type = 28 \n":
                continue
            if flag:
                f_txt = open(os.path.join(diff_path, filename + ".txt"), "a")
                str = content.split("(")
                class_label = str[0].split(":")[0]
                # left_x:  -10   top_y:  -52   width:  458   height:  521
                box = str[1].strip(")\n").split(":")
                # qq = box[1].strip(" ").split(" ")[0]
                # ww = box[2].strip(" ").split(" ")[0]
                # ee = box[3].strip(" ").split(" ")[0]
                # dd = box[4].strip(" ").split(" ")[0]
                bbox = [int(box[1].strip(" ").split(" ")[0]), int(box[2].strip(" ").split(" ")[0]), int(box[3].strip(" ").split(" ")[0]),
                        int(box[4].strip(" ").split(" ")[0])]
                image = cv2.imread("E:/data/" + f + "/" + filename + ".jpg")
                cv2.imwrite(os.path.join(diff_path, filename + ".jpg"), image)
                rows, cols, chn = image.shape
                bbbox = convert((int(cols), int(rows)), bbox)
                f_txt.write("%s %s %s %s %s\n" % (labels.index(class_label), bbbox[0], bbbox[1], bbbox[2], bbbox[3]))
            else:
                continue
