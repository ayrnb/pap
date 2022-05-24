from math import ceil
from random import choice
from tqdm import tqdm
import cv2
import os
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from skimage.io.tests.test_mpl_imshow import plt


# 定义需要提取的类别
# labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
#           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
#           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
#           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
#           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
#           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
#           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
#           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
#           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
#           'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
#           'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def get_coco_roi(ins_coco, key_coco, label, image_path, save_path):
    try:
        # 采用getCatIds函数获取"person"类别对应的ID
        ins_ids = ins_coco.getCatIds(label)[0]
        print("%s 对应的类别ID: %d" % (label, ins_ids))
    except:
        print("[ERROR] 请输入正确的类别名称")
        return

    # 获取某一类的所有图片集合, 比如获取包含dog的所有图片
    imgIds = ins_coco.getImgIds(catIds=ins_ids)
    print("包含 {} 的图片共有 {} 张".format(label, len(imgIds)))
    # 获取某一类的所有图片集合, 比如获取包含dog的所有图片
    imgIds = ins_coco.getImgIds(catIds=ins_ids)
    print("包含 {} 的图片共有 {} 张".format(label, len(imgIds)))

    for img in imgIds:
        try:
            img_info = ins_coco.loadImgs(img)[0]
        except:
            continue

        annIds = ins_coco.getAnnIds(imgIds=img_info['id'])
        imgpath = os.path.join(image_path, img_info['file_name'])
        jpg_img = cv2.imread(imgpath, 1)
        if jpg_img is None:
            continue

        for ann in annIds:
            outline = ins_coco.loadAnns(ann)[0]

            # 只提取类别对应的标注信息
            if outline['category_id'] != ins_ids:
                continue

            # 对人同时使用关键点判断, 如果关键点中含有0的数量比较多, 代表这个人是不完整或姿态不好的
            if outline['category_id'] == 1:
                key_outline = key_coco.loadAnns(ann)[0]
                if key_outline['keypoints'].count(0) >= 10:
                    continue

            # 将轮廓信息转为Mask信息并转为numpy格式
            mask = ins_coco.annToMask(outline)
            mask = np.array(mask)

            # 复制并扩充维度与原图片相等, 用于后续计算
            mask_three = np.expand_dims(mask, 2).repeat(3, axis=2)

            jpg_img = np.array(jpg_img)

            # 如果mask矩阵中元素大于0, 则置为原图的像素信息, 否则置为黑色
            result = np.where(mask_three > 0, jpg_img, 0)

            # 如果mask矩阵中元素大于0, 则置为白色, 否则为黑色, 用于生成第4通道图像信息
            alpha = np.where(mask > 0, 255, 0)
            alpha = alpha.astype(np.uint8)  # 转换格式, 防止拼接时由于数据格式不匹配报错

            b, g, r = cv2.split(result)  # 分离三通道, 准备衔接上第4通道
            rgba = [b, g, r, alpha]  # 将三通道图片转化为四通道(背景透明)的图片
            dst = cv2.merge(rgba, 4)  # 拼接4个通道
            dst = dst[int(outline['bbox'][1]):int(outline['bbox'][1] + outline['bbox'][3]),
                  int(outline['bbox'][0]):int(outline['bbox'][0] + outline['bbox'][2])]

            name, shuifx = os.path.splitext(img_info['file_name'])
            imPath = os.path.join(save_path, name + "_%05d" % (int(annIds.index(ann))) + ".png")
            print("[INFO] 当前进度: %d /%d" % (imgIds.index(img), len(imgIds)))
            # cv2.imwrite(imPath, dst)
            cv2.imencode('.png', dst)[1].tofile(imPath)  # 保存中文路径的方法


# 给检测物添加噪点
def addNoidsToObject(x1, x2, y1, y2, image, path, name):
    num = int((x2 - x1) * (y2 - y1) / 20)
    for i in range(num):
        n = np.random.randint(int(x1), int(x2))
        m = np.random.randint(int(y1), int(y2))
        img_noise[m, n, :] = 255
    cv2.imwrite(os.path.join(path, name), image)


# 给整张图片添加噪点
def addNoidsToAll(rows, cols, image, name):
    num = int(rows * cols / 20)
    for i in range(num):
        n = np.random.randint(0, rows)
        m = np.random.randint(0, cols)
        img_noise[n, m, :] = 255
    cv2.imwrite(os.path.join("E:/all", name), image)
    # plt.imshow(image)
    # plt.show()

# 给背景添加噪点
def addNoiseToBack(rows, cols, x1, x2, y1, y2, image, name):
    num = int(rows * cols / 20)
    for i in range(num):
        n = np.random.randint(0, rows)
        m = np.random.randint(0, cols)
        if m in range(round(x1),round(x2)) and n in range(round(y1),round(y2)):
            continue
        img_noise[n, m, :] = 255

    # num = int(((rows * cols) - ((x2 - x1) * (y2 - y1))) / 20)
    # for i in range(num):
    #     nl = [np.random.randint(0, ceil(y1)+1), np.random.randint(round(y2)-1, rows)]
    #     nc = choice(nl)
    #     ml = np.random.randint(0, cols)
    #     arr = [nc, ml]
    #     n2 =  np.random.randint(int(y1),int(y2))
    #     mll = [np.random.randint(0, ceil(x1)+1), np.random.randint(round(x2)-1, cols)]
    #     m2 = choice(mll)
    #     arr2 = [n2, m2]
    #     point = [arr, arr2]
    #     p=choice(point)
    #     n=p[0]
    #     m=p[1]
    #     img_noise[n, m, :] = 255
    cv2.imwrite(os.path.join("E:/background", name), image)


if __name__ == "__main__":
    # 定义Coco数据集根目录
    coco_root = r"F:/newdata/"

    json_root = r"F:/xlxz/data/annotations_trainval2017/annotations"

    coco_path = {
        'image_path': os.path.join(coco_root),
        'instances_json_path': json_root + r"/instances_train2017.json",
        'keypoints_json_path': json_root + r"/person_keypoints_train2017.json"
    }
    ins_coco = COCO(coco_path['instances_json_path'])
    # key_coco = COCO(coco_path['keypoints_json_path'])
    save_path = "E:/noise"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    namelist = os.listdir(coco_path['image_path'])
    for name in tqdm(namelist):
        imgid = int(name[6:12])
        annsid = ins_coco.getAnnIds(imgid)
        outline = ins_coco.loadAnns(annsid)
        image = cv2.imread(os.path.join(coco_root, name))
        img_noise = image
        x, y, w, h = outline[0]["bbox"]
        x1, y1, x2, y2 = x, y, int(x + w), int(y + h)
        rows, cols, chn = img_noise.shape
        addNoiseToBack(rows,cols,x1,x2,y1,y2,img_noise,name)
        # addNoidsToAll(rows, cols, img_noise, name)
        # addNoidsToObject(x1,x2,y1,y2,img_noise,save_path,name)
