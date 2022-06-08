from math import ceil
from random import choice
from tqdm import tqdm
import cv2
import os
import numpy as np
# from PIL import Image
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
# 改变图片大小
def adjustImage(path, x1, x2, y1, y2, rows, cols):
    object = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    x1 = round(x1)
    x2 = ceil(x2)
    y1 = round(y1)
    y2 = ceil(y2)
    region = object[y1:y2, x1:x2]

    # 右侧距离
    right_distance = cols - x2
    # 左侧距离
    left_distance = x1
    # 上方距离
    top_distance = y1
    # 下方距离
    bottom_distance = rows - y2
    if bottom_distance >= (y2 - y1) or top_distance >= (y2 - y1):
        return region
    if left_distance >= (x2 - x1) or right_distance >= (x2 - x1):
        return region
    a = 0
    number = 1
    # 左边面积
    left = round(left_distance) * rows
    a = left
    # 右面积
    right = ceil(right_distance) * rows
    if right > a:
        a = right
        number = 2
    # 下面积
    bottom = cols * ceil(bottom_distance)
    if bottom > a:
        a = bottom
        number = 3
    # 上面积
    top = round(top_distance) * cols
    if top > a:
        number = 4
    if number == 1:
        ratio = int((x2 - x1) / left_distance)
        h = int((y2 - y1) / ratio)
        if h == 0:
            return region
        region = cv2.resize(region, (left_distance, h))

    elif number == 2:
        ratio = int((x2 - x1) / right_distance)
        h = int((y2 - y1) / ratio)
        if h == 0:
            return region
        region = cv2.resize(region, (right_distance, h))

    elif number == 3:
        ratio = int((y2 - y1) / bottom_distance)
        w = int((x2 - x1) / ratio)
        if w == 0:
            return region
        region = cv2.resize(region, (w, bottom_distance))
    else:

        ratio = int((y2 - y1) / top_distance)
        w = int((x2 - x1) / ratio)
        if w == 0:
            return region
        region = cv2.resize(region, (w, top_distance))

    return region


# 插入实例
def addObject(img, path, x1, x2, y1, y2, rows, cols):
    # 三通道转四通道
    b_channel, g_channel, r_channel = cv2.split(img)  # 剥离jpg图像通道
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255  # 创建Alpha通道
    alpha_channel = alpha_channel.astype(np.uint8)
    img_new = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))  # 融合通道

    # 右侧距离
    right_distance = cols - ceil(x2)
    # 左侧距离
    left_distance = ceil(x1)
    # 上方距离
    top_distance = ceil(y1)
    # 下方距离
    bottom_distance = rows - ceil(y2)

    region = adjustImage(path, x1, x2, y1, y2, rows, cols)
    # plt.imshow(region)
    # plt.show()
    yy1 = 0
    yy2 = region.shape[0]
    xx1 = 0
    xx2 = region.shape[1]

    x_l = xx2 - xx1
    x_r = cols - x_l
    y_t = yy2 - yy1
    y_b = rows - y_t
    alpha_png = region[yy1:yy2, xx1:xx2, 3] / 255
    alpha_jpg = 1 - alpha_png

    # 右
    if right_distance >= (xx2 - xx1):
        for c in range(0, 3):
            img_new[0:y_t, x_r:cols, c] = (
                    (alpha_jpg * img_new[0:y_t, x_r:cols, c]) + (alpha_png * region[yy1:yy2, xx1:xx2, c]))
        return img_new

    # 左
    elif left_distance >= (xx2 - xx1):
        for c in range(0, 3):
            img_new[0:y_t, 0:x_l, c] = (
                    (alpha_jpg * img_new[0:y_t, 0:x_l, c]) + (alpha_png * region[yy1:yy2, xx1:xx2, c]))
        return img_new

    # 上
    elif top_distance >= (yy2 - yy1):
        for c in range(0, 3):
            img_new[0:y_t, 0:x_l, c] = (
                    (alpha_jpg * img_new[0:y_t, 0:x_l, c]) + (alpha_png * region[yy1:yy2, xx1:xx2, c]))
        return img_new

    # 下角
    elif bottom_distance >= (yy2 - yy1):
        for c in range(0, 3):
            img_new[y_b:rows, 0:x_l, c] = (
                    (alpha_jpg * img_new[y_b:rows, 0:x_l, c]) + (alpha_png * region[yy1:yy2, xx1:xx2, c]))
        return img_new


# 改变亮度
def updateBrightness(img, img_3, rows, cols):
    # 加载图片 读取彩色图像归一化且转换为浮点型\

    process_image = img.astype(np.float32) / 255.0
    # 颜色空间转换 BGR转为HLS
    hlsImg = cv2.cvtColor(process_image, cv2.COLOR_BGR2HLS)
    # 调整亮度
    hlsImg[:, :, 1] = (1.0 + 7 / float(10)) * hlsImg[:, :, 1]
    hlsImg[:, :, 1][hlsImg[:, :, 1] > 1] = 1
    lsImg = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR)

    # newimg=img_3
    lsImg = (lsImg * 255).astype(np.uint8)
    # for h in range(rows):
    #     for w in range(cols):
    #         if img[h][w][3]==0:
    #             continue
    #         else:
    #             newimg[h][w]=lsImg[h][w]
    # return newimg

    return lsImg


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
        if m in range(round(x1), round(x2)) and n in range(round(y1), round(y2)):
            continue
        img_noise[n, m, :] = 255
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

    backgroundpath = "E:/extract/background/"
    objectpath = "E:/extract/object/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    namelist = os.listdir(coco_path['image_path'])
    i = 0
    for name in tqdm(namelist):

        if i == 1654:
            print(name)

        i = i + 1
        imageNubmer = name[0:12]
        imgid = int(name[6:12])

        # annotation的id
        annsid = ins_coco.getAnnIds(imgid)
        # annotation信息
        outline = ins_coco.loadAnns(annsid)
        # 类别id
        catId = outline[0]["category_id"]
        image = cv2.imread(os.path.join(coco_root, name))
        img_noise = image

        # boundingbox位置
        x, y, w, h = outline[0]["bbox"]
        x1, y1, x2, y2 = x, y, int(x + w), int(y + h)

        rows, cols, chn = img_noise.shape

        if (int(x2) - int(x1)) == cols and int(int(y2) - int(y1)) == rows:
            continue
        # m = ins_coco.annToMask(outline[0])
        # 实例路径
        path = os.path.join(objectpath, str(catId) + "/", str(imageNubmer) + ".png")
        c = addObject(image, path, x1, x2, y1, y2, rows, cols)

        cv2.imwrite(os.path.join("E:/data/repeat/", name), c)
        # cv2.imwrite(os.path.join("E:\data\synPhoto", name), c)
        # mask = np.array(m)
        # # 复制并扩充维度与原图片相等, 用于后续计算
        # mask_three = np.expand_dims(mask, 2).repeat(3, axis=2)
        # jpg_img = np.array(img_noise)
        # # 如果mask矩阵中元素大于0, 则置为原图的像素信息, 否则置为黑色
        # result = np.where(mask_three > 0, jpg_img, 0)
        # backgroundResult = np.where(mask_three < 1, jpg_img, 0)
        # # 如果mask矩阵中元素大于0, 则置为白色, 否则为黑色, 用于生成第4通道图像信息
        # # alpha保留目标检测物
        # # backgroundalpha为保留背景
        # alpha = np.where(mask > 0, 255, 0)
        # backgroundalpha = np.where(mask < 1, 255, 0)
        # alpha = alpha.astype(np.uint8)  # 转换格式, 防止拼接时由于数据格式不匹配报错
        # backgroundalpha = backgroundalpha.astype(np.uint8)  # 转换格式, 防止拼接时由于数据格式不匹配报错
        # b, g, r = cv2.split(result)  # 分离三通道, 准备衔接上第4通道
        # bb, bg, br = cv2.split(backgroundResult)
        # brgba = [bb, bg, br, backgroundalpha]
        # rgba = [b, g, r, alpha]
        # # 将三通道图片转化为四通道(背景透明)的图片
        # dst = cv2.merge(rgba, 4)  # 拼接4个通道
        # bdst = cv2.merge(brgba, 4)
        #
        # c = updateBrightness(img_noise,image,rows,cols)
        # # c2=c[: , : , : : -1]
        # cv2.imwrite(os.path.join("E:/lightnessAll", name), c)
        # dst = dst[int(outline[0]['bbox'][1]):int(outline[0]['bbox'][1] + outline[0]['bbox'][3]),
        #       int(outline[0]['bbox'][0]):int(outline[0]['bbox'][0] + outline[0]['bbox'][2])]
        # 储存提取出来的图像和背景
        # bPath =os.path.join(backgroundpath,str(catId))
        # oPath =os.path.join(objectpath,str(catId))
        # if not os.path.exists(bPath):
        #     os.makedirs(bPath)
        # if not os.path.exists(oPath):
        #     os.makedirs(oPath)
        #
        # cv2.imwrite(os.path.join(oPath,imageNubmer+".png"), dst)
        # cv2.imwrite(os.path.join(bPath,imageNubmer+".png"), bdst)

        # # ins_coco.showAnns(outline)

    # addNoiseToBack(rows,cols,x1,x2,y1,y2,img_noise,name)
    # addNoidsToAll(rows, cols, img_noise, name)
    # addNoidsToObject(x1,x2,y1,y2,img_noise,save_path,name)
