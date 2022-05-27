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


# 改变亮度
def updateBrightness(img,img_3,rows,cols):
    # 加载图片 读取彩色图像归一化且转换为浮点型


    process_image = img.astype(np.float32) / 255.0
    # 颜色空间转换 BGR转为HLS
    hlsImg = cv2.cvtColor(process_image, cv2.COLOR_BGR2HLS)
    # 调整亮度
    hlsImg[:, :, 1] = (1.0 + 7/ float(10)) * hlsImg[:, :, 1]
    hlsImg[:, :, 1][hlsImg[:, :, 1] > 1] = 1
    lsImg = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR)

    # newimg=img_3
    lsImg= (lsImg * 255).astype(np.uint8)
    # for h in range(rows):
    #     for w in range(cols):
    #         if img[h][w][3]==0:
    #             continue
    #         else:
    #             newimg[h][w]=lsImg[h][w]
    # return newimg

    return  lsImg


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
#插入实例

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
    for name in tqdm(namelist):
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
        x, y, w, h = outline[0]["bbox"]
        x1, y1, x2, y2 = x, y, int(x + w), int(y + h)
        rows, cols, chn = img_noise.shape
        m = ins_coco.annToMask(outline[0])
        mask = np.array(m)
        # 复制并扩充维度与原图片相等, 用于后续计算
        mask_three = np.expand_dims(mask, 2).repeat(3, axis=2)
        jpg_img = np.array(img_noise)
        # 如果mask矩阵中元素大于0, 则置为原图的像素信息, 否则置为黑色
        result = np.where(mask_three > 0, jpg_img, 0)
        backgroundResult = np.where(mask_three < 1, jpg_img, 0)
        # 如果mask矩阵中元素大于0, 则置为白色, 否则为黑色, 用于生成第4通道图像信息
        # alpha保留目标检测物
        # backgroundalpha为保留背景
        alpha = np.where(mask > 0, 255, 0)
        backgroundalpha = np.where(mask < 1, 255, 0)
        alpha = alpha.astype(np.uint8)  # 转换格式, 防止拼接时由于数据格式不匹配报错
        backgroundalpha = backgroundalpha.astype(np.uint8)  # 转换格式, 防止拼接时由于数据格式不匹配报错
        b, g, r = cv2.split(result)  # 分离三通道, 准备衔接上第4通道
        bb, bg, br = cv2.split(backgroundResult)
        brgba = [bb, bg, br, backgroundalpha]
        rgba = [b, g, r, alpha]
        # 将三通道图片转化为四通道(背景透明)的图片
        dst = cv2.merge(rgba, 4)  # 拼接4个通道
        bdst = cv2.merge(brgba, 4)

        c = updateBrightness(img_noise,image,rows,cols)
        # c2=c[: , : , : : -1]
        cv2.imwrite(os.path.join("E:/lightnessAll", name), c)
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
