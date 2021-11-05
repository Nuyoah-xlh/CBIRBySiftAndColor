# @Author   : 919106840638肖林航
# @time     : 2021/10/07 上午10:36
# @Software : PyCharm
import gc

import cv2
import numpy as np
import os
import joblib
from joblib.numpy_pickle_utils import xrange
from scipy.cluster.vq import *
from sklearn import preprocessing
from colorFeature import ColorDescriptor


# 获取训练集名称并将其存储在列表中
# 默认的数据集放在dataset/目录下
train_path = "dataset/"



# 获取dataset下的所有文件夹
training_folders = os.listdir(train_path)

print(training_folders)

# 获取所有图像的路径并将它们保存在列表中
image_paths = []  # 所有图片路径

# 遍历dataset目录下的所有文件夹
print("开始加载数据集......")
c=0
for folder in training_folders:
    # 对每个子文件夹，获取所有图片文件
    ls = os.listdir(train_path + "/" + folder)

    # 遍历该子文件夹下的所有图像
    for image_path in ls[:int(len(ls))]:
        image_path = os.path.join(train_path,folder+"/", image_path)
        print("正在获取  "+folder+"  文件夹下的图片："+image_path)
        image_paths += [image_path]
        c+=1
print(c)
del c
del training_folders
###################### 开始提取颜色特征描述符 #################################

print("准备提取颜色特征描述符......")

# 初始化颜色描述符
cd = ColorDescriptor((8, 12, 3))

# 打开索引文件进行写入,默认为index.csv
output = open("index.csv", "w")

print("开始提取图像颜色特征描述符......")
for image_path in image_paths:
    print("正在提取图像的颜色特征描述符："+image_path)
    # imageID唯一标注图片
    imageID = image_path[image_path.find("dataset"):]
    # print("imageID:"+imageID)
    image = cv2.imread(image_path)
    # 获取特征描述符，并转为list形式
    features = list(np.array(cd.describe(image)))
    # 将特征描述符写入索引文件
    # print(features)
    features = [str(f) for f in features]
    # print(features)
    output.write("%s,%s\n" % (imageID, ",".join(features)))



print("颜色特征描述符完毕")

################################# 颜色特征描述符完毕 #############################################

############################# 准备开始提取所有图片的sift特征   #####################################

print("准备提取所有图片的sift特征......")

# 设置聚类中心数
numWords = 64

# 创建特征提取和关键点检测器对象
sift_det=cv2.SIFT_create()

# 列出所有描述符的存储位置
des_list=[]  # 特征描述

print("开始提取图像sift特征描述符......")
s=0
for image_path in image_paths:
    print("正在提取图像的sift特征描述符："+image_path)
    # 读取图片文件
    img = cv2.imread(image_path)
    # 将图像转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 检测关键点并计算描述符
    kp, des = sift_det.detectAndCompute(gray, None)
    des_list.append((image_path, des))


# 将所有描述符垂直堆叠在一个 numpy 数组中
descriptors = des_list[0][1]
print('生成向量数组中......')
count=1
for image_path, descriptor in des_list[1:]:
    print(count)
    count+=1
    descriptors = np.vstack((descriptors, descriptor))

# 执行 k-means clustering
print ("开始 k-means 聚类: %d words, %d key points" %(numWords, descriptors.shape[0]))
voc, variance = kmeans(descriptors, numWords, 1)

# 计算特征的直方图
print("计算特征直方图中......")
im_features = np.zeros((len(image_paths), numWords), "float32")
# print(len(image_paths))
# for i in range(len(image_paths)):
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    print(i)
    for w in words:
        im_features[i][w] += 1

# 执行 Tf-Idf 矢量化
print("进行Tf-Idf 矢量化中......")
nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Perform L2 normalization
# 执行 L2 规范化
print("正在进行归一化处理......")
im_features = im_features*idf
im_features = preprocessing.normalize(im_features, norm='l2')

print('保存词袋模型文件中.......')
joblib.dump((im_features, image_paths, idf, numWords, voc), "bow.pkl", compress=3)

print("sift特征提取完毕！")

################################# sift特征提取结束 #############################################

print("特征描述符提取完毕！")





