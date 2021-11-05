# @Author   : 919106840638肖林航
# @time     : 2021/10/07 上午10:50
# @Software : PyCharm
import asyncio
import csv
from time import ctime

import cv2
import joblib
from scipy.cluster.vq import *
from sklearn import preprocessing
from pylab import *
from colorFeature import ColorDescriptor

# 根据颜色特征搜索相关图像
def searchByColor(search_image):
	# 颜色特征描述符初始化
	cd = ColorDescriptor((8, 12, 3))
	# 加载图片
	im = cv2.imread(search_image)
	# 计算获取被检索图像的颜色特征描述符
	features = list(np.array(cd.describe(im)))
	# 获得检索结果
	results= find_img(features)
	path_list=[]
	# 获取结果的图片地址
	for distance,imageID in results:
		path_list.append(imageID)
	# 返回检索结果
	return path_list


def find_img(queryFeatures, limit = 10):
	# 初始化我们的结果字典
	results = {}

	# 打开索引文件进行读取
	with open("index.csv") as f:
		# 初始化 CSV 阅读器
		reader = csv.reader(f)
		# 遍历索引文件中的每一行
		for row in reader:
			# 解析出图像 ID 和特征，然后计算
			# 计算被检索的图像特征与索引文件中的每行的特征的距离
			features = [float(x) for x in row[1:]]
			d = distance(features, queryFeatures)
			print(d)
			# 将计算结果存储
			results[row[0]] = d

		# 关闭阅读器
		f.close()



	results = sorted([(v, k) for (k, v) in results.items()])

	# 返回结果
	# print(len(results))
	return results[:limit]



def distance(histA, histB, eps = 1e-10):
	# 计算卡方距离
	d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
		for (a, b) in zip(histA, histB)])

	# 返回卡方距离
	return d



# 根据sift特征搜索相关图像
def searchBySift(search_image):
	# 加载分类器、类名、缩放器、簇数和词汇
	im_features, image_paths, idf, numWords, voc = joblib.load("bow.pkl")

	# 创建特征提取和关键点检测器对象
	sift_det = cv2.SIFT_create()

	# 列出所有描述符的存储位置
	des_list = []

	im = cv2.imread(search_image)
	gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
	kp, des = sift_det.detectAndCompute(gray, None)

	des_list.append((search_image, des))

	# 将所有描述符垂直堆叠在一个 numpy 数组中
	descriptors = des_list[0][1]

	test_features = np.zeros((1, numWords), "float32")
	words, distance = vq(descriptors, voc)
	for w in words:
		test_features[0][w] += 1

	# 执行 Tf-Idf 矢量化和 L2 归一化
	test_features = test_features * idf
	test_features = preprocessing.normalize(test_features, norm='l2')


	score = np.dot(test_features, im_features.T)
	rank_ID = np.argsort(-score)

	result_list=[]

	for i, ID in enumerate(rank_ID[0][0:10]):
		result_list.append(image_paths[ID])

	return result_list






