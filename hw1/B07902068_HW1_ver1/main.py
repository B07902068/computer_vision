import cv2
import numpy as np

#img is the np array returned by cv2.imread
def up_down(img):
	new = np.empty(img.shape, dtype = np.uint8)
	row = img.shape[0]
	col = img.shape[1]
	for r in range(row):
		for c in range(col):
			new[r][c] = img[row - r - 1][c]
	cv2.imwrite('up_down.bmp', new)
	#cv2.imshow('up_down', new)
	return

def right_left(img):
	new = np.empty(img.shape, dtype = np.uint8)
	row = img.shape[0]
	col = img.shape[1]
	for r in range(row):
		for c in range(col):
			new[r][c] = img[r][col - c - 1]
	cv2.imwrite('right_left.bmp', new)
	#cv2.imshow('right_left', new)
	return

def diagonal(img):
	new = np.empty(img.shape, dtype = np.uint8)
	row = img.shape[0]
	col = img.shape[1]
	if row != col:
		print('cannot flip diagonally')
		return
	for r in range(row):
		for c in range(r, col):
			new[r][c] = img[c][r]
			new[c][r] = img[r][c]
	cv2.imwrite('diagonal.bmp', new)
	#cv2.imshow('diagonal', new)
	return
def rotate(img, angle): #reference: https://jennaweng0621.pixnet.net/blog/post/403495031-opencv-%E6%97%8B%E8%BD%89%E5%9C%96%E7%89%87%28rotate%29
	new = np.empty(img.shape, dtype = np.uint8)
	(h, w) = img.shape[:2]
	center = (h / 2, w / 2)
	M = cv2.getRotationMatrix2D(center, angle, 1)
	new = cv2.warpAffine(img, M, (w, h))
	cv2.imwrite('rotate.bmp', new)
	#cv2.imshow('rotate', new)
	return

def shrink(img, rate):
	dim = (int (img.shape[1] * rate), int (img.shape[0] * rate))
	new = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	cv2.imwrite('shrink.bmp', new)
	#cv2.imshow('shrink', new)
	return
def binarize(img):
	new = np.empty(img.shape, dtype = np.uint8)
	row = img.shape[0]
	col = img.shape[1]
	
	for r in range(row):
		for c in range(col):
			if (img[r][c][0] >= 128):
				new[r][c] = (255, 255, 255)
			else:
				new[r][c] = (0, 0, 0)
	cv2.imwrite('binary.bmp', new)
	#cv2.imshow('binary', new)
	return

if __name__ == '__main__':
	img_path = 'lena.bmp'
	img = cv2.imread(img_path)
	
	up_down(img)
	right_left(img)
	diagonal(img)
	rotate(img, -45)
	shrink(img, 0.5)
	binarize(img)


	'''cv2.imshow('My Image', img)

	# 按下任意鍵則關閉所有視窗
	cv2.waitKey(0)
	cv2.destroyAllWindows()#'''
	