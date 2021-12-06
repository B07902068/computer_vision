import cv2
import numpy as np

def binarize(img):#img is returned by cv2.imreads
	new = np.empty(img.shape, dtype = np.uint8)
	row = img.shape[0]
	col = img.shape[1]
	
	for r in range(row):
		for c in range(col):
			if (img[r][c] >= 128):
				new[r][c] = 255
			else:
				new[r][c] = 0
	#cv2.imwrite('binary.bmp', new)
	#cv2.imshow('binary', new)
	return new

def down_sample(img):
	h, w = img.shape
	img_new = np.empty((h // 8, w // 8), dtype = np.uint8)

	for r in range(0, h // 8):
		for c in range(0, w // 8):
			img_new[r, c] = img[8*r, 8*c]
	return img_new

def padding(img):
	h, w = img.shape
	new = np.zeros((h + 2, w + 2), dtype = np.uint8)
	new[1:h+1, 1:w+1] = img
	return new


def h(b, c, d, e):
	if b == c and (d != b or e != b):
		return 'q'
	if b == c and (d == b and e == b):
		return 'r'
	if b != c:
		return 's'

def yokoi_number(r, c, img):
	a1 = h(img[r, c], img[r, c+1], img[r-1, c+1], img[r-1, c])
	a2 = h(img[r, c], img[r-1, c], img[r-1, c-1], img[r, c-1])
	a3 = h(img[r, c], img[r, c-1], img[r+1, c-1], img[r+1, c])
	a4 = h(img[r, c], img[r+1, c], img[r+1, c+1], img[r, c+1])

	if a1 == 'r' and a2 == 'r' and a3 == 'r' and a4 == 'r':
		return 5
	count = 0
	if a1 == 'q':
		count += 1
	if a2 == 'q':
		count += 1
	if a3 == 'q':
		count += 1
	if a4 == 'q':
		count += 1
	return count

	return
def cal_connectivity(img):
	h, w = img.shape
	connectivity = np.zeros((h+2, w+2), dtype = int)
	img_pad = padding(img)
	for r in range(1, h+1):
		for c in range(1, w+1):
			if img_pad[r, c] == 255:
				connectivity[r, c] = yokoi_number(r, c, img_pad)
	return connectivity

def pair_label(r, c, connectivity): #q = 0, p = 1, m = 1
	if connectivity[r, c] != 1:
		return 0
	sum = 0
	if connectivity[r, c+1] == 1: #x1
		sum += 1
	if connectivity[r-1, c] == 1: #x2
		sum += 1
	if connectivity[r, c-1] == 1: #x3
		sum += 1
	if connectivity[r+1, c] == 1: #x4
		sum += 1
	if sum < 1:
		return 0
	return 1


def mark(img, connectivity): #q = 0, p = 1, connectivity.shape = (h + 2, w + 2)
	h, w = img.shape
	mark_map = np.zeros((h+2, w+2), dtype = int)
	for r in range(1, h+1):
		for c in range(1, w+1):
			mark_map[r, c] = pair_label(r, c, connectivity)
	return mark_map

def thinning(img, mark_map): #mark_map.shape = (h+2, w+2)
	h, w = img.shape
	img_pad = padding(img)

	for r in range(1, h+1):
		for c in range(1, w+1):
			if mark_map[r, c] == 1 and yokoi_number(r, c, img_pad) == 1:
				img_pad[r, c] = 0
	img = img_pad[1:h+1, 1:w+1]
	return img


if __name__ == '__main__':
	img_path = 'lena.bmp'
	img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
	img = binarize(img)
	img = down_sample(img)
	for i in range(7):
		connectivity = cal_connectivity(img)
		mark_map = mark(img, connectivity)
		img = thinning(img, mark_map)
	cv2.imwrite("result.bmp", img)
