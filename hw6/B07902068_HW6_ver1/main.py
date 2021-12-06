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
	connectivity = np.zeros((h, w), dtype = int)
	img = padding(img)
	for r in range(1, h+1):
		for c in range(1, w+1):
			if img[r, c] == 255:
				connectivity[r-1, c-1] = yokoi_number(r, c, img)
	return connectivity

def output(matrix):
	h, w = matrix.shape
	for r in range(h):
		for c in range(w):
			if (matrix[r, c] != 0):
				print('%d' % matrix[r,c], end = "")
			else:
				print(' ', end = "")
		print()

	return

if __name__ == '__main__':
	img_path = 'lena.bmp'
	img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
	img_binary = binarize(img)
	img_sample = down_sample(img_binary)
	connectivity = cal_connectivity(img_sample)
	output(connectivity)