import cv2
import numpy as np

def binarize(img):#img is returned by cv2.imreads
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
	return new

def inside(r, c, h, w):
	if r < 0 or c < 0 or r >= h or c >= w:
		return False
	return True
def dilation(img, mask):
	h, w = img.shape[:2]
	new = np.zeros(img.shape, dtype = np.uint8)
	for r in range(h):
		for c in range(w):
			if img[r, c, 0] == 255:
				for k in mask:
					if inside(r + k[0], c + k[1], h, w):
						new[r + k[0], c + k[1]] = np.full(3, 255)
	return new

def erosion(img, mask):
	h, w = img.shape[:2]
	new = np.full(img.shape, 255, dtype = np.uint8)
	for r in range(h):
		for c in range(w):
			for k in mask:
				if (not inside(r + k[0], c + k[1], h, w)) or img[r + k[0], c + k[1], 0] == 0:
					new[r, c] = np.zeros(3, dtype = np.uint8)
	return new

def complement(img):
	#new = -img + 255
	h, w = img.shape[:2]
	new = np.zeros(img.shape, dtype = np.uint8)
	for r in range(h):
		for c in range(w):
			if img[r, c, 0] == 0:
				new[r, c] = np.full(3, 255)
	return new

def intersection(img1, img2):
	h, w = img1.shape[:2]
	new = np.zeros(img1.shape, dtype = np.uint8)
	for r in range(h):
		for c in range(w):
			if img1[r, c, 0] == 255 and img2[r, c, 0] == 255:
				new[r, c] = np.full(3, 255)
	return new

def hit_and_miss(img):
	J = [(0, -1), (0, 0), (1, 0)]
	K = [(-1, 0), (-1, 1), (0, 1)]

	img_J = erosion(img, J)
	img_K = erosion(complement(img), K)
	new = intersection(img_J, img_K)
	return new

if __name__ == '__main__':
	img_path = 'lena.bmp'
	img = cv2.imread(img_path)
	img_binary = binarize(img)
	mask = [(-2, -1), (-2, 0), (-2, 1), 
			(-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
			(0, -2), (0, -1), (0, 0), (0, 1), (0, 2),
			(1, -2), (1, -1), (1, 0), (1, 1), (1, 2),
			(2, -1), (2, 0), (2, 1)]
	'''
	0 1 1 1 0
	1 1 1 1 1
	1 1 1 1 1
	1 1 1 1 1
	0 1 1 1 0
	'''
	#(a)
	img_dil = dilation(img_binary, mask)
	cv2.imwrite('dilation.bmp', img_dil)
	
	#(b)
	img_ero = erosion(img_binary, mask)
	cv2.imwrite('erosion.bmp', img_ero)
	
	#(c)
	img_open = dilation(img_ero, mask)
	cv2.imwrite('opening.bmp', img_open)
	
	#(d)
	img_close = erosion(img_dil, mask)
	cv2.imwrite('closing.bmp', img_close)
	
	#(e)
	img_HnM = hit_and_miss(img_binary)
	cv2.imwrite('hit_and_miss.bmp', img_HnM)