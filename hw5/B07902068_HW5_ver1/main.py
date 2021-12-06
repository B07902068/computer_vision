import cv2
import numpy as np

def inside(r, c, h, w):
	if r < 0 or c < 0 or r >= h or c >= w:
		return False
	return True
def dilation(img, mask):
	h, w = img.shape[:2]
	new = np.zeros(img.shape, dtype = np.uint8)
	for r in range(h):
		for c in range(w):
			temp = []
			for (z, t) in mask:
				if inside(r - z, c - t, h, w):
					temp.append(img[r - z, c - t, 0])
			new[r, c] = np.full(3, max(temp))
	return new

def erosion(img, mask):
	h, w = img.shape[:2]
	new = np.zeros(img.shape, dtype = np.uint8)
	for r in range(h):
		for c in range(w):
			temp = []
			for (z, t) in mask:
				if inside(r + z, c + t, h, w):
					temp.append(img[r + z, c + t, 0])
				else:
					temp.append(0)
			new[r, c] = np.full(3, min(temp))
	return new



if __name__ == '__main__':
	img_path = 'lena.bmp'
	img = cv2.imread(img_path)
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
	img_dil = dilation(img, mask)
	cv2.imwrite('dilation.bmp', img_dil)
	
	#(b)
	img_ero = erosion(img, mask)
	cv2.imwrite('erosion.bmp', img_ero)

	#(c)
	img_open = dilation(img_ero, mask)
	cv2.imwrite('opening.bmp', img_open)
	
	#(d)
	img_close = erosion(img_dil, mask)
	cv2.imwrite('closing.bmp', img_close)
	