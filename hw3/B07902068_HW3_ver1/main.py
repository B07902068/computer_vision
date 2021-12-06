import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_hist(img):
	count = np.zeros(256, dtype = int)
	(h, w) = img.shape[:2]
	for r in range(h):
		for c in range(w):
			count[img[r, c, 0]] += 1
	return count

def draw_hist(name, count):
	plt.clf()
	index = np.arange(0, 256)
	plt.bar(index, count)
	plt.xlabel('Intensity')
	plt.ylabel('Count')

	plt.title('Histogram of ' + name)
	plt.savefig(name + '_hist.png')
	#plt.show()
	return

def hist_equalization(img):
	h, w = img.shape[:2]
	n = h * w
	count = calculate_hist(img)
	cumulation = np.zeros(256, dtype = int)
	s = np.zeros(256, dtype = int)
	img_new = np.empty(img.shape, dtype = np.uint8)

	

	cumulation[0] = count[0]
	for i in range(1, 256):
		cumulation[i] = cumulation[i - 1] + count[i]
	for i in range(256):
		s[i] = int(255 * (cumulation[i] / n))
	for r in range(h):
		for c in range(w):
			img_new[r, c] = np.full(3, s[img[r, c, 0]])
	return img_new

if __name__ == '__main__':
	img_path = 'lena.bmp'
	img = cv2.imread(img_path)
	h, w =  img.shape[:2]
	#(a)
	count = calculate_hist(img)
	draw_hist('lena', count)

	#(b)
	img_divide3 = np.empty(img.shape, dtype = np.uint8)
	for r in range(h):
		for c in range(w):
			img_divide3[r, c] = img[r, c] // 3
	cv2.imwrite('lena_divide3.bmp', img_divide3)
	count = calculate_hist(img_divide3)
	draw_hist('lena_divide3', count)

	#(c)
	img_hist_equal = hist_equalization(img_divide3)
	cv2.imwrite('lena_hist_equal.bmp', img_hist_equal)
	count = calculate_hist(img_hist_equal)
	draw_hist('lena_hist_equal', count)

	'''
	cv2.imshow('lena', img)
	cv2.imshow('lena_divide3', img_divide3)
	cv2.imshow('lena_hist_equal', img_hist_equal)
	cv2.waitKey(0)
	cv2.destroyAllWindows()#'''