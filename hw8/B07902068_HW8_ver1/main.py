import cv2
import numpy as np
import random
import statistics
import math

def inside(r, c, h, w):
	if r < 0 or c < 0 or r >= h or c >= w:
		return False
	return True
def dilation(img, mask):
	h, w = img.shape
	new = np.zeros(img.shape, dtype = np.uint8)
	for r in range(h):
		for c in range(w):
			temp = []
			for (z, t) in mask:
				if inside(r - z, c - t, h, w):
					temp.append(img[r - z, c - t])
			new[r, c] =  max(temp)
	return new

def erosion(img, mask):
	h, w = img.shape
	new = np.zeros(img.shape, dtype = np.uint8)
	for r in range(h):
		for c in range(w):
			temp = []
			for (z, t) in mask:
				if inside(r + z, c + t, h, w):
					temp.append(img[r + z, c + t])
			new[r, c] = min(temp)
	return new
def opening(img, mask):
	return dilation(erosion(img, mask), mask)
def closing(img, mask):
	return erosion(dilation(img, mask), mask)


def gaussian_noise(img, amplitude):
	h, w = img.shape
	new = np.zeros(img.shape, dtype = np.uint8)
	for r in range(h):
		for c in range(w):
			value = int(img[r, c] + amplitude * random.gauss(0, 1))
			if(value > 255):
				value = 255
			new[r, c] = value
	return new
def salt_and_pepper_noise(img, threshold):
	h, w = img.shape
	new = np.zeros(img.shape, dtype = np.uint8)
	for r in range(h):
		for c in range(w):
			random_value = random.uniform(0, 1)
			if(random_value <= threshold):
				new[r, c] = 0
			elif (random_value >= 1 - threshold):
				new[r, c] = 255
			else:
				new[r, c] = img[r, c]
	return new

def box_filter(src_img, size):
	if (size % 2 != 1):
		print('wrong size for box filter')
		exit(-1)

	n = size * size
	center = size // 2
	pad_img = cv2.copyMakeBorder(src_img, center, center, center, center, cv2.BORDER_REFLECT101)


	h, w = src_img.shape
	new = np.zeros(img.shape, dtype = np.uint8)
	for r in range(h):
		for c in range(w):
			sum_value = 0
			for i in range(size):
				for j in range(size):
					sum_value += pad_img[r + i, c + j]
			new[r, c] = sum_value // n
	return new


def median_filter(src_img, size):
	if (size % 2 != 1):
		print('wrong size for median filter')
		exit(-1)

	n = size * size
	center = size // 2
	pad_img = cv2.copyMakeBorder(src_img, center, center, center, center, cv2.BORDER_REFLECT101)
	h, w = src_img.shape
	new = np.zeros(img.shape, dtype = np.uint8)
	for r in range(h):
		for c in range(w):
			temp = []
			for i in range(size):
				for j in range(size):
					temp.append(pad_img[r + i, c + j])
			new[r, c] = statistics.median(temp)
	return new

def cal_SNR(noise_img, src_img):
	if (src_img.shape != noise_img.shape):
		print('wrong shape')
		exit(-1)
	h, w = src_img.shape
	n = float(h * w)

	src_array = np.zeros(src_img.shape, dtype = np.float64)
	noise_array = np.zeros(noise_img.shape, dtype = np.float64)

	for r in range(h):
		for c in range(w):
			src_array[r, c] = float(src_img[r, c]) / 255.0
			noise_array[r, c] = float(noise_img[r, c]) / 255.0


	temp = 0.0
	for r in range(h):
		for c in range(w):
			temp += src_array[r, c]
	mean = temp / n

	temp = 0.0
	for r in range(h):
		for c in range(w):
			temp += (src_array[r, c] - mean)**2
	VS = temp / n


	temp = 0.0
	for r in range(h):
		for c in range(w):
			temp += (noise_array[r, c] - src_array[r, c])
	mean_n = temp / n

	temp = 0.0
	for r in range(h):
		for c in range(w):
			temp += ((noise_array[r, c] - src_array[r, c] - mean_n)**2)
	VN = temp / n

	SNR = 20 * math.log10( (VS**(0.5))/(VN**(0.5)))
	return SNR

if __name__ == '__main__':
	img_path = 'lena.bmp'
	img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
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
	
	#
	gaussian_10 = gaussian_noise(img, 10)
	cv2.imwrite('gaussian_10.bmp', gaussian_10)
	print('gaussin_10 SNR: ' + str(cal_SNR(gaussian_10, img)))

	gaussian_10_box_33 = box_filter(gaussian_10, 3)
	cv2.imwrite('gaussian_10_box_33.bmp', gaussian_10_box_33)
	print('gaussin_10_box_33 SNR: ' + str(cal_SNR(gaussian_10_box_33, img)))
	
	gaussian_10_box_55 = box_filter(gaussian_10, 5)
	cv2.imwrite('gaussian_10_box_55.bmp', gaussian_10_box_55)
	print('gaussin_10_box_55 SNR: ' + str(cal_SNR(gaussian_10_box_55, img)))

	gaussian_10_median_33 = median_filter(gaussian_10, 3)
	cv2.imwrite('gaussian_10_median_33.bmp', gaussian_10_median_33)
	print('gaussin_10_median_33 SNR: ' + str(cal_SNR(gaussian_10_median_33, img)))
	
	gaussian_10_median_55 = median_filter(gaussian_10, 5)
	cv2.imwrite('gaussian_10_median_55.bmp', gaussian_10_median_55)
	print('gaussin_10_median_55 SNR: ' + str(cal_SNR(gaussian_10_median_55, img)))

	gaussian_10_open_close = closing(opening(gaussian_10, mask), mask)
	cv2.imwrite('gaussian_10_open_close.bmp', gaussian_10_open_close)
	print('gaussin_10_open_close SNR: ' + str(cal_SNR(gaussian_10_open_close, img)))

	gaussian_10_close_open = opening(closing(gaussian_10, mask), mask)
	cv2.imwrite('gaussian_10_close_open.bmp', gaussian_10_close_open)
	print('gaussian_10_close_open SNR: ' + str(cal_SNR(gaussian_10_close_open, img)))



	
	#
	gaussian_30 = gaussian_noise(img, 30)
	cv2.imwrite('gaussian_30.bmp', gaussian_30)
	print('gaussin_30 SNR: ' + str(cal_SNR(gaussian_30, img)))

	gaussian_30_box_33 = box_filter(gaussian_30, 3)
	cv2.imwrite('gaussian_30_box_33.bmp', gaussian_30_box_33)
	print('gaussian_30_box_33 SNR: ' + str(cal_SNR(gaussian_30_box_33, img)))
	
	gaussian_30_box_55 = box_filter(gaussian_30, 5)
	cv2.imwrite('gaussian_30_box_55.bmp', gaussian_30_box_55)
	print('gaussian_30_box_55 SNR: ' + str(cal_SNR(gaussian_30_box_55, img)))

	gaussian_30_median_33 = median_filter(gaussian_30, 3)
	cv2.imwrite('gaussian_30_median_33.bmp', gaussian_30_median_33)
	print('gaussian_30_median_33 SNR: ' + str(cal_SNR(gaussian_30_median_33, img)))
	
	gaussian_30_median_55 = median_filter(gaussian_30, 5)
	cv2.imwrite('gaussian_30_median_55.bmp', gaussian_30_median_55)
	print('gaussian_30_median_55 SNR: ' + str(cal_SNR(gaussian_30_median_55, img)))

	gaussian_30_open_close = closing(opening(gaussian_30, mask), mask)
	cv2.imwrite('gaussian_30_open_close.bmp', gaussian_30_open_close)
	print('gaussian_30_open_close SNR: ' + str(cal_SNR(gaussian_30_open_close, img)))

	gaussian_30_close_open = opening(closing(gaussian_30, mask), mask)
	cv2.imwrite('gaussian_30_close_open.bmp', gaussian_30_close_open)
	print('gaussian_30_close_open SNR: ' + str(cal_SNR(gaussian_30_close_open, img)))
	#
	
	
	#
	salt_pepper_1 = salt_and_pepper_noise(img, 0.1)
	cv2.imwrite('salt_pepper_1.bmp', salt_pepper_1)
	print('salt_pepper_1 SNR: ' + str(cal_SNR(salt_pepper_1, img)))

	salt_pepper_1_box_33 = box_filter(salt_pepper_1, 3)
	cv2.imwrite('salt_pepper_1_box_33.bmp', salt_pepper_1_box_33)
	print('salt_pepper_1_box_33 SNR: ' + str(cal_SNR(salt_pepper_1_box_33, img)))

	salt_pepper_1_box_55 = box_filter(salt_pepper_1, 5)
	cv2.imwrite('salt_pepper_1_box_55.bmp', salt_pepper_1_box_55)
	print('salt_pepper_1_box_55 SNR: ' + str(cal_SNR(salt_pepper_1_box_55, img)))

	salt_pepper_1_median_33 = median_filter(salt_pepper_1, 3)
	cv2.imwrite('salt_pepper_1_median_33.bmp', salt_pepper_1_median_33)
	print('salt_pepper_1_median_33 SNR: ' + str(cal_SNR(salt_pepper_1_median_33, img)))

	salt_pepper_1_median_55 = median_filter(salt_pepper_1, 5)
	cv2.imwrite('salt_pepper_1_median_55.bmp', salt_pepper_1_median_55)
	print('salt_pepper_1_median_55 SNR: ' + str(cal_SNR(salt_pepper_1_median_55, img)))

	salt_pepper_1_open_close = closing(opening(salt_pepper_1, mask), mask)
	cv2.imwrite('salt_pepper_1_open_close.bmp', salt_pepper_1_open_close)
	print('salt_pepper_1_open_close SNR: ' + str(cal_SNR(salt_pepper_1_open_close, img)))

	salt_pepper_1_close_open = opening(closing(salt_pepper_1, mask), mask)
	cv2.imwrite('salt_pepper_1_close_open.bmp', salt_pepper_1_close_open)
	print('salt_pepper_1_close_open SNR: ' + str(cal_SNR(salt_pepper_1_close_open, img)))
	'''
	'''
	
	#
	salt_pepper_05 = salt_and_pepper_noise(img, 0.05)
	cv2.imwrite('salt_pepper_05.bmp', salt_pepper_05)
	print('salt_pepper_05 SNR: ' + str(cal_SNR(salt_pepper_05, img)))

	salt_pepper_05_box_33 = box_filter(salt_pepper_05, 3)
	cv2.imwrite('salt_pepper_05_box_33.bmp', salt_pepper_05_box_33)
	print('salt_pepper_05_box_33 SNR: ' + str(cal_SNR(salt_pepper_05_box_33, img)))

	salt_pepper_05_box_55 = box_filter(salt_pepper_05, 5)
	cv2.imwrite('salt_pepper_05_box_55.bmp', salt_pepper_05_box_55)
	print('salt_pepper_05_box_55 SNR: ' + str(cal_SNR(salt_pepper_05_box_55, img)))

	salt_pepper_05_median_33 = median_filter(salt_pepper_05, 3)
	cv2.imwrite('salt_pepper_05_median_33.bmp', salt_pepper_05_median_33)
	print('salt_pepper_05_median_33 SNR: ' + str(cal_SNR(salt_pepper_05_median_33, img)))

	salt_pepper_05_median_55 = median_filter(salt_pepper_05, 5)
	cv2.imwrite('salt_pepper_05_median_55.bmp', salt_pepper_05_median_55)
	print('salt_pepper_05_median_55 SNR: ' + str(cal_SNR(salt_pepper_05_median_55, img)))

	salt_pepper_05_open_close = closing(opening(salt_pepper_05, mask), mask)
	cv2.imwrite('salt_pepper_05_open_close.bmp', salt_pepper_05_open_close)
	print('salt_pepper_05_open_close SNR: ' + str(cal_SNR(salt_pepper_05_open_close, img)))

	salt_pepper_05_close_open = opening(closing(salt_pepper_05, mask), mask)
	cv2.imwrite('salt_pepper_05_close_open.bmp', salt_pepper_05_close_open)
	print('salt_pepper_05_close_open SNR: ' + str(cal_SNR(salt_pepper_05_close_open, img)))
	#cv2.waitKey(0)
	#'''