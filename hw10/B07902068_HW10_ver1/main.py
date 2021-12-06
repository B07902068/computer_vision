import cv2
import numpy as np

def zero_cross(matrix):
	h, w = matrix.shape
	new_img = np.zeros(matrix.shape, dtype = np.uint8)
	new_img[:] = 255

	pad_matrix = cv2.copyMakeBorder(matrix, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
	matrix = np.zeros(pad_matrix.shape, dtype = int)
	matrix[:] = pad_matrix

	for r in range(h):
		for c in range(w):
			if (matrix[r+1, c+1] >= 1):
				for i in range(3):
					for j in range(3):
						if (matrix[r+i, c+j] <= -1):
							new_img[r, c] = 0
	return new_img

def Laplacian_1(img, threshold):
	h, w = img.shape
	matrix = np.zeros(img.shape, dtype = int)

	pad_img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
	img = np.zeros(pad_img.shape, dtype = int)
	img[:] = pad_img

	mask = np.array([[0, 1, 0],
					 [1, -4, 1],
					 [0, 1, 0]])

	for r in range(h):
		for c in range(w):
			gradient = 0.0
			for i in range(3):
				for j in range(3):
					if (mask[i, j] != 0):
						gradient += img[r+i, c+j] * mask[i, j]

			if (gradient >= threshold):
				matrix[r, c] = 1
			elif (gradient  <= -threshold):
				matrix[r, c] = -1			

	return zero_cross(matrix)

def Laplacian_2(img, threshold):
	h, w = img.shape
	matrix = np.zeros(img.shape, dtype = int)

	pad_img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
	img = np.zeros(pad_img.shape, dtype = int)
	img[:] = pad_img

	mask = np.array([[1, 1, 1],
					 [1, -8, 1],
					 [1, 1, 1]])

	for r in range(h):
		for c in range(w):
			gradient = 0.0
			for i in range(3):
				for j in range(3):
					gradient += img[r+i, c+j] * mask[i, j]
					
			gradient /= 3
			if (gradient >= threshold):
				matrix[r, c] = 1
			elif (gradient  <= -threshold):
				matrix[r, c] = -1			

	return zero_cross(matrix)

def minimum_variance_Laplacian(img, threshold):
	h, w = img.shape
	matrix = np.zeros(img.shape, dtype = int)

	pad_img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
	img = np.zeros(pad_img.shape, dtype = int)
	img[:] = pad_img

	mask = np.array([[2, -1, 2],
					 [-1, -4, -1],
					 [2, -1, 2]])

	for r in range(h):
		for c in range(w):
			gradient = 0.0
			for i in range(3):
				for j in range(3):
					gradient += img[r+i, c+j] * mask[i, j]
					
			gradient /= 3
			if (gradient >= threshold):
				matrix[r, c] = 1
			elif (gradient  <= -threshold):
				matrix[r, c] = -1			

	return zero_cross(matrix)

def Laplacian_of_Gaussian(img, threshold):
	h, w = img.shape
	matrix = np.zeros(img.shape, dtype = int)

	pad_img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_REPLICATE)
	img = np.zeros(pad_img.shape, dtype = int)
	img[:] = pad_img

	mask = np.array([[0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0],
					 [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
					 [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
					 [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
					 [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
					 [-2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2],
					 [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
					 [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
					 [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
					 [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
					 [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0],])

	for r in range(h):
		for c in range(w):
			gradient = 0.0
			for i in range(11):
				for j in range(11):
					if (mask[i, j] != 0):
						gradient += img[r+i, c+j] * mask[i, j]
					
			if (gradient >= threshold):
				matrix[r, c] = 1
			elif (gradient  <= -threshold):
				matrix[r, c] = -1			

	return zero_cross(matrix)

def Diffirence_of_Gaussian(img, threshold):
	h, w = img.shape
	matrix = np.zeros(img.shape, dtype = int)

	pad_img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_REPLICATE)
	img = np.zeros(pad_img.shape, dtype = int)
	img[:] = pad_img

	mask = np.array([[-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],
					 [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
					 [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
					 [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
					 [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
					 [-8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8],
					 [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
					 [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
					 [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
					 [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
					 [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1]])

	for r in range(h):
		for c in range(w):
			gradient = 0.0
			for i in range(11):
				for j in range(11):
					gradient += img[r+i, c+j] * mask[i, j]
					
			if (gradient >= threshold):
				matrix[r, c] = 1
			elif (gradient  <= -threshold):
				matrix[r, c] = -1			

	return zero_cross(matrix)

if __name__ == '__main__':
	img_path = 'lena.bmp'
	img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

	cv2.imwrite('Laplacian_1.bmp', Laplacian_1(img, 15))
	cv2.imwrite('Laplacian_2.bmp', Laplacian_2(img, 15))
	cv2.imwrite('minimum_variance_Laplacian.bmp', minimum_variance_Laplacian(img, 20))
	cv2.imwrite('Laplacian_of_Gaussian.bmp', Laplacian_of_Gaussian(img, 3000))
	cv2.imwrite('Diffirence_of_Gaussian.bmp', Diffirence_of_Gaussian(img, 1))