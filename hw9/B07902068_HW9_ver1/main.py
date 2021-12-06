import cv2
import numpy as np

def Robert(img, threshold):
	h, w = img.shape
	new = np.zeros(img.shape, dtype = np.uint8)

	pad_img = cv2.copyMakeBorder(img, 0, 1, 0, 1, cv2.BORDER_REPLICATE)
	img = np.zeros(pad_img.shape, dtype = int)
	img[:] = pad_img
	
	for r in range(h):
		for c in range(w):
			r1 = img[r+1, c+1] - img[r, c]
			r2 = img[r+1, c] - img[r, c+1]
			gradient = (r1**2 + r2**2)** 0.5
			if (gradient < threshold):
				new[r, c] = 255
	return new

def Prewitt(img, threshold):
	h, w = img.shape
	new = np.zeros(img.shape, dtype = np.uint8)

	pad_img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
	img = np.zeros(pad_img.shape, dtype = int)
	img[:] = pad_img
	
	for r in range(h):
		for c in range(w):
			p1 = (img[r+2, c] + img[r+2, c+1] + img[r+2, c+2]) - (img[r, c] + img[r, c+1] + img[r, c+2])
			p2 = (img[r, c+2] + img[r+1, c+2] + img[r+2, c+2]) - (img[r, c] + img[r+1, c] + img[r+2, c])
			gradient = (p1**2 + p2**2)** 0.5
			if (gradient < threshold):
				new[r, c] = 255
	return new

def Sobel(img, threshold):
	h, w = img.shape
	new = np.zeros(img.shape, dtype = np.uint8)

	pad_img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
	img = np.zeros(pad_img.shape, dtype = int)
	img[:] = pad_img
	
	for r in range(h):
		for c in range(w):
			s1 = (img[r+2, c] + 2 * img[r+2, c+1] + img[r+2, c+2]) - (img[r, c] + 2 * img[r, c+1] + img[r, c+2])
			s2 = (img[r, c+2] + 2 * img[r+1, c+2] + img[r+2, c+2]) - (img[r, c] + 2 * img[r+1, c] + img[r+2, c])
			gradient = (s1**2 + s2**2)** 0.5
			if (gradient < threshold):
				new[r, c] = 255
	return new

def Frei_and_Chen(img, threshold):
	h, w = img.shape
	new = np.zeros(img.shape, dtype = np.uint8)

	pad_img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
	img = np.zeros(pad_img.shape, dtype = int)
	img[:] = pad_img
	
	for r in range(h):
		for c in range(w):
			f1 = (img[r+2, c] + (2**0.5) * img[r+2, c+1] + img[r+2, c+2]) - (img[r, c] + (2**0.5) * img[r, c+1] + img[r, c+2])
			f2 = (img[r, c+2] + (2**0.5) * img[r+1, c+2] + img[r+2, c+2]) - (img[r, c] + (2**0.5) * img[r+1, c] + img[r+2, c])
			gradient = (f1**2 + f2**2)** 0.5
			if (gradient < threshold):
				new[r, c] = 255
	return new

def Kirsch(img, threshold):
	h, w = img.shape
	new = np.zeros(img.shape, dtype = np.uint8)

	pad_img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
	img = np.zeros(pad_img.shape, dtype = int)
	img[:] = pad_img

	mask = np.array([[[-3, -3, 5], 
					  [-3, 0, 5], 
					  [-3, -3, 5]],

					 [[-3, 5, 5], 
					  [-3, 0, 5], 
					  [-3, -3, -3]],

					 [[5, 5, 5], 
					  [-3, 0, -3], 
					  [-3, -3, -3]],

					 [[5, 5, -3], 
					  [5, 0, -3], 
					  [-3, -3, -3]], 

					 [[5, -3, -3], 
					  [5, 0, -3], 
					  [5, -3, -3]], 

					 [[-3, -3, -3], 
					  [5, 0, -3], 
					  [5, 5, -3]], 

					 [[-3, -3, -3], 
					  [-3, 0, -3], 
					  [5, 5, 5]], 

					 [[-3, -3, -3], 
					  [-3, 0, 5], 
					  [-3, 5, 5]]])
	k = np.zeros(8)
	
	for r in range(h):
		for c in range(w):
			k[:] = 0
			for x in range(8):
				for i in range(3):
					for j in range(3):
						if(mask[x, i, j] != 0):
							k[x] += img[r+i, c+j] * mask[x, i, j]
			
			maximum = max(k)
			
			if (maximum < threshold):
				new[r, c] = 255
	return new

def Robinson(img, threshold):
	h, w = img.shape
	new = np.zeros(img.shape, dtype = np.uint8)

	pad_img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
	img = np.zeros(pad_img.shape, dtype = int)
	img[:] = pad_img

	mask = np.array([[[-1, 0, 1], 
					  [-2, 0, 2], 
					  [-1, 0, 1]],

					 [[0, 1, 2], 
					  [-1, 0, 1], 
					  [-2, -1, 0]],

					 [[1, 2, 1], 
					  [0, 0, 0], 
					  [-1, -2, -1]],

					 [[2, 1, 0], 
					  [1, 0, -1], 
					  [0, -1, -2]], 

					 [[1, 0, -1], 
					  [2, 0, -2], 
					  [1, 0, -1]], 

					 [[0, -1, -2], 
					  [1, 0, -1], 
					  [2, 1, 0]], 

					 [[-1, -2, -1], 
					  [0, 0, 0], 
					  [1, 2, 1]], 

					 [[-2, -1, 0], 
					  [-1, 0, 1], 
					  [0, 1, 2]]])
	k = np.zeros(8)
	
	for r in range(h):
		for c in range(w):
			k[:] = 0
			for x in range(8):
				for i in range(3):
					for j in range(3):
						if(mask[x, i, j] != 0):
							k[x] += img[r+i, c+j] * mask[x, i, j]
			
			maximum = max(k)
			
			if (maximum < threshold):
				new[r, c] = 255
	return new

def Nevatia_Babu_5x5(img, threshold):
	h, w = img.shape
	new = np.zeros(img.shape, dtype = np.uint8)

	pad_img = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
	img = np.zeros(pad_img.shape, dtype = int)
	img[:] = pad_img

	mask = np.array([[[100, 100, 100, 100, 100], 
					  [100, 100, 100, 100, 100],
					  [0, 0, 0, 0, 0],
					  [-100, -100, -100, -100, -100],
					  [-100, -100, -100, -100, -100]],

					 [[100, 100, 100, 100, 100], 
					  [100, 100, 100, 78, -32],
					  [100, 92, 0, -92, -100],
					  [32, -78, -100, -100, -100],
					  [-100, -100, -100, -100, -100]],

					  [[100, 100, 100, 32, -100], 
					  [100, 100, 92, -78, -100],
					  [100, 100, 0, -100, -100],
					  [100, 78, -92, -100, -100],
					  [100, -32, -100, -100, -100]],

					 [[-100, -100, 0, 100, 100], 
					  [-100, -100, 0, 100, 100],
					  [-100, -100, 0, 100, 100],
					  [-100, -100, 0, 100, 100],
					  [-100, -100, 0, 100, 100]], 

					 [[-100, 32, 100, 100, 100], 
					  [-100, -78, 92, 100, 100],
					  [-100, -100, 0, 100, 100],
					  [-100, -100, -92, 78, 100],
					  [-100, -100, -100, -32, 100]], 

					 [[100, 100, 100, 100, 100], 
					  [-32, 78, 100, 100, 100],
					  [-100, -92, 0, 92, 100],
					  [-100, -100, -100, -78, 32],
					  [-100, -100, -100, -100, -100]]])
	k = np.zeros(6)
	
	for r in range(h):
		for c in range(w):
			k[:] = 0
			for x in range(6):
				for i in range(5):
					for j in range(5):
						if(mask[x, i, j] != 0):
							k[x] += img[r+i, c+j] * mask[x, i, j]
			
			maximum = max(k)
			
			if (maximum < threshold):
				new[r, c] = 255
	return new


if __name__ == '__main__':
	img_path = 'lena.bmp'
	img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
	
	cv2.imwrite('Robert.bmp', Robert(img, 12))
	cv2.imwrite('Prewitt.bmp', Prewitt(img, 24))
	cv2.imwrite('Sobel.bmp', Sobel(img, 38))
	cv2.imwrite('Frei_and_Chen.bmp', Frei_and_Chen(img, 30))
	cv2.imwrite('Kirsch.bmp', Kirsch(img, 135))
	cv2.imwrite('Robinson.bmp', Robinson(img, 43))
	cv2.imwrite('Nevatia_Babu_5x5.bmp', Nevatia_Babu_5x5(img, 12500))
	
