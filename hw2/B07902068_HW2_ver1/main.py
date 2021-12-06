import cv2
import numpy as np
import matplotlib.pyplot as plt


#(a)
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
#(b)
def histogram(img):
	count = np.zeros(256, dtype = int)
	(h, w) = img.shape[:2]
	for r in range(h):
		for c in range(w):
			count[img[r, c, 0]] += 1
	index = np.arange(0, 256)
	plt.bar(index, count)
	plt.xlabel('Intensity')
	plt.ylabel('Count')

	plt.title('Histogram of lena.bmp')
	plt.savefig('lena_hist.png')
	#plt.show()
	return

#(c)
class Component:
	def __init__(self, r, c, label): #initialize properties of a component
		self.top = self.bottom = self.sumR = r # 'top, bottom, left, right' are the boundaries of bounding box of component
		self.left = self.right = self.sumC = c #sumR and sumC are sum of row and sum of column
		self.area = 1
		self.label = label # used as pointer of disjoint set
	def new_point(self, r, c):
		self.area += 1
		self.sumR += r
		self.sumC += c
		if c < self.left:
			self.left = c
		elif c > self.right:
			self.right = c

		if r < self.top:
			self.top = r
		elif r > self.bottom:
			self.bottom = r
	def find_final_label(compo_list, start): # compo_list is a list of component objects, 'start, final' are indices 
		current = start
		while compo_list[current].label != current:
			current = compo_list[current].label
		final = current
		Component.point_to_final(compo_list, start, final)
		return final
	def point_to_final(compo_list, start, final): # connect each component on the path directly to the root
		current = start
		while compo_list[current].label != final:
			nexti = compo_list[current].label
			compo_list[current].label = final
			current = nexti
	def merge_to_final(compo_list, start, final): # merge the properties of the start to the final
		compo_list[final].area += compo_list[start].area
		compo_list[final].sumR += compo_list[start].sumR
		compo_list[final].sumC += compo_list[start].sumC

		if compo_list[start].left < compo_list[final].left:
			compo_list[final].left = compo_list[start].left

		if compo_list[start].right > compo_list[final].right:
			compo_list[final].right = compo_list[start].right 

		if compo_list[start].top < compo_list[final].top:
			compo_list[final].top = compo_list[start].top 

		if compo_list[start].bottom > compo_list[final].bottom:
			compo_list[final].bottom = compo_list[start].bottom




def connected_component(img_binary):
	compo_list = [] #store component objects
	(h, w) = img_binary.shape[:2]
	index_matrix = np.zeros((h, w), dtype = int) #pixel position to its index of component
	index_matrix -= 1

	new_label = 0
	if  img_binary[0, 0, 0] == 255: #do for (0, 0)
		index_matrix[0, 0] = new_label
		compo_list.append(Component(0, 0, new_label))
		new_label += 1
	for c in range(1, w): # do for the firt row
		if img_binary[0, c, 0] == 255:
			if index_matrix[0, c - 1] == -1:
				index_matrix[0, c] = new_label # add a new component and label
				compo_list.append(Component(0, c, new_label))
				new_label += 1
			else:
				index_matrix[0, c] = index_matrix[0, c - 1] # add to existing component and label
				compo_list[index_matrix[0, c - 1]].new_point(0, c)
	
	for r in range(1, h):
		if img_binary[r, 0, 0] == 255: # do for the first column
			if index_matrix[r - 1, 0] == -1:# add a new component and label
				index_matrix[r, 0] = new_label
				compo_list.append(Component(r, 0, new_label))
				new_label += 1
			else:
				index_matrix[r, 0] = index_matrix[r - 1, 0]# add to existing component and label
				compo_list[index_matrix[r - 1, 0]].new_point(r, 0)

		for c in range(1, w):
			if img_binary[r, c, 0] == 255:

				if index_matrix[r, c - 1] > -1:
					index_matrix[r, c] = index_matrix[r, c - 1]# add to existing component and label
					compo_list[index_matrix[r, c - 1]].new_point(r, c)

					if index_matrix[r - 1, c] > -1: #label two connected components as one by connect the root of two disjoint sets
						label_1 = Component.find_final_label(compo_list, index_matrix[r, c - 1])
						label_2 = Component.find_final_label(compo_list, index_matrix[r - 1, c])
						if label_1 < label_2:
							compo_list[label_2].label = label_1
						else:
							compo_list[label_1].label = label_2


				elif index_matrix[r - 1, c] > -1:# add to existing component and label
					index_matrix[r, c] = index_matrix[r - 1, c]
					compo_list[index_matrix[r - 1, c]].new_point(r, c)
				else:
					index_matrix[r, c] = new_label# add a new component and label
					compo_list.append(Component(r, c, new_label))
					new_label += 1

	final_list = [] #the list of resulting connected components and labels
	for i in range(len(compo_list)):
		final = Component.find_final_label(compo_list, i)
		if i != final:
			Component.merge_to_final(compo_list, i, final) # merge the properties to the final

		if final not in final_list:
			final_list.append(final)

	for i in final_list:
		if compo_list[i].area >= 500: #draw bounding box and centroid
			y1 = compo_list[i].top
			x1 = compo_list[i].left
			y2 = compo_list[i].bottom
			x2 = compo_list[i].right
			cv2.rectangle(img_binary, (x1, y1), (x2, y2), (0, 255, 0), 2)
			r_bar = compo_list[i].sumR // compo_list[i].area
			c_bar = compo_list[i].sumC // compo_list[i].area

			#print(c_bar, r_bar)
			cv2.circle(img_binary, (c_bar, r_bar), 3, (0, 0, 255), -1)

	cv2.imwrite('connected_component.bmp', img_binary)
	'''cv2.imshow('bounding box', img_binary)

	cv2.waitKey(0)
	cv2.destroyAllWindows()#'''

	return

if __name__ == '__main__':
	img_path = 'lena.bmp'
	img = cv2.imread(img_path)
	img_binary = binarize(img)
	histogram(img)
	connected_component(img_binary)
