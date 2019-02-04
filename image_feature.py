
import cv2
import base64
import numpy as np

class image_feature_extract:

	def __init__(self):

		self.red_color_range = [(0, 0, 20), (50, 150, 255)]
		self.blue_color_range = [(50,0,0),(255,0,0)]
		self.white_color_range = [(235, 235, 235), (255, 255, 255)]
		self.pink_color_range = [(255, 0, 102),(255, 204, 255)]

	def extract_feature(self,img_location):
		#img_location = 'data/cell-images/Uninfected/C33P1thinF_IMG_20150619_121300a_cell_135.png'
		#img_location = 'data/cell-images/Parasitized/C33P1thinF_IMG_20150619_114756a_cell_179.png'
		vector_size = 32
		img = np.asarray(cv2.imread(img_location))
		alg = cv2.KAZE_create()
		kps = alg.detect(img)
		kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
		kps, dsc = alg.compute(img, kps)
		dsc = dsc.flatten()
		needed_size = (vector_size * 64)
		if dsc.size < needed_size:
			dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
		return dsc

	def bgr2rgb(self,img):
		return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	def rgb2bgr(self,img):
		return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

	def count_color(self,img_color, mask_color1, mask_color2=None, mask_color3=None, mask_color4=None, blur=None):
		mask = cv2.inRange(img_color, mask_color1[0], mask_color1[1])
		if mask_color2:
			mask2 = cv2.inRange(img_color, mask_color2[0], mask_color2[1])
			mask = cv2.bitwise_or(mask, mask2)
		if mask_color3:
			mask3 = cv2.inRange(img_color, mask_color3[0], mask_color3[1])
			mask = cv2.bitwise_or(mask, mask3)
		if mask_color4:
			mask4 = cv2.inRange(img_color, mask_color4[0], mask_color4[1])
			mask = cv2.bitwise_or(mask, mask4)
		if blur:
			mask = cv2.medianBlur(mask, blur)
		mask_color_count = cv2.countNonZero(mask)
		return mask_color_count

	def flatten_rgb(self,img):
		r, g, b = cv2.split(img)
		r_filter = (r == np.maximum(np.maximum(r, g), b)) & (r >= 120) & (g < 150) & (b < 150)
		g_filter = (g == np.maximum(np.maximum(r, g), b)) & (g >= 120) & (r < 150) & (b < 150)
		b_filter = (b == np.maximum(np.maximum(r, g), b)) & (b >= 120) & (r < 150) & (g < 150)
		y_filter = ((r >= 128) & (g >= 128) & (b < 100))
		# w_filter = ((r <= 160) & (r >= 50) & (g <= 160) & (g >= 50) & (b <= 160) & (b >= 50))

		r[y_filter], g[y_filter] = 255, 255
		b[np.invert(y_filter)] = 0
		# r[w_filter], g[w_filter] = 255, 255
		# b[np.invert(w_filter)] = 0

		b[b_filter], b[np.invert(b_filter)] = 255, 0
		r[r_filter], r[np.invert(r_filter)] = 255, 0
		g[g_filter], g[np.invert(g_filter)] = 255, 0

		flattened = cv2.merge((r, g, b))
		return flattened
