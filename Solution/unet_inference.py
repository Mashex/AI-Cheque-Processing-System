import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm

from model import *
from utils import get_contours
from pathlib import Path
from itertools import groupby
import json
import glob
import pytesseract
import time
import math
import imutils
import cv2
from skimage.segmentation import clear_border
from imutils import contours
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

WIDTH = 512
HEIGHT = 256
category_num = 5 + 1

if torch.cuda.is_available():
	device = "cuda:0"
else:
	device = 'cpu'

net = UNet(n_channels=3, n_classes=category_num).to(device)
net.load_state_dict(torch.load(r"..\models\newdata-50-4-0.01.pth"))

def get_all_regions(image_path):
	
	net.eval()
	count = 0
	for img_name, img, resized_image, original_image in test_generator(image_path):
		all_regions = []
		micr_region = None
		X = torch.tensor(img, dtype=torch.float32).to(device)
		mask_pred = net(X)
		mask_pred1 = mask_pred.cpu().detach().numpy()
		mask_prob = np.argmax(mask_pred1, axis=1)

		mask_prob = mask_prob.transpose((1, 2, 0))
		for i in range(5):            
			data = class_x_processing(img,mask_prob,i, resized_image, original_image)
			if len(data[0])==4:
				(gX, gY, gW, gH), group = data
				if i!=4 and len(group) != 0:
					all_regions.append((i,(gX, gY, gW, gH) , group))
				else:
					micr_region = ((gX, gY, gW, gH), group)
			else:
				continue
	return original_image, resized_image, all_regions, micr_region

def class_x_processing(img,mask_prob,class_id, resized_image, original_image):

	#print (resized_image.shape)
	#print (original_image.shape)

	ori_x = original_image.shape[1]
	ori_y = original_image.shape[0]
	Sx = ori_x/resized_image.shape[1]
	Sy = ori_y/resized_image.shape[0]
	
	mask = np.zeros((HEIGHT, WIDTH),np.float32)
	indices = np.where(mask_prob==class_id)
	mask[indices[0],indices[1]] = 255

	rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7))
	gray = cv2.cvtColor(np.repeat(mask[:, :, np.newaxis],3,axis = 2), cv2.COLOR_BGR2GRAY).astype("uint8")
	groupLocs = get_contours(gray)
	
	for (gX, gY, gW, gH) in groupLocs:
		gX2 = gX + gW
		gY2 = gY + gH
		scaled_gY = int(gY * Sy)
		scaled_gY2 = int(gY2 * Sy)
		scaled_gX = int(gX * Sx)
		scaled_gX2 = int(gX2 * Sx)

		group = original_image[scaled_gY-5:scaled_gY2+5, scaled_gX-5:scaled_gX2+5]
		#group = resized_image[gY-5:gY + gH+5, gX-5:gX + gW+5]
		curr_time = time.time()

		#cv2.imwrite(f"test_temp\\{curr_time}.jpg",group)

		# if class_id != 4:
		#     continue#text = get_text(group)
		# else:
		#     text = get_MICR_text(group)
		# print (text)
		return (scaled_gX, scaled_gY,scaled_gX2, scaled_gY2), group
		#return (gX, gY, gW, gH), group
	return (),()

#image_path = r"test_images/"
#predict(test_img_dir)