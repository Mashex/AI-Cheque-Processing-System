import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Function, Variable
from pathlib import Path
from itertools import groupby
import json
import glob
import pytesseract
import time
import math
import os 
import imutils
import re
from MICR.predict import *
os.environ.setdefault('GOOGLE_APPLICATION_CREDENTIALS',os.path.join("..","OCRProject.json"))

class_mapping = ["Account Holder Name","Account Number","Bank Name","IFSC Code","MICR"]

def get_contours(gray):
	groupCnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
	groupCnts = imutils.grab_contours(groupCnts)
	groupLocs = []
	for (i, c) in enumerate(groupCnts):
		(x, y, w, h) = cv2.boundingRect(c)
		if w > 5 and h > 2:
			groupLocs.append((x, y, w, h))
	groupLocs = sorted(groupLocs, key=lambda x:x[0])
	
	return groupLocs

def detect_text(path):
	"""Detects text in the file."""
	from google.cloud import vision
	import io
	client = vision.ImageAnnotatorClient()

	with io.open(path, 'rb') as image_file:
		content = image_file.read()

	image = vision.types.Image(content=content)

	response = client.text_detection(image=image)
	texts = response.text_annotations
	#print('Texts:')

	# for text in texts:
	# 	print('\n"{}"'.format(text.description))

	# 	vertices = (['({},{})'.format(vertex.x, vertex.y)
	# 				for vertex in text.bounding_poly.vertices])

	# 	print('bounds: {}'.format(','.join(vertices)))

	if response.error.message:
		raise Exception(
			'{}\nFor more info on error messages, check: '
			'https://cloud.google.com/apis/design/errors'.format(
				response.error.message))
	
	return texts
		
def get_google_text(image_path):
	
	#cv2.imwrite(f"temp\\temp.jpg",img)
	text = detect_text(image_path)
	
	return text

def get_all_rois_and_text(image_path):
	texts = get_google_text(image_path)

	first = True

	data = []
	for text in texts:
		if first==True:
			all_text_info = text
			all_text_desc = all_text_info.description
			all_text_vertices = [(vertex.x, vertex.y) for vertex in all_text_info.bounding_poly.vertices]
			first = False
		text_desc = text.description
		vertices = ([(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices])
		x1=min([x[0] for x in vertices])
		x2=max([x[0] for x in vertices])
		y1=min([x[1] for x in vertices])
		y2=max([x[1] for x in vertices])

		vertices = (x1,y1,x2,y2)
		data.append((text_desc, vertices))
	return data

def calculateIntersection(a0, a1, b0, b1):
	if a0 >= b0 and a1 <= b1: # Contained
		intersection = a1 - a0
	elif a0 < b0 and a1 > b1: # Contains
		intersection = b1 - b0
	elif a0 < b0 and a1 > b0: # Intersects right
		intersection = a1 - b0
	elif a1 > b1 and a0 < b1: # Intersects left
		intersection = b1 - a0
	else: # No intersection (either side)
		intersection = 0

	return intersection

def get_intersection(region1, region2):
	X0, Y0, X1, Y1 = region1
	x0, y0, x1, y1 = region2

	AREA = float((x1 - x0) * (y1 - y0))
	if AREA != 0:

		width = calculateIntersection(x0, x1, X0, X1)        
		height = calculateIntersection(y0, y1, Y0, Y1)        
		area = width * height

		percent = area / AREA

		return percent
	else:
		return 0


def get_unfiltered_text(original_image, data, extracted_regions):

	shape = original_image.shape
	unfiltered_data = {}
	mask = np.zeros(shape)
	threshold = 0.5

	ver1 = [vertices_1 for class_id, vertices_1, region in extracted_regions]
	ver2 = [vertices_2 for text, vertices_2 in data]

	for class_id, vertices_1, region in extracted_regions:
		class_name = class_mapping[class_id]
		region1 = vertices_1

		intersection_regions = []
		expected_text = ""
		for text, vertices_2 in data:
			region2 = vertices_2
			Width = region2[2] - region2[0]
			Height = region2[3] - region2[1]

			if Width<shape[1]/2 and Height<shape[0]/2:

				intersection_val = get_intersection(region1, region2)
				if intersection_val>threshold:
					intersection_regions.append(region2)
					expected_text += text.strip() + " "
		unfiltered_data[class_name] = [expected_text, intersection_regions]

		#print (f"{class_name}: {expected_text}")
	return unfiltered_data


def get_MICR_text(image, micr_region):
	((gX, gY, gW, gH), group) = micr_region
	#image = image[gY:gY+gH, gX: gX + gW]
	text = predict_MICR_code(group)
	return text


def overall_region_size(intersection_regions):
	x1=min([x[0] for x in intersection_regions])
	x2=max([x[2] for x in intersection_regions])
	y1=min([x[1] for x in intersection_regions])
	y2=max([x[3] for x in intersection_regions])

	return x1,y1,x2,y2

def find_missing_text(data, region, text_current, intersection_regions):
	threshold = 10
	X0, Y0, X1, Y1 = region
	for text, vertices in data:
		x0, y0, x1, y1 = vertices

		if abs(y0-Y0)<=5:
			#print (text, vertices, region)
			if x0>X1 and x0-X1<=threshold :
				text_current += text + " "
				intersection_regions.append(vertices)
			elif x1< X0 and X0 - x1<=threshold:
				temp_text  = text + " " + text_current
				text_current = temp_text
				intersection_regions.append(vertices)
	#print (text_current)
	return text_current, intersection_regions			

def find_ifsc_code(data):
	for text, vertices_2 in data:

		obj = re.search("[A-Za-z]{4}[a-zA-Z0-9]{7}",text)
		if obj != None:
			return obj.group()
	else:
		return None

def filter_text_data(unfiltered_text_data, micr_text, full_text_data):
	

	if unfiltered_text_data['Account Number'][0].strip().isdigit() == False:
		x = re.search(r"\d{5,15}", unfiltered_text_data['Account Number'][0].strip())
		unfiltered_text_data['Account Number'][0] = x

	overall_account_name_region = overall_region_size(unfiltered_text_data['Account Holder Name'][1])
	#print (unfiltered_text_data['Account Holder Name'][1] , overall_account_name_region)
	output = find_missing_text(full_text_data, overall_account_name_region, unfiltered_text_data['Account Holder Name'][0],unfiltered_text_data['Account Holder Name'][1])
	unfiltered_text_data['Account Holder Name'][0], unfiltered_text_data['Account Holder Name'][1] = output

	if "IFSC Code" in unfiltered_text_data:
		unfiltered_text_data["IFSC Code"] = re.search("[A-Za-z]{4}[a-zA-Z0-9]{7}",unfiltered_text_data["IFSC Code"]).group()
	else:
		ifsc_code = find_ifsc_code(full_text_data)
		if ifsc_code != None:
			unfiltered_text_data["IFSC Code"] = [ifsc_code]
	for x in unfiltered_text_data:
		print (f"{x}: {unfiltered_text_data[x][0]}")
	if micr_text != "":
		print (f"MICR: {micr_text}")



def get_text(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
	#gray = cv2.medianBlur(gray, 3)
	
	cv2.imwrite(f"temp\\temp.jpg",gray)
	
	config = ('-c tessedit_char_whitelist=/0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-psm 6')
	text = pytesseract.image_to_string(Image.open(f"temp\\temp.jpg"), lang = 'eng',config=config)
	
	return text

