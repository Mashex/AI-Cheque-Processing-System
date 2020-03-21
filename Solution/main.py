import argparse
import os, errno
import sys
from utils import *
from unet_inference import get_all_regions
import time
def get_info(image_path):
	original_image, resized_image, extracted_regions, micr_region = get_all_regions(image_path)
	data = get_all_rois_and_text(image_path)
	unfiltered_text_data = get_unfiltered_text(original_image, data, extracted_regions)
	if micr_region != None:
		micr_text = get_MICR_text(resized_image, micr_region)
	else:
		micr_text = ""
	filtered_text_data = filter_text_data(unfiltered_text_data, micr_text, data)
	return filtered_text_data


if  __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description="Extract Information from Cheques to a structured format"
	)
	parser.add_argument("-i","--image", type=str, nargs="?", help="The Image from which you want to extract information", required=True)
	
	args = parser.parse_args()
	if args.image:
		print ("Extracting Data")
		start = time.time()
		get_info(args.image)
		print (time.time() - start)



