import glob, os, threading
import numpy as np

filenames = glob.glob('out/*.jpg')
np.random.shuffle(filenames)

total_size = 500000
train_file_len = int(0.8*total_size)
test_file_len = total_size - train_file_len

path = os.getcwd()


train_files = filenames[:train_file_len]

file = open('labels.txt','a+')
for x in train_files:
	folder,filename = os.path.split(x)
	text, idx = filename.split('_')
	file.write(os.path.join(path, x)+"\t"+text+"\n")
file.close()

test_files = filenames[train_file_len:train_file_len + test_file_len]

file = open('val_labels.txt','a+')
for x in test_files:
	folder,filename = os.path.split(x)
	text, idx = filename.split('_')
	file.write(os.path.join(path, x)+"\t"+text+"\n")
file.close()