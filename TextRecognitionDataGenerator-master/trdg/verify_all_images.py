import glob, os, threading
from PIL import Image, ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True

filenames = glob.glob('out/*.jpg')
steps = int(len(filenames)/10)
i = 0
file_list = []
while i<len(filenames):
	files = (i,i+steps)
	file_list.append(files)
	i+=steps

k = 0

def verify_files(start,end):
	global k
	files = filenames[start:end]
	for x in files:
		if k%50000==0: print (k)
		try:
				Image.open(x).load()
		except:
			os.remove(x)
		k+=1

thread_list = []
for x in file_list:
	thread = threading.Thread(target=verify_files, args=(x[0],x[1]))
	thread_list.append(thread)
for thread in thread_list:
	thread.start()
for thread in thread_list:
	thread.join()
	print ("Finished Task!")