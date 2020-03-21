import glob, os

path = os.getcwd()

filenames = glob.glob('out/*.jpg')
file = open('labels.txt','a+')
for x in filenames:
	folder,filename = os.path.split(x)
	text, idx = filename.split('_')
	file.write(os.path.join(path, x)+"\t"+text+"\n")
file.close()
