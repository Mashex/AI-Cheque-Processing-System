import torch
from torch.autograd import Variable
import MICR.utils as MICRutils
import MICR.mydataset as mydataset
from PIL import Image
import numpy as np
import MICR.crnn as crnn
import cv2
import torch.nn.functional as F
import MICR.keys as keys
import MICR.config as config
import numpy as np

alphabet = keys.alphabet_v2
converter = MICRutils.strLabelConverter(alphabet.copy())

model_path = 'MICR/crnn_models/CRNN-1010.pth'

gpu = True
if not torch.cuda.is_available():
	gpu = False
model = crnn.CRNN(config.imgH, 1, len(alphabet) + 1, 256)
if gpu:
	model = model.cuda()

print('loading pretrained model from %s' % model_path)
if gpu:
	model.load_state_dict(torch.load(model_path))
else:
	model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))


def val_on_image(img,model,gpu):
	imgH = config.imgH
	h,w = img.shape[:2]
	imgW = imgH*w//h

	transformer = mydataset.resizeNormalize((imgW, imgH), is_test=True)
	img = cv2.cvtColor( img, cv2.COLOR_BGR2RGB )
	image = Image.fromarray(np.uint8(img)).convert('L')
	image = transformer( image )
	if gpu:
		image = image.cuda()
	image = image.view( 1, *image.size() )
	image = Variable( image )

	model.eval()
	preds = model( image )

	preds = F.log_softmax(preds,2)
	conf, preds = preds.max( 2 )
	preds = preds.transpose( 1, 0 ).contiguous().view( -1 )

	preds_size = Variable( torch.IntTensor( [preds.size( 0 )] ) )
	# raw_pred = converter.decode( preds.data, preds_size.data, raw=True )
	sim_pred = converter.decode( preds.data, preds_size.data, raw=False )#.encode("utf-8")
	return sim_pred

def predict_MICR_code(image):

	text = val_on_image(image,model,gpu).strip()
	return text
