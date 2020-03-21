import os
import sys
import json
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa
import glob
import json

# Root directory of the project
ROOT_DIR = os.path.abspath(".")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join("mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join("logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join("results")

VAL_IMAGE_IDS = glob.glob(r'COCO\DIR\annotations\val\*.jpg')
TRAIN_IMAGE_IDS = glob.glob(r'COCO\DIR\annotations\train\*.jpg')


file = open(r'COCO\DIR\annotations\instances_train.json','r')
annotations_train = json.load(file)
train_images = [(x['file_name'],x['id'],x['height'],x['width']) for x in annotations_train['images']]
train_dict = {}
for x in annotations_train['annotations']:
    if x['image_id'] not in train_dict:
        train_dict[x['image_id']] = [[x, train_images[x['image_id']]]]
    else:
        train_dict[x['image_id']].append([x, train_images[x['image_id']]])

file = open(r'C:\Users\masan\Downloads\IDRBT Cheque Image Dataset\COCO\DIR\annotations\instances_val.json','r')
annotations_val = json.load(file)
val_images = [(x['file_name'],x['id'],x['height'],x['width']) for x in annotations_val['images']]
val_dict = {}
for x in annotations_val['annotations']:
    if x['image_id'] not in val_dict:
        val_dict[x['image_id']] = [[x, val_images[x['image_id']]]]
    else:
        val_dict[x['image_id']].append([x, val_images[x['image_id']]])

class ChequeConfig(Config):
    """Configuration for training on the nucleus segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "cheque"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # Background + nucleus

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = len(TRAIN_IMAGE_IDS) // IMAGES_PER_GPU
    VALIDATION_STEPS = len(VAL_IMAGE_IDS) // IMAGES_PER_GPU
     
    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 256
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 28)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400


class ChequeInferenceConfig(ChequeConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

class ChequeDataset(utils.Dataset):

    def load_Cheque(self, dataset_dir, subset):
        """Load a subset of the nuclei dataset.
        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        self.add_class("cheque", 1, "Bank Name")
        self.add_class("cheque", 2, "IFSC")
        self.add_class("cheque", 3, "MICR")
        self.add_class("cheque", 4, "Account Number")
        self.add_class("cheque", 5, "Account Holder Name")
        
        # Add images
        if subset=='train':
            image_ids = train_images
        else:
            image_ids = val_images
        
        for image_loc,image_id,_,_ in image_ids:
            self.add_image(
                "cheque",
                image_id=image_id,
                path=os.path.join(dataset_dir, image_loc))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = train_dict[image_id]
        
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = image_info[0][1]
        mask = np.zeros([info[2], info[3], len(image_info)],
                        dtype=np.uint8)
        
        for i, x in enumerate(image_info):
                mask_img_seg = x[0]['bbox']

                x1 = mask_img_seg[1]
        		y1 = mask_img_seg[0]
        		h = mask_img_seg[2]
        		w = mask_img_seg[3]
                
                mask[y1:y1+h, x1:x1+w, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        class_ids =  [x[0]['category_id'] for x in image_info]
        
        return mask.astype(np.bool), class_ids.astype(np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = train_dict[image_id]
        return info[0][0]["image_id"]


def train(model, dataset_dir):
    """Train the model."""
    # Training dataset.
    dataset_train = ChequeDataset()
    dataset_train.load_Cheque(dataset_dir, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ChequeDataset()
    dataset_val.load_Cheque(dataset_dir, "val")
    dataset_val.prepare()

    
    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                augmentation=augmentation,
                layers='heads')
    
    history = model.keras_model.history.history
    
    model = f"models/{time.time()}.h5"
    model.keras_model.save_weights(model_path)

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                augmentation=augmentation,
                layers='all')


def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = NucleusDataset()
    dataset.load_Cheque(dataset_dir, subset)
    dataset.prepare()
    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)
        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=False, show_mask=False,
            title="Predictions")
        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))

    # Save to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)


command = "train"
# Configurations
if command == "train":
    config = ChequeConfig()
else:
    config = ChequeInferenceConfig()
config.display()


# Create model
if command == "train":
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir='models')
else:
    model = modellib.MaskRCNN(mode="inference", config=config,
                              model_dir='models')

weights = "coco"
# Select weights file to load
if weights.lower() == "coco":
    weights_path = COCO_WEIGHTS_PATH
elif weights.lower() == "last":
    # Find last trained weights
    weights_path = model.find_last()
elif weights.lower() == "imagenet":
    # Start from ImageNet trained weights
    weights_path = model.get_imagenet_weights()
else:
    weights_path = weights

# Load weights
print("Loading weights ", weights_path)
if weights.lower() == "coco":
    # Exclude the last layers because they require a matching
    # number of classes
    model.load_weights(weights_path, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])
else:
    model.load_weights(weights_path, by_name=True)

# Train or evaluate
dataset_dir = r"COCO\DIR\annotations"
dataset = dataset_dir
if command == "train":
    train(model, dataset)
elif command == "detect":
    detect(model, args.dataset, subset)
else:
    print("'{}' is not recognized. "
          "Use 'train' or 'detect'".format(args.command))

