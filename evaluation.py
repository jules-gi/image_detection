import os
import warnings
import argparse
import pickle

from keras import backend as K
from keras.optimizers import Adam
import numpy as np

from models.ssd300 import ssd300
from keras_loss.keras_ssd_loss import SSDLoss
from data_generator.object_detection_2d_data_generator import DataGenerator
from eval_utils.average_precision_evaluator import Evaluator


# Get arguments
ap = argparse.ArgumentParser()
ap.add_argument('-b', '--backbone',
                type=str,
                required=True,
                help='The name of the backbone')
ap.add_argument('-s', '--batch_size',
                type=int,
                required=True,
                help='Size of the batch (positive integer)')

args = vars(ap.parse_args())
if len(args) < 2:
    raise ValueError(f'You must pass these arguments in command line :\n'
                     f'\t(1) -bb backbone_name [str: "vgg16", "vgg19", '
                     f'"resnet50" or "inception_v3"]\n'
                     f'\t(2) -bs batch_size [int: > 0]')

backbone = args['backbone']
batch_size = args['batch_size']

root_path = f'{os.path.abspath(".")}/image_detection'

# Setup model
#############

img_height = 300
img_width = 300

# Per-channel mean VOC_07+12 trainval : [123, 117, 104]
subtract_mean = [116, 110, 102]  # Per-channel mean of images in the dataset
# (only train).
divide_by_std = None
swap_channels = [2, 1, 0]  # In the original SSD, channel order is BGR

l2_regularization = 5e-4
n_classes = 20
min_scale = .2
max_scale = .9
aspect_ratios = [[1., 2., .5],  # Aspect ratios in original SSD300
                 [1., 2., .5, 3., 1. / 3.],
                 [1., 2., .5, 3., 1. / 3.],
                 [1., 2., .5, 3., 1. / 3.],
                 [1., 2., .5],
                 [1., 2., .5]]
two_boxes_for_ar1 = True  # Add a second default box if aspect ratio contains 1
variances = [.1, .1, .2, .2]  # Divide encoded target coords by this variance
steps = [8, 16, 32, 64, 100, 300]  # Space between 2 adjacent anchor box
offsets = [.5, .5, .5, .5, .5, .5]  # Offsets of the first anchor box center
# points from the top and left borders of the image as a fraction of the step
# size for each predictor layer
scales = list(np.linspace(min_scale, max_scale, len(aspect_ratios) + 1))

# Inference parameters
confidence_thresh = .01
iou_threshold = .45
top_k = 200
nms_max_output_size = 400

# Load model
K.clear_session()

ssd = ssd300(n_classes=n_classes,
             backbone=backbone,
             mode='inference',
             l2_regularization=l2_regularization,
             min_scale=min_scale,
             max_scale=max_scale,
             aspect_ratios_per_layer=aspect_ratios,
             two_boxes_for_ar1=two_boxes_for_ar1,
             steps=steps,
             offsets=offsets,
             variances=variances,
             subtract_mean=subtract_mean,
             divide_by_std=divide_by_std,
             swap_channels=swap_channels,
             confidence_thresh=confidence_thresh,
             iou_threshold=iou_threshold,
             top_k=top_k,
             nms_max_output_size=nms_max_output_size)

weights_path = f'{root_path}/results/weights/ssd300_VOC_07+12_{backbone}.h5'
ssd.load_weights(weights_path, by_name=True)

adam = Adam(lr=1e-03, beta_1=.9, beta_2=.999, epsilon=1e-08, decay=.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.)

ssd.compile(optimizer=adam, loss=ssd_loss.compute_loss)


# Load data
###########
test_dataset = DataGenerator(load_images_into_memory=False,
                             hdf5_dataset_path=f'{root_path}/data/VOC_07+12_test.h5')


evaluator = Evaluator(model=ssd,
                      n_classes=n_classes,
                      data_generator=test_dataset,
                      model_mode='inference')

results = evaluator(img_height=img_height,
                    img_width=img_width,
                    batch_size=batch_size,
                    data_generator_mode='resize',
                    round_confidences=False,
                    matching_iou_threshold=.5,
                    border_pixels='include',
                    sorting_algorithm='quicksort',
                    average_precision_mode='sample',
                    num_recall_points=11,
                    ignore_neutral_boxes=True,
                    return_precisions=True,
                    return_recalls=True,
                    return_average_precisions=True,
                    verbose=True)

mean_average_precision, average_precisions, precisions, recalls = results

average_precisions = list(average_precisions[1:])

try:
    with open(f'{root_path}/results/evaluation/measures.p', 'rb') as fp:
        data = pickle.load(fp)
except Exception:
    data = {}

data[backbone] = {
    'average_precisions': average_precisions,
    'mean_average_precision': mean_average_precision,
}

with open(f'{root_path}/results/evaluation/measures.p', 'wb') as fp:
    pickle.dump(data, fp)


try:
    precisions = np.array(precisions[1:])
    recalls = np.array(recalls[1:])

    np.savez(f'{root_path}/results/evaluation/precisionsrecalls{backbone}.npz',
             precisions, recalls)

except Exception:
    warnings.warn('Cannot convert precisions and recalls list to array.')
