import os
import argparse
import pickle
import warnings
import numpy as np

from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger

from models.ssd300 import ssd300
from keras_loss.keras_ssd_loss import SSDLoss
from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation


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
ap.add_argument('-n', '--n_epochs',
                type=int,
                required=True,
                help='Number of epochs (positive integer)')

args = vars(ap.parse_args())
if len(args) < 3:
    raise ValueError(f'You must pass these arguments in command line :\n'
                     f'\t(1) -bb backbone_name [str: "vgg16", "vgg19", '
                     f'"resnet50" or "inception_v3"]\n'
                     f'\t(2) -bs batch_size [int: > 0]\n'
                     f'\t(3) -n n_epochs [int: > 0]')

backbone = args['backbone']
batch_size = args['batch_size']
n_epochs = args['n_epochs']

root_path = f'{os.path.abspath(".")}/image_detection'

# Setup model
#############

img_height = 300
img_width = 300

# Per-channel mean VOC_07+12 trainval : [123, 117, 104]
subtract_mean = [116, 110, 102]  # Per-channel mean of images in the dataset
# (only train dataset).
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

# Instantiate the model
K.clear_session()

ssd = ssd300(n_classes=n_classes,
             backbone=backbone,
             mode='training',
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
             swap_channels=swap_channels)

adam = Adam(lr=1e-03, beta_1=.9, beta_2=.999, epsilon=1e-08, decay=.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.)

ssd.compile(optimizer=adam, loss=ssd_loss.compute_loss)

# Load data
###########

train_dataset = DataGenerator(load_images_into_memory=False,
                              hdf5_dataset_path=f'{root_path}/data/VOC_07+12_train.h5')
val_dataset = DataGenerator(load_images_into_memory=False,
                            hdf5_dataset_path=f'{root_path}/data/VOC_07+12_val.h5')

# Set training parameters
#########################

# For training generator
ssd_data_augmentation = SSDDataAugmentation(img_height=img_height,
                                            img_width=img_width)

# For validation generator
convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=img_height, width=img_width)

# Create an encoder that can encode ground truth labels into
# the format needed by the SSD loss function

predictor_sizes = [
    ssd.get_layer('bb_pred_norm_mbox_conf').output_shape[1:3],
    ssd.get_layer('conv7_mbox_conf').output_shape[1:3],
    ssd.get_layer('conv8_2_mbox_conf').output_shape[1:3],
    ssd.get_layer('conv9_2_mbox_conf').output_shape[1:3],
    ssd.get_layer('conv10_2_mbox_conf').output_shape[1:3],
    ssd.get_layer('conv11_2_mbox_conf').output_shape[1:3]
]

ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_per_layer=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=False,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=.5,
                                    neg_iou_limit=.5,
                                    normalize_coords=True)

# Create generator handles that will be passed
# to Keras `fit_generator()` function

train_generator = train_dataset.generate(
    batch_size=batch_size,
    shuffle=True,
    transformations=[ssd_data_augmentation],
    label_encoder=ssd_input_encoder,
    returns={'processed_images', 'encoded_labels'},
    keep_images_without_gt=False)

val_generator = val_dataset.generate(
    batch_size=batch_size,
    shuffle=False,
    transformations=[convert_to_3_channels, resize],
    label_encoder=ssd_input_encoder,
    returns={'processed_images', 'encoded_labels'},
    keep_images_without_gt=False)

# Get number of samples
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size = val_dataset.get_dataset_size()

print(f'Number of images in training dataset:    {train_dataset_size}')
print(f'Number of images in validation dataset:  {val_dataset_size}')

# Define model callbacks
########################


def lr_schedule(epoch):
    if epoch < 80:
        return 1e-03
    elif epoch < 100:
        return 1e-04
    else:
        return 1e-05


model_cp = ModelCheckpoint(
    filepath=f'{root_path}/results/weights/ssd300_VOC_07+12_{backbone}.h5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode='auto',
    period=1)

csv_logger = CSVLogger(filename=f'{root_path}/results/logs/ssd300_VOC_07+12_{backbone}.csv',
                       separator=',',
                       append=True)

learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule,
                                                verbose=1)

terminate_on_nan = TerminateOnNaN()

callbacks = [model_cp,
             csv_logger,
             learning_rate_scheduler,
             terminate_on_nan]

# Train
#######

history = ssd.fit_generator(
    generator=train_generator,
    steps_per_epoch=np.ceil(train_dataset_size / batch_size),
    epochs=n_epochs,
    callbacks=callbacks,
    validation_data=val_generator,
    validation_steps=np.ceil(val_dataset_size / batch_size),
    initial_epoch=0
)

try:
    with open(f'{root_path}/results/monitoring/ssd300_VOC_07+12_{backbone}_history.p', 'wb') as fp:
        pickle.dump(history.history, fp)
except Exception:
    warnings.warn('Cannot save "history" file')
