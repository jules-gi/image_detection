import os
import argparse
from data_generator.object_detection_2d_data_generator import DataGenerator

# Get arguments
ap = argparse.ArgumentParser()
ap.add_argument('-n', '--name',
                nargs='*',
                type=str,
                required=True,
                help='The name of datasets to convert: '
                     '"train", "valid" or "test"')

args = vars(ap.parse_args())
datasets = args['name']

root_path = f'{os.path.abspath(".")}/image_detection'

# Directories containing images
VOC2007_images_dir = f'{root_path}/data/VOC2007/JPEGImages/'
VOC2012_images_dir = f'{root_path}/data/VOC2012/JPEGImages/'

# Directories containing annotations
VOC2007_annotations_dir = f'{root_path}/data/VOC2007/Annotations/'
VOC2012_annotations_dir = f'{root_path}/data/VOC2012/Annotations/'

# Paths to the image sets
VOC2007_image_set_train = f'{root_path}/data/VOC2007/ImageSets/Main/train.txt'
VOC2012_image_set_train = f'{root_path}/data/VOC2012/ImageSets/Main/train.txt'
VOC2007_image_set_val = f'{root_path}/data/VOC2007/ImageSets/Main/val.txt'
VOC2012_image_set_val = f'{root_path}/data/VOC2012/ImageSets/Main/val.txt'
VOC2007_image_set_trainval = f'{root_path}/data/VOC2007/ImageSets/Main/trainval.txt'
VOC2012_image_set_trainval = f'{root_path}/data/VOC2012/ImageSets/Main/trainval.txt'
VOC2007_image_set_test = f'{root_path}/data/VOC2007/ImageSets/Main/test.txt'
VOC2012_image_set_test = f'{root_path}/data/VOC2012/ImageSets/Main/test.txt'

# XML parser need to know what object class names to look for
# and in which order to map them to integers
classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

if 'train' in datasets:
    train_dataset = DataGenerator(load_images_into_memory=False,
                                  hdf5_dataset_path=None)

    train_dataset.parse_xml(images_dirs=[VOC2007_images_dir,
                                         VOC2012_images_dir],
                            image_set_filenames=[VOC2007_image_set_train,
                                                 VOC2012_image_set_train],
                            annotations_dirs=[VOC2007_annotations_dir,
                                              VOC2012_annotations_dir],
                            classes=classes,
                            include_classes='all',
                            exclude_truncated=False,
                            exclude_difficult=False,
                            ret=False)

    train_dataset.create_hdf5_dataset(file_path=f'{root_path}/data/VOC_07+12_train.h5',
                                      resize=False,
                                      variable_image_size=True,
                                      verbose=True)

if 'valid' in datasets:
    val_dataset = DataGenerator(load_images_into_memory=False,
                                hdf5_dataset_path=None)

    val_dataset.parse_xml(images_dirs=[VOC2007_images_dir,
                                       VOC2012_images_dir],
                          image_set_filenames=[VOC2007_image_set_val,
                                               VOC2012_image_set_val],
                          annotations_dirs=[VOC2007_annotations_dir,
                                            VOC2012_annotations_dir],
                          classes=classes,
                          include_classes='all',
                          exclude_truncated=False,
                          exclude_difficult=True,
                          ret=False)

    val_dataset.create_hdf5_dataset(file_path=f'{root_path}/data/VOC_07+12_val.h5',
                                    resize=False,
                                    variable_image_size=True,
                                    verbose=True)

if 'test' in datasets:
    test_dataset = DataGenerator(load_images_into_memory=False,
                                 hdf5_dataset_path=None)

    test_dataset.parse_xml(images_dirs=[VOC2007_images_dir,
                                        VOC2012_images_dir],
                           image_set_filenames=[VOC2007_image_set_test,
                                                VOC2012_image_set_test],
                           annotations_dirs=[VOC2007_annotations_dir,
                                             VOC2012_annotations_dir],
                           classes=classes,
                           include_classes='all',
                           exclude_truncated=False,
                           exclude_difficult=False,
                           ret=False)

    test_dataset.create_hdf5_dataset(file_path=f'{root_path}/data/VOC_07+12_test.h5',
                                     resize=False,
                                     variable_image_size=True,
                                     verbose=True)
