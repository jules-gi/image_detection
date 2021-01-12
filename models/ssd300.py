"""
SSD300 based on Pierluigi Ferrari implementation:
https://github.com/pierluigiferrari/ssd_keras
"""

import numpy as np

from keras.layers import Input, Lambda, Conv2D, ZeroPadding2D, Reshape, Concatenate, Activation
from keras.regularizers import l2
from keras.models import Model
import keras.backend as K

from keras_layers.keras_layer_L2Normalization import L2Normalization
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections

from models import vgg16, vgg19, resnet50, inception_v3


def load_backbone(name,
                  input_shape,
                  l2_regularization=5e-4,
                  subtract_mean=None,
                  divide_by_std=None,
                  swap_channels=None):

    # Define functions for Lambda layers
    def identity_layer(tensor):
        return tensor

    def input_mean_normalization(tensor):
        return tensor - np.array(subtract_mean)

    def input_std_normalization(tensor):
        return tensor / np.array(divide_by_std)

    def input_channel_swap(tensor):
        return K.stack([tensor[..., swap_channels[i]]
                        for i in range(len(swap_channels))], axis=-1)

    x = Input(shape=input_shape)
    x1 = Lambda(identity_layer,
                output_shape=input_shape,
                name='identity_layer')(x)
    if subtract_mean:
        x1 = Lambda(input_mean_normalization,
                    output_shape=input_shape,
                    name='input_mean_normalization')(x1)
    if divide_by_std:
        x1 = Lambda(input_std_normalization,
                    output_shape=input_shape,
                    name='input_std_normalization')(x1)
    if swap_channels:
        x1 = Lambda(input_channel_swap,
                    output_shape=input_shape,
                    name='input_channel_swap')(x1)

    if name == 'vgg16':
        net = vgg16.load(inputs=(x, x1), l2_regularization=l2_regularization)
    elif name == 'vgg19':
        net = vgg19.load(inputs=(x, x1), l2_regularization=l2_regularization)
    elif name == 'resnet50':
        net = resnet50.load(inputs=(x, x1), l2_regularization=l2_regularization)
    elif name == 'inception_v3':
        net = inception_v3.load(inputs=(x, x1), l2_regularization=l2_regularization)
    else:
        raise ImportError(f'Backbone \'{name}\' is not implemented. '
                          f'List of valid backbone names: '
                          f'[\'vgg16\', \'vgg19\', \'resnet50\', '
                          f'\'inception_v3\']')

    return net


def ssd300(n_classes,
           backbone,
           mode='training',
           use_bb_layer_pred=True,
           input_shape=(300, 300, 3),
           return_predictor_sizes=False,
           l2_regularization=5e-4,
           min_scale=.2,
           max_scale=.9,
           scales=None,
           aspect_ratios_per_layer=None,
           aspect_ratios_global=None,
           two_boxes_for_ar1=True,
           steps=None,
           offsets=None,
           clip_boxes=False,
           variances=None,
           coords='centroids',
           normalize_coords=True,
           subtract_mean=None,
           divide_by_std=None,
           swap_channels=None,
           confidence_thresh=.01,
           iou_threshold=.45,
           top_k=200,
           nms_max_output_size=400):

    n_predictor_layers = 6  # Number of predictors conv
    n_classes += 1  # Anchor for the background class
    l2_reg = l2_regularization  # Make the internal name shorter
    img_height, img_width, img_channels = input_shape

    ##################################
    # Set exceptions or default values
    ##################################

    if aspect_ratios_per_layer is None and aspect_ratios_global is None:
        aspect_ratios_per_layer = [[1., 2., .5],
                                   [1., 2., .5, 3., 1./3.],
                                   [1., 2., .5, 3., 1./3.],
                                   [1., 2., .5, 3., 1./3.],
                                   [1., 2., .5],
                                   [1., 2., .5]]
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError(f'It must be either aspect_ratios is None or '
                             f'len(aspect_ratios_per_layer) == '
                             f'{n_predictor_layers}, but len(aspect_ratios_per'
                             f'_layer) == {len(aspect_ratios_per_layer)}.')

    if scales:
        if len(scales) != n_predictor_layers + 1:
            raise ValueError(f'It must be either scales is None or len(scales)'
                             f' == {n_predictor_layers + 1}, but len(scales) '
                             f'== {len(scales)}.')
        scales = np.array(scales)
    else:
        scales = np.linspace(min_scale, max_scale, n_predictor_layers + 1)

    if variances is None:
        variances = [.1, .1, .2, .2]
    else:
        if len(variances) != 4:
            raise ValueError(f'4 variance values must be pased, '
                             f'but {len(variances)} values were received.')
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError(f'All variances must be >0, '
                         f'but the variances given are {variances}')

    if steps is None:
        steps = [8, 16, 32, 64, 100, 300]
    else:
        if len(steps) != n_predictor_layers:
            raise ValueError('You must provide 4 positive float.')

    if offsets is None:
        offsets = [.5] * n_predictor_layers
    else:
        if len(offsets) != n_predictor_layers:
            raise ValueError('You must provide at least '
                             'one offset value per predictor layer.')

    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
        n_boxes = []
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:
                n_boxes.append(len(ar) + 1)
            else:
                n_boxes.append(len(ar))
    else:
        aspect_ratios = aspect_ratios_global
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers

    #################
    # Build the model
    #################

    base_model = load_backbone(name=backbone,
                               input_shape=input_shape,
                               l2_regularization=l2_reg,
                               subtract_mean=subtract_mean,
                               divide_by_std=divide_by_std,
                               swap_channels=swap_channels)

    base_output = base_model.get_layer(index=-1).output
    if backbone == 'vgg16':
        bb_pred = base_model.get_layer(name='block4_conv3').output
    elif backbone == 'vgg19':
        bb_pred = base_model.get_layer(name='block4_conv4').output
    elif backbone == 'resnet50':
        bb_pred = base_model.get_layer(name='activation_22').output
    elif backbone == 'inception_v3':
        bb_pred = base_model.get_layer(name='mixed2').output

    # Block full Conv 6 and 7
    conv6 = Conv2D(1024, (3, 3),
                   dilation_rate=(6, 6),
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(l2_reg),
                   name='conv6')(base_output)
    conv7 = Conv2D(1024, (1, 1),
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(l2_reg),
                   name='conv7')(conv6)

    # Block 8
    conv8_1 = Conv2D(256, (1, 1),
                     activation='relu',
                     padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg),
                     name='conv8_1')(conv7)
    conv8_1 = ZeroPadding2D(padding=((1, 1), (1, 1)),
                            name='conv8_padding')(conv8_1)
    conv8_2 = Conv2D(512, (3, 3),
                     strides=(2, 2),
                     activation='relu',
                     padding='valid',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg),
                     name='conv8_2')(conv8_1)

    # Block 9
    conv9_1 = Conv2D(128, (1, 1),
                     activation='relu',
                     padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg),
                     name='conv9_1')(conv8_2)
    conv9_1 = ZeroPadding2D(padding=((1, 1), (1, 1)),
                            name='conv9_padding')(conv9_1)
    conv9_2 = Conv2D(256, (3, 3),
                     strides=(2, 2),
                     activation='relu',
                     padding='valid',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg),
                     name='conv9_2')(conv9_1)

    # Block 10
    conv10_1 = Conv2D(128, (1, 1),
                      activation='relu',
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_reg),
                      name='conv10_1')(conv9_2)
    conv10_2 = Conv2D(256, (3, 3),
                      strides=(1, 1),
                      activation='relu',
                      padding='valid',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_reg),
                      name='conv10_2')(conv10_1)

    # Block 11
    conv11_1 = Conv2D(128, (1, 1),
                      activation='relu',
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_reg),
                      name='conv11_1')(conv10_2)

    conv11_2 = Conv2D(256, (3, 3),
                      strides=(1, 1),
                      activation='relu',
                      padding='valid',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_reg),
                      name='conv11_2')(conv11_1)

    # Feed bb_pred into L2Normalization layer
    if use_bb_layer_pred:
        bb_pred_norm = L2Normalization(gamma_init=20,
                                       name='bb_pred_norm')(bb_pred)

    # Build convolutional predictor layers on top of the base network
    # Output shape of confidence: `(batch, height, width, n_boxes * n_classes)`

    if use_bb_layer_pred:
        bb_pred_norm_mbox_conf = \
            Conv2D(n_boxes[0] * n_classes, (3, 3),
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(l2_reg),
                   name='bb_pred_norm_mbox_conf')(bb_pred_norm)
    conv7_mbox_conf = Conv2D(n_boxes[1] * n_classes, (3, 3),
                             padding='same',
                             kernel_initializer='he_normal',
                             kernel_regularizer=l2(l2_reg),
                             name='conv7_mbox_conf')(conv7)
    conv8_2_mbox_conf = Conv2D(n_boxes[2] * n_classes, (3, 3),
                               padding='same',
                               kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_reg),
                               name='conv8_2_mbox_conf')(conv8_2)
    conv9_2_mbox_conf = Conv2D(n_boxes[3] * n_classes, (3, 3),
                               padding='same',
                               kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_reg),
                               name='conv9_2_mbox_conf')(conv9_2)
    conv10_2_mbox_conf = Conv2D(n_boxes[4] * n_classes, (3, 3),
                                padding='same',
                                kernel_initializer='he_normal',
                                kernel_regularizer=l2(l2_reg),
                                name='conv10_2_mbox_conf')(conv10_2)
    conv11_2_mbox_conf = Conv2D(n_boxes[5] * n_classes, (3, 3),
                                padding='same',
                                kernel_initializer='he_normal',
                                kernel_regularizer=l2(l2_reg),
                                name='conv11_2_mbox_conf')(conv11_2)

    # Predict 4 box coordinates for each box
    # Output shape of localization: `(batch, height, width, n_boxes * 4)`

    if use_bb_layer_pred:
        bb_pred_norm_mbox_loc = Conv2D(n_boxes[0] * 4, (3, 3),
                                       padding='same',
                                       kernel_initializer='he_normal',
                                       kernel_regularizer=l2(l2_reg),
                                       name='bb_pred_norm_mbox_loc')(bb_pred_norm)
    conv7_mbox_loc = Conv2D(n_boxes[1] * 4, (3, 3),
                            padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(l2_reg),
                            name='conv7_mbox_loc')(conv7)
    conv8_2_mbox_loc = Conv2D(n_boxes[2] * 4, (3, 3),
                              padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg),
                              name='conv8_2_mbox_loc')(conv8_2)
    conv9_2_mbox_loc = Conv2D(n_boxes[3] * 4, (3, 3),
                              padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg),
                              name='conv9_2_mbox_loc')(conv9_2)
    conv10_2_mbox_loc = Conv2D(n_boxes[4] * 4, (3, 3),
                               padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_reg),
                               name='conv10_2_mbox_loc')(conv10_2)
    conv11_2_mbox_loc = Conv2D(n_boxes[5] * 4, (3, 3),
                               padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_reg),
                               name='conv11_2_mbox_loc')(conv11_2)

    # Generate anchor boxes
    # Output shape of anchors: `(batch, height, width, n_boxes, 8)`

    if use_bb_layer_pred:
        bb_pred_norm_mbox_anchor = \
            AnchorBoxes(img_height, img_width,
                        this_scale=scales[0], next_scale=scales[1],
                        aspect_ratios=aspect_ratios[0],
                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0],
                        this_offsets=offsets[0], clip_boxes=clip_boxes,
                        variances=variances, coords=coords,
                        normalize_coords=normalize_coords,
                        name='bb_pred_norm_mbox_anchor')(bb_pred_norm_mbox_loc)
    conv7_mbox_anchor = \
        AnchorBoxes(img_height, img_width,
                    this_scale=scales[1], next_scale=scales[2],
                    aspect_ratios=aspect_ratios[1],
                    two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1],
                    this_offsets=offsets[1], clip_boxes=clip_boxes,
                    variances=variances, coords=coords,
                    normalize_coords=normalize_coords,
                    name='conv7_mbox_anchor')(conv7_mbox_loc)
    conv8_2_mbox_anchor = \
        AnchorBoxes(img_height, img_width,
                    this_scale=scales[2], next_scale=scales[3],
                    aspect_ratios=aspect_ratios[2],
                    two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2],
                    this_offsets=offsets[2], clip_boxes=clip_boxes,
                    variances=variances, coords=coords,
                    normalize_coords=normalize_coords,
                    name='conv8_2_mbox_anchor')(conv8_2_mbox_loc)
    conv9_2_mbox_anchor = \
        AnchorBoxes(img_height, img_width,
                    this_scale=scales[3], next_scale=scales[4],
                    aspect_ratios=aspect_ratios[3],
                    two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3],
                    this_offsets=offsets[3], clip_boxes=clip_boxes,
                    variances=variances, coords=coords,
                    normalize_coords=normalize_coords,
                    name='conv9_2_mbox_anchor')(conv9_2_mbox_loc)
    conv10_2_mbox_anchor = \
        AnchorBoxes(img_height, img_width,
                    this_scale=scales[4], next_scale=scales[5],
                    aspect_ratios=aspect_ratios[4],
                    two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[4],
                    this_offsets=offsets[4], clip_boxes=clip_boxes,
                    variances=variances, coords=coords,
                    normalize_coords=normalize_coords,
                    name='conv10_2_mbox_anchor')(conv10_2_mbox_loc)
    conv11_2_mbox_anchor = \
        AnchorBoxes(img_height, img_width,
                    this_scale=scales[5], next_scale=scales[6],
                    aspect_ratios=aspect_ratios[5],
                    two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[5],
                    this_offsets=offsets[5], clip_boxes=clip_boxes,
                    variances=variances, coords=coords,
                    normalize_coords=normalize_coords,
                    name='conv11_2_mbox_anchor')(conv11_2_mbox_loc)

    # Reshape
    #########

    # Reshape class predictions, yielding 3D tensor of shape
    #   `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    if use_bb_layer_pred:
        bb_pred_norm_mbox_conf_reshape = \
            Reshape((-1, n_classes),
                    name='bb_pred_norm_mbox_conf_reshape')(bb_pred_norm_mbox_conf)
    conv7_mbox_conf_reshape = \
        Reshape((-1, n_classes),
                name='conv7_mbox_conf_reshape')(conv7_mbox_conf)
    conv8_2_mbox_conf_reshape = \
        Reshape((-1, n_classes),
                name='conv8_2_mbox_conf_reshape')(conv8_2_mbox_conf)
    conv9_2_mbox_conf_reshape = \
        Reshape((-1, n_classes),
                name='conv9_2_mbox_conf_reshape')(conv9_2_mbox_conf)
    conv10_2_mbox_conf_reshape = \
        Reshape((-1, n_classes),
                name='conv10_2_mbox_conf_reshape')(conv10_2_mbox_conf)
    conv11_2_mbox_conf_reshape = \
        Reshape((-1, n_classes),
                name='conv11_2_mbox_conf_reshape')(conv11_2_mbox_conf)

    # Reshape box predictions, yielding 3D tensor of shape
    #   `(batch, height * width * n_boxes, 4)`
    # We want 4 boxes coordinates isolated in the last axis
    # to compute the smooth L1 loss
    if use_bb_layer_pred:
        bb_pred_norm_mbox_loc_reshape = \
            Reshape((-1, 4),
                    name='bb_pred_norm_mbox_loc_reshape')(bb_pred_norm_mbox_loc)
    conv7_mbox_loc_reshape = \
        Reshape((-1, 4),
                name='conv7_mbox_loc_reshape')(conv7_mbox_loc)
    conv8_2_mbox_loc_reshape = \
        Reshape((-1, 4),
                name='conv8_2_mbox_loc_reshape')(conv8_2_mbox_loc)
    conv9_2_mbox_loc_reshape = \
        Reshape((-1, 4),
                name='conv9_2_mbox_loc_reshape')(conv9_2_mbox_loc)
    conv10_2_mbox_loc_reshape = \
        Reshape((-1, 4),
                name='conv10_2_mbox_loc_reshape')(conv10_2_mbox_loc)
    conv11_2_mbox_loc_reshape = \
        Reshape((-1, 4),
                name='conv11_2_mbox_loc_reshape')(conv11_2_mbox_loc)

    # Reshape anchor box tensors, yielding 3D tensor of shape
    #   `(batch, height * width * n_boxes, 8)`

    if use_bb_layer_pred:
        bb_pred_norm_mbox_anchor_reshape = Reshape(
            (-1, 8),
            name='bb_pred_norm_mbox_anchor_reshape')(bb_pred_norm_mbox_anchor)
    conv7_mbox_anchor_reshape = \
        Reshape((-1, 8),
                name='conv7_mbox_anchor_reshape')(conv7_mbox_anchor)
    conv8_2_mbox_anchor_reshape = \
        Reshape((-1, 8),
                name='conv8_2_mbox_anchor_reshape')(conv8_2_mbox_anchor)
    conv9_2_mbox_anchor_reshape = \
        Reshape((-1, 8),
                name='conv9_2_mbox_anchor_reshape')(conv9_2_mbox_anchor)
    conv10_2_mbox_anchor_reshape = \
        Reshape((-1, 8),
                name='conv10_2_mbox_anchor_reshape')(conv10_2_mbox_anchor)
    conv11_2_mbox_anchor_reshape = \
        Reshape((-1, 8),
                name='conv11_2_mbox_anchor_reshape')(conv11_2_mbox_anchor)

    # Concatenate
    #############

    # Concatenate predictions from different layers
    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical
    # for all layers predictions, so we want to concatenate along axis 1
    # (the number of boxes per layer)

    if use_bb_layer_pred:
        # Output shape for `mbox_conf`: `(batch, n_boxes_total, n_classes)`
        mbox_conf = Concatenate(axis=1, name='mbox_conf')(
            [bb_pred_norm_mbox_conf_reshape, conv7_mbox_conf_reshape,
             conv8_2_mbox_conf_reshape, conv9_2_mbox_conf_reshape,
             conv10_2_mbox_conf_reshape, conv11_2_mbox_conf_reshape]
        )

        # Output shape for `mbox_loc`: `(batch, n_boxes_total, 4)`
        mbox_loc = Concatenate(axis=1, name='mbox_loc')(
            [bb_pred_norm_mbox_loc_reshape, conv7_mbox_loc_reshape,
             conv8_2_mbox_loc_reshape, conv9_2_mbox_loc_reshape,
             conv10_2_mbox_loc_reshape, conv11_2_mbox_loc_reshape]
        )

        # Output shape for `mbox_anchor`: `(batch, n_boxes_total, 8)`
        mbox_anchor = Concatenate(axis=1, name='mbox_anchor')(
            [bb_pred_norm_mbox_anchor_reshape, conv7_mbox_anchor_reshape,
             conv8_2_mbox_anchor_reshape, conv9_2_mbox_anchor_reshape,
             conv10_2_mbox_anchor_reshape, conv11_2_mbox_anchor_reshape]
        )
    else:
        # Output shape for `mbox_conf`: `(batch, n_boxes_total, n_classes)`
        mbox_conf = Concatenate(axis=1, name='mbox_conf')(
            [conv7_mbox_conf_reshape,
             conv8_2_mbox_conf_reshape, conv9_2_mbox_conf_reshape,
             conv10_2_mbox_conf_reshape, conv11_2_mbox_conf_reshape]
        )

        # Output shape for `mbox_loc`: `(batch, n_boxes_total, 4)`
        mbox_loc = Concatenate(axis=1, name='mbox_loc')(
            [conv7_mbox_loc_reshape,
             conv8_2_mbox_loc_reshape, conv9_2_mbox_loc_reshape,
             conv10_2_mbox_loc_reshape, conv11_2_mbox_loc_reshape]
        )

        # Output shape for `mbox_anchor`: `(batch, n_boxes_total, 8)`
        mbox_anchor = Concatenate(axis=1, name='mbox_anchor')(
            [conv7_mbox_anchor_reshape,
             conv8_2_mbox_anchor_reshape, conv9_2_mbox_anchor_reshape,
             conv10_2_mbox_anchor_reshape, conv11_2_mbox_anchor_reshape]
        )

    # The box coordinates predictions will go into the loss function
    # just the way they are, but for the class prediction,
    # we'll apply a softmax activation layer first
    mbox_conf_softmax = Activation('softmax',
                                   name='mbox_conf_softmax')(mbox_conf)

    # Concatenate the class, box predictions and anchors to
    # one large predictions vector
    # Output shape of predictions: `(batch, n_boxes_total, n_classes + 4 + 8)`
    predictions = Concatenate(axis=2, name='predictions')(
        [mbox_conf_softmax, mbox_loc, mbox_anchor]
    )

    if mode == 'training':
        model = Model(inputs=base_model.inputs, outputs=predictions)
    elif mode == 'inference':
        decoded_predictions = \
            DecodeDetections(confidence_thresh=confidence_thresh,
                             iou_threshold=iou_threshold,
                             top_k=top_k,
                             nms_max_output_size=nms_max_output_size,
                             coords=coords,
                             normalize_coords=normalize_coords,
                             img_height=img_height,
                             img_width=img_width,
                             name='decoded_predictions')(predictions)
        model = Model(inputs=base_model.inputs, outputs=decoded_predictions)
    else:
        ValueError(f'Argument "mode" must be "training" or "inference", '
                   f'not {mode}')

    if return_predictor_sizes:
        if use_bb_layer_pred:
            predictor_sizes = np.array([bb_pred_norm_mbox_conf.shape[1:3],
                                        conv7_mbox_conf.shape[1:3],
                                        conv8_2_mbox_conf.shape[1:3],
                                        conv9_2_mbox_conf.shape[1:3],
                                        conv10_2_mbox_conf.shape[1:3],
                                        conv11_2_mbox_conf.shape[1:3]])
        else:
            predictor_sizes = np.array([conv7_mbox_conf.shape[1:3],
                                        conv8_2_mbox_conf.shape[1:3],
                                        conv9_2_mbox_conf.shape[1:3],
                                        conv10_2_mbox_conf.shape[1:3],
                                        conv11_2_mbox_conf.shape[1:3]])

        return model, predictor_sizes

    return model
