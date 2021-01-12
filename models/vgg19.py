from keras.layers import Conv2D, MaxPooling2D
from keras.regularizers import l2
from keras.models import Model
from keras.applications import VGG19


def load(inputs, l2_regularization=5e-4):
    x, x1 = inputs

    # Block 1
    conv1_1 = Conv2D(64, (3, 3),
                     activation='relu',
                     padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_regularization),
                     name='block1_conv1')(x1)
    conv1_2 = Conv2D(64, (3, 3),
                     activation='relu',
                     padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_regularization),
                     name='block1_conv2')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2),
                         strides=(2, 2),
                         padding='same',
                         name='block1_pool')(conv1_2)

    # Block 2
    conv2_1 = Conv2D(128, (3, 3),
                     activation='relu',
                     padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_regularization),
                     name='block2_conv1')(pool1)
    conv2_2 = Conv2D(128, (3, 3),
                     activation='relu',
                     padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_regularization),
                     name='block2_conv2')(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2),
                         strides=(2, 2),
                         padding='same',
                         name='block2_pool')(conv2_2)

    # Block 3
    conv3_1 = Conv2D(256, (3, 3),
                     activation='relu',
                     padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_regularization),
                     name='block3_conv1')(pool2)
    conv3_2 = Conv2D(256, (3, 3),
                     activation='relu',
                     padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_regularization),
                     name='block3_conv2')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3),
                     activation='relu',
                     padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_regularization),
                     name='block3_conv3')(conv3_2)
    conv3_4 = Conv2D(256, (3, 3),
                     activation='relu',
                     padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_regularization),
                     name='block3_conv4')(conv3_3)
    pool3 = MaxPooling2D(pool_size=(2, 2),
                         strides=(2, 2),
                         padding='same',
                         name='block3_pool')(conv3_4)

    # Block 4
    conv4_1 = Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_regularization),
                     name='block4_conv1')(pool3)
    conv4_2 = Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_regularization),
                     name='block4_conv2')(conv4_1)
    conv4_3 = Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_regularization),
                     name='block4_conv3')(conv4_2)
    conv4_4 = Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_regularization),
                     name='block4_conv4')(conv4_3)
    pool4 = MaxPooling2D(pool_size=(2, 2),
                         strides=(2, 2),
                         padding='same',
                         name='block4_pool')(conv4_4)

    # Block 5
    conv5_1 = Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_regularization),
                     name='block5_conv1')(pool4)
    conv5_2 = Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_regularization),
                     name='block5_conv2')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_regularization),
                     name='block5_conv3')(conv5_2)
    conv5_4 = Conv2D(512, (3, 3),
                     activation='relu',
                     padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_regularization),
                     name='block5_conv4')(conv5_3)
    pool5 = MaxPooling2D(pool_size=(3, 3),
                         strides=(1, 1),
                         padding='same',
                         name='block5_pool')(conv5_4)

    net = Model(inputs=x, outputs=pool5)

    # Get VGG19 weights
    base_net = VGG19(include_top=False,
                     weights='imagenet',
                     input_tensor=x)
    weights = base_net.get_weights()
    weights = {
        'block1_conv1': [weights[0], weights[1]],
        'block1_conv2': [weights[2], weights[3]],
        'block2_conv1': [weights[4], weights[5]],
        'block2_conv2': [weights[6], weights[7]],
        'block3_conv1': [weights[8], weights[9]],
        'block3_conv2': [weights[10], weights[11]],
        'block3_conv3': [weights[12], weights[13]],
        'block3_conv4': [weights[14], weights[15]],
        'block4_conv1': [weights[16], weights[17]],
        'block4_conv2': [weights[18], weights[19]],
        'block4_conv3': [weights[20], weights[21]],
        'block4_conv4': [weights[22], weights[23]],
        'block5_conv1': [weights[24], weights[25]],
        'block5_conv2': [weights[26], weights[27]],
        'block5_conv3': [weights[28], weights[29]],
        'block5_conv4': [weights[30], weights[31]],
    }

    # Set weights to customized layers
    for layer_name, layer_weights in weights.items():
        net.get_layer(layer_name).set_weights(layer_weights)

    return net
