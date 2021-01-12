from keras.regularizers import l2
from keras.models import Model
from keras.applications import InceptionV3


def load(inputs, l2_regularization=5e-4):
    x, x1 = inputs

    # Get InceptionV3 model
    base_net = InceptionV3(include_top=False,
                           weights='imagenet',
                           input_tensor=x1)

    net = Model(inputs=x, outputs=base_net.get_layer('mixed7').output)

    # Add kernel regularizer to each layer
    regularizer = l2(l2_regularization)
    for layer in net.layers:
        if hasattr(layer, 'kernel_regularizer'):
            setattr(layer, 'kernel_regularizer', regularizer)

    return net
