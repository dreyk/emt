import models.fc_densenet.layers as fc_layers
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow as tf

def FCDensNet(
        input_shape=(None, None, 3),
        n_classes=1,
        n_filters_first_conv=48,
        n_pool=5,
        growth_rate=16,
        n_layers_per_block=[4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4],
        dropout_p=0.2
):
    if type(n_layers_per_block) == list:
        print(len(n_layers_per_block))
    elif type(n_layers_per_block) == int:
        n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)
    else:
        raise ValueError

    #####################
    # First Convolution #
    #####################
    inputs = layers.Input(shape=input_shape)
    tf.summary.image('src',inputs, max_outputs=3)
    stack = layers.Conv2D(filters=n_filters_first_conv, kernel_size=3, padding='same', kernel_initializer='he_uniform')(inputs)
    n_filters = n_filters_first_conv

    #####################
    # Downsampling path #
    #####################
    skip_connection_list = []

    for i in range(n_pool):
        for j in range(n_layers_per_block[i]):
            l = fc_layers.BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
            stack = layers.concatenate([stack, l])
            n_filters += growth_rate

        tf.summary.image(f'pool_0_{i}',stack[:,:, :, 0:1],max_outputs=1)
        tf.summary.image(f'pool_1_{i}', stack[:, :, :, 1:2], max_outputs=1)
        tf.summary.image(f'pool_2_{i}', stack[:, :, :, 2:3], max_outputs=1)
        skip_connection_list.append(stack)
        stack = fc_layers.TransitionDown(stack, n_filters, dropout_p)
    skip_connection_list = skip_connection_list[::-1]

    #####################
    #    Bottleneck     #
    #####################
    block_to_upsample = []

    for j in range(n_layers_per_block[n_pool]):
        l = fc_layers.BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
        block_to_upsample.append(l)
        stack = layers.concatenate([stack, l])
    block_to_upsample = layers.concatenate(block_to_upsample)

    #####################
    #  Upsampling path  #
    #####################
    for i in range(n_pool):
        n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
        stack = fc_layers.TransitionUp(skip_connection_list[i], block_to_upsample, n_filters_keep)

        block_to_upsample = []
        for j in range(n_layers_per_block[n_pool + i + 1]):
            l = fc_layers.BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
            block_to_upsample.append(l)
            stack = layers.concatenate([stack, l])
        tf.summary.image(f'upsample_0_{i}', stack[:,:, :, 0:1], max_outputs=1)
        tf.summary.image(f'upsample_1_{i}', stack[:,:, :, 1:2], max_outputs=1)
        tf.summary.image(f'upsample_2_{i}', stack[:,:, :, 2:3], max_outputs=1)
        block_to_upsample = layers.concatenate(block_to_upsample)

    #####################
    #  Softmax          #
    #####################
    output = fc_layers.SoftmaxLayer(stack, n_classes)
    tf.summary.image('result', output, max_outputs=3)
    model =  Model(inputs=inputs, outputs=output)
    return model