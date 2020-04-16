import models.fc_densenet.layers as fc_layers
import tensorflow as tf

def fba_fusion(alpha, img, F, B):
    F = ((alpha * img + (1 - alpha**2) * F - alpha * (1 - alpha) * B))
    B = ((1 - alpha) * img + (2 * alpha - alpha**2) * B - alpha * (1 - alpha) * F)

    F = tf.clip_by_value(F,0,1)
    B = tf.clip_by_value(B,0,1)
    la = 0.1
    alpha = (alpha * la +tf.reduce_sum((img - B) * (F - B),3,True)) / (tf.reduce_sum((F - B) * (F - B),3,True) + la)
    alpha = tf.clip_by_value(alpha,0,1)
    return alpha, F, B

def FCDensNetMatting(
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

    img = tf.keras.layers.Input(shape=(320, 320, 3), name='img')
    trimap = tf.keras.layers.Input(shape=(320, 320, 1), name='trimap')
    inputs = [img, trimap]
    input = tf.keras.layers.concatenate([img, trimap])
    print('n_filters_first_conv={}'.format(n_filters_first_conv))
    stack = tf.keras.layers.Conv2D(filters=n_filters_first_conv, kernel_size=3, padding='same',
                                   kernel_initializer='he_uniform',kernel_regularizer=fc_layers.ws_reg)(input)
    print('stack={}'.format(stack.shape))
    n_filters = n_filters_first_conv

    skip_connection_list = []

    for i in range(n_pool):
        for j in range(n_layers_per_block[i]):
            l = fc_layers.BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
            stack = tf.keras.layers.concatenate([stack, l])
            n_filters += growth_rate
        skip_connection_list.append(stack)
        stack = fc_layers.TransitionDown(stack, n_filters, dropout_p)
    skip_connection_list = skip_connection_list[::-1]

    block_to_upsample = []

    for j in range(n_layers_per_block[n_pool]):
        l = fc_layers.BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
        block_to_upsample.append(l)
        stack = tf.keras.layers.concatenate([stack, l])
    block_to_upsample = tf.keras.layers.concatenate(block_to_upsample)

    for i in range(n_pool):
        n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
        stack = fc_layers.TransitionUp(skip_connection_list[i], block_to_upsample, n_filters_keep)

        block_to_upsample = []
        for j in range(n_layers_per_block[n_pool + i + 1]):
            l = fc_layers.BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
            block_to_upsample.append(l)
            stack = tf.keras.layers.concatenate([stack, l])
        block_to_upsample = tf.keras.layers.concatenate(block_to_upsample)

    l = tf.keras.layers.Conv2D(7, kernel_size=1, padding='same', kernel_initializer='he_uniform',kernel_regularizer=fc_layers.ws_reg)(stack)
    alpha = tf.keras.activations.hard_sigmoid(l[:, :, :, 0:1])
    alpha = tf.clip_by_value(alpha,0,1)
    fg = tf.keras.activations.sigmoid(l[:, :, :, 1:4])
    bg = tf.keras.activations.sigmoid(l[:, :, :, 4:7])
    alpha, fg, bg = fba_fusion(alpha, img, fg, bg)
    model = tf.keras.Model(inputs=inputs, outputs=[alpha, fg, bg])
    return model
