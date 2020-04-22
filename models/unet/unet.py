import models.layers.layers as layers
import tensorflow as tf


def block(input,filters,norm,pooling=True):
    conv1 = tf.keras.layers.Conv2D(filters, 3,padding='same', kernel_initializer='he_normal',kernel_regularizer=layers.ws_reg)(input)
    n1 = norm(conv1)
    r1 = tf.keras.layers.Activation(tf.keras.activations.relu)(n1)
    conv2 = tf.keras.layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal',kernel_regularizer=layers.ws_reg)(r1)
    n2 = norm(conv2)
    r3 = tf.keras.layers.Activation(tf.keras.activations.relu)(n2)
    if pooling:
        pool = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(r3)
        return r3,pool
    return r3

def fba_fusion(alpha, img, F, B):
    F = ((alpha * img + (1 - alpha**2) * F - alpha * (1 - alpha) * B))
    B = ((1 - alpha) * img + (2 * alpha - alpha**2) * B - alpha * (1 - alpha) * F)

    F = tf.clip_by_value(F,0,1)
    B = tf.clip_by_value(B,0,1)
    la = 0.1
    alpha = (alpha * la +tf.reduce_sum((img - B) * (F - B),3,True)) / (tf.reduce_sum((F - B) * (F - B),3,True) + la)
    alpha = tf.clip_by_value(alpha,0,1)
    return alpha, F, B

def unet(input_shape=(None, None, 3),first_chan=16,pools=4,growth_add=0,growth_scale=2,out_chans=1,use_group_norm=True):
    if use_group_norm:
        def _norm(norm_groups):
            return layers.GroupNormalization(groups=norm_groups)
        norm = _norm
    else:
        def _norm(norm_groups):
            return tf.keras.layers.BatchNormalization()

        norm = _norm
    inputs = tf.keras.layers.Input(input_shape)
    filters = first_chan
    connections = []
    pool = tf.keras.layers.Conv2D(first_chan, 7, padding='same', kernel_initializer='he_normal',
                                  kernel_regularizer=layers.ws_reg)(inputs)
    pool = norm(first_chan // 2)(pool)
    pool = tf.keras.layers.Activation(tf.keras.activations.relu)(pool)
    for i in range(pools):
        conv,pool = block(pool,filters,norm(first_chan//2),True)
        connections.append(conv)
        if growth_add>0:
            filters += growth_add
        else:
            filters *= growth_scale

    connections.reverse()

    conv = block(pool,filters,norm(first_chan),False)


    for i in range(pools):
        up = tf.keras.layers.Conv2DTranspose(filters,3,strides=2,padding='same', kernel_initializer='he_normal',kernel_regularizer=layers.ws_reg)(conv)
        concat = tf.keras.layers.concatenate([connections[i], up])
        conv = block(concat,filters,norm(first_chan),False)

        if i<(pools-1):
            if growth_add > 0:
                filters -= growth_add
            else:
                filters = filters // growth_scale

    conv = tf.keras.layers.Conv2D(filters, 7, padding='same', kernel_initializer='he_normal',kernel_regularizer=layers.ws_reg)(conv)
    n = norm(first_chan//2)(conv)
    r = tf.keras.layers.Activation(tf.keras.activations.relu)(n)
    conv_m = tf.keras.layers.Conv2D(out_chans, 1, padding='same', kernel_initializer='he_normal',
                                   kernel_regularizer=layers.ws_reg)(r)
    conv_final = tf.keras.layers.Conv2D(out_chans, 1, padding='same', kernel_initializer='he_normal',
                                    kernel_regularizer=layers.ws_reg)(conv_m)
    alpha = tf.clip_by_value(conv_final[:, :, :, 0:1], 0, 1)
    if out_chans==7:
        fg = tf.keras.activations.sigmoid(r[:, :, :, 1:4])
        bg = tf.keras.activations.sigmoid(r[:, :, :, 4:7])
        alpha, fg, bg = fba_fusion(alpha, inputs, fg, bg)
        model = tf.keras.Model(inputs=inputs, outputs=[alpha,fg,bg])
    else:
        model = tf.keras.Model(inputs=inputs, outputs=alpha)
    return model





