import tensorflow as tf
import models.resnet.layers as clayers
import logging

class Conv3x3(tf.keras.layers.Layer):
    def __init__(self,out_planes, stride=1,dilate=1,**kwargs):
        super(Conv3x3, self).__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(out_planes,kernel_size=3,
                                           strides=stride,
                                           dilation_rate=dilate,
                                           padding='same',
                                           use_bias=False,
                                           kernel_initializer='he_uniform',kernel_regularizer=clayers.ws_reg)

    def call(self, inputs):
        return self.conv(inputs)


class Conv1x1(tf.keras.layers.Layer):
    def __init__(self,out_planes, stride=1,**kwargs):
        super(Conv1x1, self).__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(out_planes, kernel_size=1,
                                           strides=stride,
                                           padding='same',
                                           use_bias=False,
                                           kernel_initializer='he_uniform', kernel_regularizer=clayers.ws_reg)

    def call(self, inputs):
        return self.conv(inputs)


class BasicBlock(tf.keras.layers.Layer):
    expansion = 1

    def __init__(self,planes, stride=1, downsample=None,**kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        self.conv1 = Conv3x3(planes, stride)
        self.bn1 = clayers.GroupNormalization()
        self.relu = tf.keras.layers.Activation(tf.keras.activations.relu)
        self.conv2 = Conv3x3(planes, stride)
        self.bn2 = clayers.GroupNormalization()
        self.downsample = downsample
        self.stride = stride

    def call(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class Downsample(tf.keras.layers.Layer):
    def __init__(self, planes, stride=1, **kwargs):
        super(Downsample, self).__init__(**kwargs)
        self.conv = Conv1x1(planes, stride)
        self.norm = clayers.GroupNormalization()

    def call(self, x):
        x = self.conv(x)
        return self.norm(x)

class Bottleneck(tf.keras.layers.Layer):
    expansion = 4

    def __init__(self, planes, stride=1,dilate=1,layer=1, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        if stride==2 and dilate>1:
            stride = 1
            dilate = dilate//2
        self.conv1 = Conv1x1(planes)
        self.bn1 = clayers.GroupNormalization()
        self.conv2 = Conv3x3(planes, stride,dilate=dilate)
        self.bn2 = clayers.GroupNormalization()
        self.conv3 = Conv1x1(planes * self.expansion)
        self.bn3 = clayers.GroupNormalization()
        self.relu = tf.keras.layers.Activation(tf.keras.activations.relu)
        self.layer = layer
        self.stride = stride

    def call(self, x,downsample=None):
        logging.info('Bottelneck layer{} x: {}'.format(self.layer, x.shape))
        identity = x

        out = self.conv1(x)
        logging.info('Bottelneck layer{} conv1: {}'.format(self.layer,out.shape))
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        logging.info('Bottelneck layer{} conv2: {}'.format(self.layer, out.shape))
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        logging.info('Bottelneck layer{} conv3: {}'.format(self.layer, out.shape))

        if downsample is not None:
            identity = downsample(x)
        logging.info('Bottelneck layer{} identity: {}'.format(self.layer, identity.shape))

        out += identity
        out = self.relu(out)

        return out



class FBADecoderBlock(tf.keras.layers.Layer):
    def __init__(self,filters,kernel_size=1,pool_scale=0,**kwargs):
        super(FBADecoderBlock, self).__init__(**kwargs)
        self.pool = pool_scale
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size,
                                      padding='same',
                                      use_bias=True,
                                      kernel_initializer='he_uniform', kernel_regularizer=clayers.ws_reg)
        self.relu = tf.keras.layers.Activation(tf.keras.activations.relu)
        self.norm = clayers.GroupNormalization()


    def call(self, x):
        if self.pool>0:
            kernel = (x.shape[1] + self.pool - 1) // self.pool
            x = tf.keras.layers.AveragePooling2D(kernel)(x)
        x = self.conv(x)
        x = self.relu(x)
        return self.norm(x)

class FBADecoderPPM(tf.keras.layers.Layer):
    def __init__(self,pool_scales,**kwargs):
        super(FBADecoderPPM, self).__init__(**kwargs)
        self.ppm = []
        self.pool_scales = pool_scales
        for _ in pool_scales:
            conv = tf.keras.layers.Conv2D(256, kernel_size=1,
                                   padding='same',
                                   use_bias=True,
                                   kernel_initializer='he_uniform', kernel_regularizer=clayers.ws_reg)
            relu = tf.keras.layers.Activation(tf.keras.activations.relu)
            norm = clayers.GroupNormalization()
            self.ppm.append([conv,relu,norm])

    def call(self, x):
        for i,p in enumerate(self.ppm):
            kernel = (x.shape[1]+self.pool_scales[i]-1)//self.pool_scales[i]
            x = tf.keras.layers.AveragePooling2D(kernel)(x)
            for l in p:
                x = l(x)
            logging.info('Decoder PPM{}: {}'.format(i,x.shape))
        return x