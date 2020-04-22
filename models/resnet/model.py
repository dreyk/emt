import tensorflow as tf

import models.resnet.resnet_gn_ws as resnet
import models.resnet.layers as clayers
import logging


def _encoder_layer(input, planes, blocks,dilate=1, stride=1,layer=1):
    downsample = None
    if stride != 1 or input.shape[3] != planes * 4:
        downsample = resnet.Downsample(planes * 4, stride if dilate==1 else 1)
        logging.info('Encoder Layer {} downsample to {}'.format(layer,planes * 4))

    input = resnet.Bottleneck(planes, stride,dilate=dilate,layer=layer,name=f'EncoderL{layer}-0')(input,downsample)
    for i in range(1, blocks):
        input = resnet.Bottleneck(planes,layer=layer,dilate=dilate,name=f'EncoderL{layer}-{i}')(input)
    return input


def encoder(input):
    conv_out = [input]
    conv1 = tf.keras.layers.Conv2D(64,
                                   kernel_size=7,
                                   strides=2,
                                   padding='same',
                                   use_bias=False,
                                   kernel_initializer='he_uniform', kernel_regularizer=clayers.ws_reg)(input)
    logging.info("conv1: {}".format(conv1.shape))
    bn1 = clayers.GroupNormalization()(conv1)
    logging.info("bn1: {}".format(bn1.shape))
    relu = tf.keras.layers.Activation(tf.keras.activations.relu)(bn1)
    logging.info("relu: {}".format(relu.shape))
    conv_out.append(relu)
    maxpool = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(relu)
    logging.info("maxpool: {}".format(maxpool.shape))
    conv_out.append(maxpool)
    layer1 = _encoder_layer(maxpool, 64, 3,layer='1')
    conv_out.append(layer1)
    layer2 = _encoder_layer(layer1, 128, 4, stride=2,layer='2')
    conv_out.append(layer2)
    layer3 = _encoder_layer(layer2, 256, 6, stride=2,dilate=2,layer='3')
    conv_out.append(layer3)
    layer4 = _encoder_layer(layer3, 512, 3, stride=2,dilate=4,layer='4')
    conv_out.append(layer4)
    return conv_out

def fba_fusion(alpha, img, F, B):
    F = ((alpha * img + (1 - alpha**2) * F - alpha * (1 - alpha) * B))
    B = ((1 - alpha) * img + (2 * alpha - alpha**2) * B - alpha * (1 - alpha) * F)

    F = tf.clip_by_value(F,0,1)
    B = tf.clip_by_value(B,0,1)
    la = 0.1
    alpha = (alpha * la +tf.reduce_sum((img - B) * (F - B),3,True)) / (tf.reduce_sum((F - B) * (F - B),3,True) + la)
    alpha = tf.clip_by_value(alpha,0,1)
    return alpha, F, B

class FBADecoder(tf.keras.layers.Layer):
    expansion = 1
    def __init__(self,**kwargs):
        super(FBADecoder, self).__init__(**kwargs)
        pool_scales = (1, 2, 3, 6)
        self.ppm = []
        for p in pool_scales:
            self.ppm.append(resnet.FBADecoderBlock(256,1,p))

        self.conv_up1_0 = resnet.FBADecoderBlock(256,3)
        self.conv_up1_1 = resnet.FBADecoderBlock(256, 3)

        self.conv_up2 = resnet.FBADecoderBlock(256, 3)

        self.conv_up3 = resnet.FBADecoderBlock(64, 3)

        self.upool = tf.keras.layers.UpSampling2D(2)

        self.conv_up4_0 = resnet.FBADecoderBlock(32, 3)
        self.conv_up4_1 = resnet.FBADecoderBlock(16, 3)
        self.conv_up4_7 = resnet.FBADecoderBlock(7, 3)

    def call(self, conv_out,img, two_chan_trimap):
        for i,c in enumerate(conv_out):
            logging.info('Decode conv_out{}={}'.format(i,conv_out[i].shape))
        conv5 = conv_out[-1]
        logging.info('Decode conv5={}'.format(conv5.shape))
        ppm_out = [conv5]
        w = conv5.shape[2]
        h = conv5.shape[1]
        for pool_scale in self.ppm:
            x = pool_scale(conv5)
            logging.info('Decode pool_scale0={}'.format(x.shape))
            sw = w // x.shape[2]
            sh = h // x.shape[1]
            x = tf.keras.layers.UpSampling2D((sh,sw),interpolation='bilinear')(x)
            logging.info('Decode pool_scale1={}'.format(x.shape))
            ppm_out.append(x)
        ppm_out = tf.concat(ppm_out,-1)
        x = self.conv_up1_0(ppm_out)
        x = self.conv_up1_1(x)
        x = tf.keras.layers.UpSampling2D((2,2),interpolation='bilinear')(x)
        x = tf.concat([x, conv_out[-4]], -1)

        x = self.conv_up2(x)
        x = tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear')(x)
        x = tf.concat([x, conv_out[-5]], -1)

        x = self.conv_up3(x)
        x = tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear')(x)
        logging.info('Decode last x={}'.format(x.shape))
        logging.info('Decode conv_out[-6]={}'.format(conv_out[-4].shape))
        x = tf.concat([x, conv_out[-6][:,:,:,:3], img, two_chan_trimap], -1)
        output = self.conv_up4_0(x)
        output = self.conv_up4_1(output)
        output = self.conv_up4_2(output)
        alpha = tf.clip_by_value(output[:, :, :, 0:1], 0, 1)
        fg = tf.keras.activations.sigmoid(output[:, :, :, 1:4])
        bg = tf.keras.activations.sigmoid(output[:, :, :, 4:7])
        alpha, fg, bg = fba_fusion(alpha, img, fg, bg)

        return [alpha, fg, bg]

def Matting():
    img = tf.keras.layers.Input(shape=(320, 320, 3), name='img')
    trimap = tf.keras.layers.Input(shape=(320, 320, 1), name='trimap')
    inputs = [img, trimap]
    input = tf.keras.layers.concatenate(inputs)
    encoder_outs = encoder(input)
    decoder_outputs = FBADecoder()(encoder_outs,img,trimap)
    model = tf.keras.Model(inputs=inputs, outputs=decoder_outputs)
    return model