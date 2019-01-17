# Generate prototxt of waifu2x's cunet/upcunet arch. Training is not possible.
from __future__ import print_function
import sys
sys.path.insert(0, "../python") # pycaffe path
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import caffe

def seblock(bottom, o, r):
    m = int(o / r)
    gap = L.Pooling(bottom, pool=P.Pooling.AVE, global_pooling=True)
    linear1 = L.Convolution(gap,  kernel_size=1, pad=0, stride=1, num_output=m)
    relu1 = L.ReLU(linear1, in_place=True)
    linear2 = L.Convolution(relu1,  kernel_size=1, pad=0, stride=1, num_output=o)
    sigmoid1 = L.Sigmoid(linear2, in_place=True)
    flatten1 = L.Flatten(sigmoid1)
    return flatten1

def conv_relu(bottom, o, stride=1, pad=0):
    conv = L.Convolution(bottom, kernel_size=3, stride=stride, num_output=o, pad=pad)
    relu = L.ReLU(conv, in_place=True, negative_slope=0.1)
    return relu

def unet_conv(bottom, m, o, se=True):
    conv1 = L.Convolution(bottom, kernel_size=3, stride=1,
                                num_output=m, pad=0)
    relu1 = L.ReLU(conv1, in_place=True, negative_slope=0.1)
    conv2 = L.Convolution(relu1, kernel_size=3, stride=1,
                                num_output=o, pad=0)
    relu2 = L.ReLU(conv2, in_place=True, negative_slope=0.1)

    if se:
        se1 = seblock(relu2, o, 8)
        return L.Scale(relu2, se1, axis=0, bias_term=False)
    else:
        return relu2

def unet_branch(bottom, insert_f, i, o, depad):
    pool = L.Convolution(bottom, kernel_size=2, stride=2, num_output=i, pad=0)
    relu1 = L.ReLU(pool, in_place=True, negative_slope=0.1)
    feat = insert_f(relu1)
    unpool = L.Deconvolution(feat, convolution_param=dict(num_output=o, kernel_size=2, pad=0, stride=2))
    relu2 = L.ReLU(unpool, in_place=True, negative_slope=0.1)
    crop = L.Crop(bottom, relu2, crop_param=dict(axis=2, offset=depad))
    cadd = L.Eltwise(crop, relu2, operation=P.Eltwise.SUM)
    return cadd

def unet1(bottom, ch, deconv):
    block1 = lambda bottom: unet_conv(bottom, 128, 64, True)
    conv1 = unet_conv(bottom, 32, 64, se=False)
    ub1 = unet_branch(conv1, block1, 64, 64, 4)
    conv2 = conv_relu(ub1, 64)
    if deconv:
        return L.Deconvolution(conv2, convolution_param=dict(num_output=ch, kernel_size=4, pad=3, stride=2))
    else:
        return L.Convolution(conv2, kernel_size=3, stride=1, num_output=ch, pad=0)

def unet2(bottom, ch, deconv):
    def block1(bottom):
        return unet_conv(bottom, 256, 128, se=True)
    def block2(bottom):
        conv1 = unet_conv(bottom, 64, 128, se=True)
        ub1 = unet_branch(conv1, block1, 128, 128, 4)
        conv2 = unet_conv(ub1, 64, 64, se=True)
        return conv2
    conv1 = unet_conv(bottom, 32, 64, se=False)
    ub1 = unet_branch(conv1, block2, 64, 64, 16)
    conv2 = conv_relu(ub1, 64)
    if deconv:
        return L.Deconvolution(conv2, convolution_param=dict(num_output=ch, kernel_size=4, pad=3, stride=2))
    else:
        return L.Convolution(conv2, kernel_size=3, stride=1, num_output=ch, pad=0)

def make_upcunet():
    netoffset = 36
    ch = 3
    input_size = (256 / 2) + netoffset * 2
    assert(input_size % 4 == 0)

    data = L.Input(name="input", shape=dict(dim=[1, ch, input_size, input_size]))
    u1 = unet1(data, ch=ch, deconv=True)
    u2 = unet2(u1, ch=ch, deconv=False)
    crop = L.Crop(u1, u2, crop_param=dict(axis=2, offset=20))
    cadd = L.Eltwise(crop, u2, operation=P.Eltwise.SUM)
    return to_proto(cadd)

def make_cunet():
    netoffset = 28
    ch = 3
    input_size = 256 + netoffset * 2
    assert(input_size % 4 == 0)

    data = L.Input(name="input", shape=dict(dim=[1, ch, input_size, input_size]))
    u1 = unet1(data, ch=ch, deconv=False)
    u2 = unet2(u1, ch=ch, deconv=False)
    crop = L.Crop(u1, u2, crop_param=dict(axis=2, offset=20))
    cadd = L.Eltwise(crop, u2, operation=P.Eltwise.SUM)
    return to_proto(cadd)

def make_net():
    with open('upcunet.prototxt', 'w') as f:
        print(make_upcunet(), file=f)
    with open('cunet.prototxt', 'w') as f:
        print(make_cunet(), file=f)

if __name__ == '__main__':
    make_net()
    # test loading the net
    caffe.Net('upcunet.prototxt', caffe.TEST)
    caffe.Net('cunet.prototxt', caffe.TEST)
