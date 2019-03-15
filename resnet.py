"""
Reproducing paper:
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Identity Mappings in Deep Residual Networks"
"""
import mxnet as mx

from mxnet.symbol import Convolution as Conv
from mxnet.symbol import Pooling as Pool
from mxnet.symbol import Activation as Relu
from PYOP.sync_bn_wrapper import BatchNorm as BN

use_global_stats = False
fix_gamma = False
eps = 2e-5
bn_mom = 0.997

def non_local_block(insym, num_filter, mode='dot', resample=False, resample_rate=2, ith=0):
    """Return nonlocal neural network block
    Parameters
    ----------
    insym : mxnet symbol
        Input symbol
    num_filter : int
        Number of input channels
    mode : str [Not Implemented yet]
        `mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`
    """
    # resample
    if resample:
        sub_insym = Pool(data=insym, kernel=(3, 3), stride=(resample_rate, resample_rate), pad=(1, 1), pool_type='max')

    # only dot is implemented
    inter_filter = int(num_filter / 2) if num_filter >= 1024 else num_filter
    indata1 = mx.sym.Convolution(insym, kernel=(1, 1), stride=(1, 1), num_filter=inter_filter,
                                 no_bias=False, name='nonlocal_conv%d1' % ith)
    # resample
    if resample:
        indata2 = mx.sym.Convolution(sub_insym, kernel=(1, 1), stride=(1, 1), 
                                 num_filter=inter_filter, no_bias=False, 
                                 name='nonlocal_conv%d2' % ith)
    else:
        indata2 = mx.sym.Convolution(insym, kernel=(1, 1), stride=(1, 1), num_filter=inter_filter,
                                 no_bias=False, name='nonlocal_conv%d2' % ith)

    # data size: batch_size x (num_filter / 2) x HW
    indata1 = mx.sym.reshape(indata1, shape=(0, 0, -1))
    indata2 = mx.sym.reshape(indata2, shape=(0, 0, -1))

    # f size: batch_size x HW x HW
    f = mx.sym.batch_dot(lhs=indata1, rhs=indata2, transpose_a=True, name='nonlocal_dot%d1' % ith)

    # add softmax layer
    f = inter_filter ** -.5 * f
    f = mx.sym.softmax(f, axis=2)

    # resample
    if resample:
        indata3 = mx.sym.Convolution(sub_insym, kernel=(1, 1), stride=(1, 1), 
                                 num_filter=num_filter, no_bias=False, 
                                 name='nonlocal_conv3%d' % ith)
    else:
        indata3 = mx.sym.Convolution(insym, kernel=(1, 1), stride=(1, 1), num_filter=num_filter,
                                 no_bias=False, name='nonlocal_conv3%d' % ith)
    # g size: batch_size x (num_filter / 2) x HW
    g = mx.sym.reshape(indata3, shape=(0, 0, -1))

    y = mx.sym.batch_dot(lhs=f, rhs=g, transpose_b=True, name='nonlocal_dot%d2' % ith)
    y = mx.sym.reshape_like(lhs=mx.sym.transpose(y, axes=(0, 2, 1)), rhs=insym)
    # y = mx.sym.reshape_like(lhs=y, rhs=indata3)
    y = mx.sym.Convolution(y, kernel=(1, 1), stride=(1, 1), num_filter=num_filter,
                           no_bias=False, name='nonlocal_conv%d4' % ith)
    y = BN(data=y, fix_gamma=fix_gamma, use_global_stats=use_global_stats,
             eps=eps, momentum=bn_mom, name='nonlocal_bn%d4' % ith)

    outsym = insym + y
    return outsym

def residual_unit(data, num_filter, stride, dilate, dim_match, name):
    s = stride
    d = dilate

    bn1 = BN(data=data, fix_gamma=fix_gamma, use_global_stats=use_global_stats,
             eps=eps, momentum=bn_mom, name=name + '_bn1')
    act1 = Relu(data=bn1, act_type='relu', name=name + '_relu1')
    conv1 = Conv(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1),
                 no_bias=True, name=name + '_conv1')

    bn2 = BN(data=conv1, fix_gamma=fix_gamma, use_global_stats=use_global_stats,
             eps=eps, momentum=bn_mom, name=name + '_bn2')
    act2 = Relu(data=bn2, act_type='relu', name=name + '_relu2')
    conv2 = Conv(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), pad=(d, d),
                 stride=(s, s), dilate=(d, d), no_bias=True, name=name + '_conv2')

    bn3 = BN(data=conv2, fix_gamma=fix_gamma, use_global_stats=use_global_stats,
             eps=eps, momentum=bn_mom, name=name + '_bn3')
    act3 = Relu(data=bn3, act_type='relu', name=name + '_relu3')
    conv3 = Conv(data=act3, num_filter=num_filter, kernel=(1, 1), no_bias=True, name=name + '_conv3')
    if dim_match:
        shortcut = data
    else:
        shortcut = Conv(data=act1, num_filter=num_filter, kernel=(1, 1), stride=(s, s),
                no_bias=True, name=name + '_sc')

    shortcut._set_attr(mirror_stage='True')
    return conv3 + shortcut


def resnet101_c2():
    # preprocessing
    data = mx.sym.Variable(name='data')
    data = BN(data=data, fix_gamma=True, use_global_stats=use_global_stats, eps=eps,
              momentum=bn_mom, name='bn_data')

    # C1, 7x7
    data = Conv(data=data, num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3), no_bias=True, name="conv0")
    data = BN(data=data, fix_gamma=fix_gamma, use_global_stats=use_global_stats, eps=eps, momentum=bn_mom,
              name='bn0')
    data = Relu(data=data, act_type='relu', name='relu0')

    # C2, 3 blocks
    data = Pool(data=data, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')
    #                          c,   s, d, sc   , n
    data = residual_unit(data, 256, 1, 1, False, "stage1_unit1")
    data = residual_unit(data, 256, 1, 1, True,  "stage1_unit2")
    data = residual_unit(data, 256, 1, 1, True,  "stage1_unit3")

    # os = 4
    return data

def resnet101_c3(aspp_with_separable_conv=False):
    feat = resnet101_c2()
    # C3, 4 blocks
    data = residual_unit(feat, 512, 2, 1, False, "stage2_unit1")
    data = residual_unit(data, 512, 1, 1, True,  "stage2_unit2")
    data = residual_unit(data, 512, 1, 1, True,  "stage2_unit3")
    data = residual_unit(data, 512, 1, 1, True,  "stage2_unit4")

    # vanilla non_local block
    #if non_local:
    #    data = non_local_block(data, 512, mode='dot', resample=True, resample_rate=2, ith=2)

    # os = 8
    if aspp_with_separable_conv:
        return feat, data
    else:
        return data


def resnet101_c4(output_stride, aspp_with_separable_conv=False, non_local=False):
    if aspp_with_separable_conv:
        feat, data = resnet101_c3(aspp_with_separable_conv, non_local=non_local)
    else:
        data = resnet101_c3()

    # C4, 23 blocks
    # os=16 rate=1 / os=8 rate=2
    if output_stride == 32 or output_stride == 16:
        s = 2
        d = 1
    elif output_stride == 8:
        s = 1
        d = 2
    else:
        raise ValueError("unsupported output stride: {}".format(output_stride))

    data = residual_unit(data, 1024, s, 1, False, "stage3_unit1")
    data = residual_unit(data, 1024, 1, d, True,  "stage3_unit2")
    data = residual_unit(data, 1024, 1, d, True,  "stage3_unit3")
    data = residual_unit(data, 1024, 1, d, True,  "stage3_unit4")
    data = residual_unit(data, 1024, 1, d, True,  "stage3_unit5")
    data = residual_unit(data, 1024, 1, d, True,  "stage3_unit6")
    data = residual_unit(data, 1024, 1, d, True,  "stage3_unit7")
    data = residual_unit(data, 1024, 1, d, True,  "stage3_unit8")
    data = residual_unit(data, 1024, 1, d, True,  "stage3_unit9")
    data = residual_unit(data, 1024, 1, d, True,  "stage3_unit10")
    data = residual_unit(data, 1024, 1, d, True,  "stage3_unit11")
    data = residual_unit(data, 1024, 1, d, True,  "stage3_unit12")
    data = residual_unit(data, 1024, 1, d, True,  "stage3_unit13")
    data = residual_unit(data, 1024, 1, d, True,  "stage3_unit14")
    data = residual_unit(data, 1024, 1, d, True,  "stage3_unit15")
    data = residual_unit(data, 1024, 1, d, True,  "stage3_unit16")
    data = residual_unit(data, 1024, 1, d, True,  "stage3_unit17")
    data = residual_unit(data, 1024, 1, d, True,  "stage3_unit18")
    data = residual_unit(data, 1024, 1, d, True,  "stage3_unit19")
    data = residual_unit(data, 1024, 1, d, True,  "stage3_unit20")
    data = residual_unit(data, 1024, 1, d, True,  "stage3_unit21")
    data = residual_unit(data, 1024, 1, d, True,  "stage3_unit22")
    data = residual_unit(data, 1024, 1, d, True,  "stage3_unit23")

    # vanilla non_local block
    if non_local:
        data = non_local_block(data, 1024, mode='dot', resample=True, resample_rate=2, ith=4)

    # os = 16/ os = 8
    if aspp_with_separable_conv:
        return feat, data
    else:
        return data


def resnet101_c5(non_local=False):
    data = resnet101_c4(non_local)

    # C5, 3 blocks
    data = residual_unit(data, 2048, 2, 1, False, "stage4_unit1")
    data = residual_unit(data, 2048, 1, 1, True,  "stage4_unit2")
    data = residual_unit(data, 2048, 1, 1, True,  "stage4_unit3")

    # os = 32
    return data


def extra_stage(data, multi_grid, stage):
    MG = multi_grid
    # extra stage, 3 blocks
    data = residual_unit(data, 2048, 1, MG[0], False, "%s_unit1" % stage)
    data = residual_unit(data, 2048, 1, MG[1], True,  "%s_unit2" % stage)
    data = residual_unit(data, 2048, 1, MG[2], True,  "%s_unit3" % stage)

    return data
