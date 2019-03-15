import pdb
from config.deeplab import config
from PYOP.broadcast_like import broadcast_like
from PYOP.separable_conv import split_separable_conv as Sepconv

import mxnet as mx
from mxnet.symbol import Convolution as Conv
from mxnet.symbol import Pooling as Pool
from mxnet.symbol import Activation as Relu
from mxnet.symbol import Activation
from mxnet.symbol import Dropout as Dropout
from mxnet.symbol import Deconvolution as Deconv
from mxnet.symbol import concat
from PYOP.sync_bn_wrapper import BatchNorm as BN

use_global_stats = False
fix_gamma = False
eps = 2e-5
bn_mom = 0.997
args = {}

def oc_context_block(insym, key_channels, value_channels, out_channels, resample_rate=2):
    """OC Context Block
    https://github.com/PkuRainBow/OCNet.pytorch/blob/master/oc_module/base_oc_block.py
    default param: 
      key_channels=out_channels//2,
      value_channels=out_channels,
      resample_rate=2
    """
    # pre_conv
    insym = mx.sym.Convolution(insym, kernel=(3, 3), stride=(1, 1), pad=(1,1), 
                                 num_filter=out_channels, no_bias=False, name='oc_pre_conv')
    insym = BN(insym, use_global_stats=use_global_stats, fix_gamma=fix_gamma,
                         momentum=bn_mom, name="oc_pre_bn", eps=eps, **args)

    # resample
    if resample_rate > 1:
        sub_insym = Pool(data=insym, kernel=(3, 3), stride=(resample_rate, resample_rate), pad=(1, 1), pool_type='max')

    # only dot is implemented
    query = mx.sym.Convolution(insym, kernel=(1, 1), stride=(1, 1), 
                                 num_filter=key_channels, no_bias=True, name='oc_conv_query')
    if resample_rate > 1:
        value = mx.sym.Convolution(sub_insym, kernel=(1, 1), stride=(1, 1), 
                                 num_filter=value_channels, no_bias=True, name='oc_conv_value')
        key = mx.sym.Convolution(sub_insym, kernel=(1, 1), stride=(1, 1), num_filter=key_channels,
                                 no_bias=True, name='oc_conv_key')
    else:
        value = mx.sym.Convolution(insym, kernel=(1, 1), stride=(1, 1), 
                                 num_filter=value_channels, no_bias=True, name='oc_conv_value')
        key = mx.sym.Convolution(insym, kernel=(1, 1), stride=(1, 1), num_filter=key_channels,
                                 no_bias=True, name='oc_conv_key')
    query = BN(query, use_global_stats=use_global_stats, fix_gamma=fix_gamma,
                         momentum=bn_mom, name="oc_query_bn", eps=eps, **args)
    key = BN(key, use_global_stats=use_global_stats, fix_gamma=fix_gamma,
                         momentum=bn_mom, name="oc_key_bn", eps=eps, **args)

    # data size: batch_size x (num_filter / 2) x HW
    value = mx.sym.reshape(value, shape=(0, 0, -1))
    query = mx.sym.reshape(query, shape=(0, 0, -1))
    key = mx.sym.reshape(key, shape=(0, 0, -1))

    # f size: batch_size x HW x HW
    sim_map = mx.sym.batch_dot(lhs=query, rhs=key, transpose_a=True, name='oc_sim')
    sim_map = (key_channels ** -0.5) * sim_map

    # add softmax layer
    sim_map = mx.sym.softmax(sim_map, axis=2)

    context = mx.sym.batch_dot(lhs=sim_map, rhs=value, transpose_b=True, name='oc_context')
    context = mx.sym.reshape_like(lhs=mx.sym.transpose(context, axes=(0, 2, 1)), rhs=insym)
    context = mx.sym.Convolution(context, kernel=(1, 1), stride=(1, 1), num_filter=out_channels,
                           no_bias=False, name='oc_W')

    # reverse resample
    #if resample_rate > 1:
    #    [_, __, scaled_height, scaled_width] = context.infer_shape()*2
    #    context = mx.symbol.contrib.BilinearResize2D(data=context, height=scaled_height, width=scaled_width)

    # conv_bn_dropout
    #context = mx.sym.Convolution(context, kernel=(1, 1), stride=(1, 1), num_filter=out_channels,
    #                       no_bias=True, name='oc_conv_dropout')
    #context = BN(context, use_global_stats=use_global_stats, fix_gamma=fix_gamma,
    #                     momentum=bn_mom, name="oc_bn_dropout", eps=eps, **args)

    return context

def atrous_spatial_pyramid_pooling(feat, rate, aspp_with_separable_conv, oc_context=False):
    conv_1x1 = Conv(feat, num_filter=256, kernel=(1, 1), name="aspp_1x1")
    conv_1x1 = BN(conv_1x1, use_global_stats=use_global_stats, fix_gamma=fix_gamma,
                  momentum=bn_mom, name="aspp_1x1_bn", eps=eps, **args)
    conv_1x1 = Relu(conv_1x1, act_type='relu', name='aspp_1x1_relu')

    if aspp_with_separable_conv:
        conv_3x3_d6 = Sepconv(data=feat, in_channel=2048, num_filter=256, stride=1,
                              dilate=6 * rate, name="aspp_3x3_d6")
        conv_3x3_d6 = BN(conv_3x3_d6, use_global_stats=use_global_stats, fix_gamma=fix_gamma,
                         momentum=bn_mom, name="aspp_3x3_d6_bn", eps=eps, **args)
        conv_3x3_d6 = Relu(conv_3x3_d6, act_type='relu', name='aspp_3x3_d6_relu')
        conv_3x3_d12 = Sepconv(data=feat, in_channel=2048, num_filter=256, stride=1,
                               dilate=12 * rate, name="aspp_3x3_d12")
        conv_3x3_d12 = BN(conv_3x3_d12, use_global_stats=use_global_stats, fix_gamma=fix_gamma,
                          momentum=bn_mom, name="aspp_3x3_d12_bn", eps=eps, **args)
        conv_3x3_d12 = Relu(conv_3x3_d12, act_type='relu', name='aspp_3x3_d12_relu')
        conv_3x3_d18 = Sepconv(data=feat, in_channel=2048, num_filter=256, stride=1,
                               dilate=18 * rate, name="aspp_3x3_d18")
        conv_3x3_d18 = BN(conv_3x3_d18, use_global_stats=use_global_stats, fix_gamma=fix_gamma,
                          momentum=bn_mom, name="aspp_3x3_d18_bn", eps=eps, **args)
        conv_3x3_d18 = Relu(conv_3x3_d18, act_type='relu', name='aspp_3x3_d18_relu')
    else:
        conv_3x3_d6 = Conv(feat, num_filter=256, kernel=(3, 3), dilate=(6 * rate, 6 * rate),
                           pad=(6 * rate, 6 * rate), name="aspp_3x3_d6")
        conv_3x3_d6 = BN(conv_3x3_d6, use_global_stats=use_global_stats, fix_gamma=fix_gamma,
                         momentum=bn_mom, name="aspp_3x3_d6_bn", eps=eps)
        conv_3x3_d6 = Relu(conv_3x3_d6, act_type='relu', name='aspp_3x3_d6_relu')
        conv_3x3_d12 = Conv(feat, num_filter=256, kernel=(3, 3), dilate=(12 * rate, 12 * rate),
                            pad=(12 * rate, 12 * rate), name="aspp_3x3_d12")
        conv_3x3_d12 = BN(conv_3x3_d12, use_global_stats=use_global_stats, fix_gamma=fix_gamma,
                          momentum=bn_mom, name="aspp_3x3_d12_bn", eps=eps)
        conv_3x3_d12 = Relu(conv_3x3_d12, act_type='relu', name='aspp_3x3_d12_relu')
        conv_3x3_d18 = Conv(feat, num_filter=256, kernel=(3, 3), dilate=(18 * rate, 18 * rate),
                            pad=(18 * rate, 18 * rate), name="aspp_3x3_d18")
        conv_3x3_d18 = BN(conv_3x3_d18, use_global_stats=use_global_stats, fix_gamma=fix_gamma,
                          momentum=bn_mom, name="aspp_3x3_d18_bn", eps=eps)
        conv_3x3_d18 = Relu(conv_3x3_d18, act_type='relu', name='aspp_3x3_d18_relu')

    if oc_context:
        gap = oc_context_block(feat, 128, 256, 256, resample_rate=2)
    else:
        gap = Pool(feat, kernel=(1, 1), global_pool=True, pool_type="avg", name="aspp_gap")
    gap = Conv(gap, num_filter=256, kernel=(1, 1), name="aspp_gap_1x1")
    gap = BN(gap, use_global_stats=use_global_stats, fix_gamma=fix_gamma, momentum=bn_mom,
             name="aspp_gap_1x1_bn", eps=eps, **args)
    if not oc_context:
        gap = Relu(gap, act_type='relu', name='aspp_gap_1x1_relu')
        gap = broadcast_like(gap, conv_1x1, name="aspp_gap_broadcast")
    aspp = concat(conv_1x1, conv_3x3_d6, conv_3x3_d12, conv_3x3_d18, gap, dim=1, name="aspp_concat")
    aspp_1x1 = Conv(aspp, num_filter=256, kernel=(1, 1), name="aspp_concat_1x1")
    aspp_1x1 = BN(aspp_1x1, use_global_stats=use_global_stats, fix_gamma=fix_gamma, momentum=bn_mom,
                  name="aspp_concat_1x1_bn", eps=eps, **args)
    aspp_1x1._set_attr(mirror_stage='True')
    aspp_1x1 = Relu(aspp_1x1, act_type='relu', name='aspp_concat_1x1_relu')
    return aspp_1x1
