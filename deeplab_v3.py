from PYOP.atrous_spatial_pyramid_pooling import atrous_spatial_pyramid_pooling
from PYOP.bilinear_upsampling import bilinear_upsampling, refine_decoder
from symbols.resnet import resnet101_c4, resnet101_c2, extra_stage

def get_symbol(num_classes, output_stride=16, use_refine_decoder=False,
               aspp_with_separable_conv=False, 
               non_local=False, oc_context=False):
    if use_refine_decoder:
        # os = 16
        feat, data = deeplabv3_aspp(output_stride=output_stride,
                                    aspp_with_separable_conv=aspp_with_separable_conv,
                                    non_local=non_local, oc_context=oc_context)
        data = refine_decoder(feat, data, num_classes=num_classes)
    else:
        data = deeplabv3_aspp(output_stride=output_stride,
                              aspp_with_separable_conv=aspp_with_separable_conv,
                              non_local=non_local, oc_context=oc_context)
        data = bilinear_upsampling(data, output_stride=output_stride, num_classes=num_classes)
    return data


def _deeplabv3_aspp_backbone(output_stride, rate, non_local=False):
    data = resnet101_c4(output_stride=output_stride, non_local=non_local)
    data = extra_stage(data, multi_grid=(1*rate, 2*rate, 4*rate), stage="stage4")
    return data


def _deeplabv3_plus_aspp_backbone(output_stride, rate, aspp_with_separable_conv, non_local=False):
    feat, data = resnet101_c4(output_stride=output_stride, aspp_with_separable_conv=aspp_with_separable_conv, non_local=non_local)
    data = extra_stage(data, multi_grid=(1*rate, 2*rate, 4*rate), stage="stage4")
    return feat, data


def deeplabv3_aspp(output_stride, aspp_with_separable_conv, non_local=False, oc_context=False):
    # rate changes due to output stride
    # when os=16, rate=1, os=8, rate=2
    if aspp_with_separable_conv:
        rate = 16 // output_stride
        feat, data = _deeplabv3_plus_aspp_backbone(output_stride, rate, 
                                                   aspp_with_separable_conv,
                                                   non_local=non_local)
        data = atrous_spatial_pyramid_pooling(data, 
                                              rate, 
                                              aspp_with_separable_conv, 
                                              oc_context=oc_context)
        return feat, data
    else:
        rate = 16 // output_stride
        data = _deeplabv3_aspp_backbone(output_stride, rate, non_local=non_local)
        data = atrous_spatial_pyramid_pooling(data, 
                                              rate, 
                                              aspp_with_separable_conv, 
                                              oc_context=oc_context)
        return data
