from __future__ import absolute_import
from datetime import datetime
import sys
import os
import cv2
import argparse
import logging
import importlib
import numpy as np
import mxnet as mx
print(mx.__version__)

from config.deeplabv3_city import config
from config.deeplabv3_city import train as p
from config.deeplabv3_city import val as p_val
from core.iter.iterator_multiprocess_city import FileIter
from core.solver import Solver
from utils.lr_scheduler import WarmupMultiFactorScheduler
import utils.memonger as memonger
import utils.functions as func


## generate fake inputs to make number of validation samples the int times as batchsize
def generate_dummy_image(config=None):
    val_list = config.val_list + '.new'
    lines = open(os.path.join(config.root_dir, config.val_list)).read().splitlines()
    num = len(lines)
    data_img_name, label_img_name = lines[0].strip('\n').split('\t')
    data_image_root = data_img_name.split('/')[0]
    label_image_root = label_img_name.split('/')[0]
    dummy_image_path = os.path.join(data_image_root, 'dummy.jpg')
    dummy_label_path = os.path.join(label_image_root, 'dummy.png')
    dummy_image = np.zeros((1024, 2048, 3))
    dummy_label = np.ones((1024, 2048)) * 255
    cv2.imwrite(os.path.join(config.root_dir, dummy_image_path), dummy_image)
    cv2.imwrite(os.path.join(config.root_dir, dummy_label_path), dummy_label)
    dummy_str = dummy_image_path + '\t' + dummy_label_path
    if num % config.batch_size == 0:
        return config.val_list
    else:
        num_pad = config.batch_size - (num % config.batch_size)
        for n in range(num_pad):
            lines.append(dummy_str)
        lst = open(os.path.join(config.root_dir, val_list), 'w')
        for line in lines:
            lst.write('{}\n'.format(line))
        return val_list


def train_enet():
    from symbols import deeplab_v3

    # handle gpu
    ctx = [mx.gpu(_) for _ in p.gpus]
    p_val.batch_size *= len(ctx)

    # setup multi-gpu
    p.batch_size *= len(ctx)

    # print out setting
    p_attrs = vars(p)

    # loadding data
    train_iter = FileIter(p.train_list, p=p, rgb_mean=(72.30608881, 82.09696889, 71.60167789))	# cityscape rgb mean:(72.30608881, 82.09696889, 71.60167789)
    if os.path.exists(os.path.join(p_val.root_dir, p.val_list)):
        val_list = generate_dummy_image(config=p_val)
        print(val_list)
        eval_iter = FileIter(val_list, p=p_val, rgb_mean=(72.30608881, 82.09696889, 71.60167789))
    else:
        print(os.path.join(p_val.root_dir, p.val_list), ' not exist')
        eval_iter = None

    # infer max shape
    max_data_shapes = [('data', (p.batch_size, 3, p.output_size[0], p.output_size[1]))]

    max_label_shapes = [('softmax_label', (p.batch_size,
                                           p.output_size[0] // p.fac,  p.output_size[1] // p.fac))]

    # init model or load trained model
    opt_states = None
    arg_params, aux_params = None, None

    if p.resume:
        arg_params, aux_params = func.load_params(p.pretrained_prefix, p.pretrained_epoch)

    net_symbol = importlib.import_module(
        'symbols.' + p.network).get_symbol(p.num_class,
                                           p.output_stride,
                                           p.use_refine_decoder,
                                           p.use_aspp_with_separable_conv,
                                           non_local=p.non_local,
                                           oc_context=p.oc_context)

    if not config.use_sync_bn:
        net_symbol = memonger.search_plan(net_symbol,
                data=(p.batch_size // len(ctx), 3, p.output_size[0], p.output_size[1]),
                softmax_label=(p.batch_size // len(ctx), 1,
                    p.output_size[0] // p.fac, p.output_size[1] // p.fac))

    model = Solver(net_symbol, ctx=ctx, begin_epoch=p.begin_epoch,
                   end_epoch=p.end_epoch,
                   arg_params=arg_params,
                   aux_params=aux_params,
                   opt_states=opt_states,
                   num_class=p.num_class)

    # initializer
    initializer = mx.initializer.Xavier(magnitude=2., rnd_type='gaussian', factor_type="in")
    initializer.set_verbosity(verbose=True)

    # optimizer
    optimizer = p.optimizer
    kv = mx.kvstore.create(p.kvstore)
    epoch_size = max(int( train_iter.num_data / p.batch_size / kv.num_workers), 1)
    print("1 epoch contains {} iterations".format(epoch_size))
    begin_epoch = p.begin_epoch
    steps = p.steps
    lr_iters = [epoch_size * (x - begin_epoch) for x in steps if x - begin_epoch > 0]
    num_epoch = epoch_size * p.end_epoch
    lr_sch = WarmupMultiFactorScheduler(lr_iters, lr_policy=p.lr_policy, num_epoch=num_epoch,
                                        warmup=p.warmup, warmup_lr=p.warmup_lr, warmup_step=p.warmup_step)
    optimizer_params = {
        'wd': p.wd,
        'learning_rate': p.lr,
        'rescale_grad': (1.0 / len(ctx)),	##### lr is changed when num of gpu changed #####
        'lr_scheduler':lr_sch
    }
    print("batch size over multiple GPUs: {}".format(p.batch_size))
    if p.optimizer == 'SGD':
        print('------------------------use SGD------------------------------------')
        optimizer_params.update(**{'momentum':p.momentum,})

    # callback
    batch_end_callback = mx.callback.Speedometer(train_iter.batch_size, frequent=p.frequent)

    # set model save path
    filename = os.path.join('/home/kfxw/ProjectFiles/Python_scripts/tusimple_seg/model', p.save_prefix+'_'+datetime.now().strftime('%Y_%m_%d_%H_%M'))
    if not os.path.exists(filename):
        os.makedirs(filename)

    #model_path = os.path.join('./model', p.save_prefix)
    model_path = filename
    model_full_path = filename

    # seg log
    func.save_log(p.save_prefix, model_full_path)
    logging.info('---------------------------TIME-------------------------------')
    logging.info('-------------------{}------------------------'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    for k, v in sorted(p_attrs.items(), key=lambda _: _[0]):
        logging.info("%s : %s", k, v)

    prefix = os.path.join(model_full_path, p.save_prefix)
    epoch_end_callback = mx.callback.module_checkpoint(model, prefix, period=5, save_optimizer_states=True)

    # evaluation metric
    eval_metric = ['f1']

    model.fit(prefix=prefix,
              start_evl=p.start_evl,
              train_data=train_iter, eval_data=eval_iter,
              eval_metric=eval_metric,
              max_data_shapes=max_data_shapes,
              max_label_shapes=max_label_shapes,
              batch_end_callback=batch_end_callback,
              epoch_end_callback=epoch_end_callback,
              fixed_param_regex=p.fixed_param_regex,
              initializer=initializer,
              optimizer=optimizer,
              optimizer_params=optimizer_params,
              kvstore=p.kvstore,
	      p=p_val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Model for Image Segmentation")
    parser.add_argument('--root_dir',                     type=str,   default=p.root_dir)
    parser.add_argument('--train_list',                   type=str,   default=p.train_list)
    parser.add_argument('--val_list',                     type=str,   default=p.val_list)
    parser.add_argument('--begin_epoch',                  type=int,   default=p.begin_epoch)
    parser.add_argument('--end_epoch',                    type=int,   default=p.end_epoch)
    parser.add_argument('--gpus',                         type=int,   nargs="+")
    parser.add_argument('--test_gpus',                    type=int,   nargs="?", default=None)
    parser.add_argument('--output_stride',                type=int,   default=p.output_stride)
    parser.add_argument('--batch_size',                   type=int,   default=p.batch_size)
    parser.add_argument('--frequent',                     type=int,   default=p.frequent)
    parser.add_argument('--kvstore',                      type=str,   default=p.kvstore)
    parser.add_argument('--network',                      type=str,   default=p.network)
    parser.add_argument('--save_prefix',                  type=str,   default=p.save_prefix)
    parser.add_argument('--pretrained_prefix',            type=str,   default=p.pretrain_prefix)
    parser.add_argument('--pretrained_epoch',             type=int,   default=p.pretrain_epoch)
    parser.add_argument('--fixed_param_regex',            type=str,   default=p.fixed_param_regex)
    parser.add_argument('--steps',                        type=int,   default=p.steps, nargs="+")
    parser.add_argument('--lr',                           type=float, required=True)
    parser.add_argument('--resume',                       action='store_true')
    parser.add_argument('--use_refine_decoder',           action='store_true')
    parser.add_argument('--use_aspp_with_separable_conv', action='store_true')
    args = vars(parser.parse_args())

    for k in args:
        p[k] = args[k]
    for k in p:
        config[k] = p[k]
    p_val['test_gpus'] = args['test_gpus']
    p_val['output_stride'] = args['output_stride']
    p_val['use_refine_decoder'] = args['use_refine_decoder']
    p_val['use_aspp_with_separable_conv'] = args['use_aspp_with_separable_conv']

    train_enet()
