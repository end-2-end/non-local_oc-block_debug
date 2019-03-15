import pdb
import os
import cv2
import multiprocessing

import logging
import time
import importlib
import numpy as np
from tensorboardX import SummaryWriter

import mxnet as mx
import core.metrics as metrics
from collections import namedtuple
from core.module import MutableModule
from utils.evl_seg import load_params, calculate, get_data, fasttrainID2labels

BatchEndParam = namedtuple('BatchEndParams', ['epoch', 'nbatch', 'eval_metric'])

def resize_wrapper(i):
    if result_list[i] is None:
        return cv2.resize(unresized_result_list[i],
                     tuple(val.output_size), interpolation=cv2.INTER_LINEAR)
    return result_list[i] + cv2.resize(unresized_result_list[i],
                     tuple(val.output_size), interpolation=cv2.INTER_LINEAR)

def eval_IOU(epoch, val, args_params, auxs_params, mpool, logger, writer, visualize=False):
    bind_shape = 0
    tp = [0.0] * 21
    denom = [0.0] * 21

    num_class = val.num_class
    root_dir = val.root_dir
    epoch = epoch
    ctx = mx.gpu(val.test_gpus)
    scales = val.scales
    enet_args = args_params
    enet_auxs = auxs_params
    enet = importlib.import_module(
                'symbols.' + val.network).get_symbol(val.num_class,
                                                     val.output_stride,
                                                     val.use_refine_decoder,
                                                     val.use_aspp_with_separable_conv,
                                                     non_local=val.non_local, 
                                                     oc_context=val.oc_context)
    enet = mx.symbol.contrib.BilinearResize2D(enet, 
                                  height=val.output_size[0], width=val.output_size[1])
    lines = open(os.path.join(val.root_dir, val.val_list)).read().splitlines()
    num_image = len(lines)
    outlist = []
    result_list = [None] * num_image
    unresized_result_list = [None] * num_image
    for scale in scales:
        for idx, line in enumerate(lines):
            data_img_name, label_img_name = line.strip('\n').split("\t")
            label_name = os.path.join(root_dir, label_img_name)
            filename = os.path.join(root_dir, data_img_name)
            index = 0
            for flip in range(val.flip+1):
                img_data, img_label, nh, nw = get_data(filename, label_name, scale, flip, val.output_size)
                if bind_shape != img_data.shape[1]:
                    exector = enet.simple_bind(ctx, data=(1, 3, img_data.shape[1], 
                                           img_data.shape[2]), grad_req="null")
                    bind_shape = img_data.shape[1]
                exector.copy_params_from(enet_args, enet_auxs, False)
                img_data = np.expand_dims(img_data, 0)
                data = mx.nd.array(img_data, ctx)
                exector.forward(is_train=False, data=data)
                if flip == 1:			# CxHxW
                    output = exector.outputs[0].flip(axis=2).asnumpy()
                else:
                    output = exector.outputs[0].asnumpy()
                output = np.squeeze(output)
                if flip == 1:
                    labels += output
                    labels /= 2.0
                else:
                    labels = output
            unresized_result_list[idx] = labels.transpose([1,2,0])	# HxWxC

            if result_list[idx] is None:
                result_list[idx] = labels
            else:
                result_list[idx] += labels

            print(idx)

        """
        for idx, i in enumerate(unresized_result_list):
            unresized_result_list[idx]=[i, tuple(val.output_size), cv2.INTER_LINEAR, result_list[idx]]
        result_list = mpool.map(resize_wrapper, range(len(unresized_result_list)))

        for idx, labels in enumerate(unresized_result_list):		# HxWxC
            if result_list[idx] is None:
                result_list[idx] = labels
            else:
                result_list[idx] += labels
        """


    for idx, line in enumerate(lines):
        if result_list[idx] is None:
            continue
        output = result_list[idx]/float(len(scales))
        #if np.isnan(np.mean(output)):
        #    print(np.mean(output))
        #    sys.exit(0)
        pre_label = np.argmax(output, axis=-1)				# HxWxC
        iou = calculate(pre_label, img_label, num_class, tp, denom)
        #print(iou)

        if visualize:
            heat = np.max(output, axis=0)
            heat *= 255
            heat_map = cv2.applyColorMap(heat.astype(np.uint8), cv2.COLORMAP_JET)

            pre_label, result = fasttrainID2labels(pre_label, num_class)
            img = cv2.imread(filename)
            gt_label = cv2.imread(label_name)
            gt_label = np.transpose(gt_label, [2, 0, 1])
            _label = np.zeros((3, 512, 512), np.single)
            _label[:, :gt_label.shape[1], :gt_label.shape[2]] = gt_label
            _label = np.transpose(_label, [1, 2, 0])
            img = np.transpose(img, [2, 0, 1])
            _img_data = np.zeros((3, 512, 512), np.single)
            _img_data[:, :img.shape[1], :img.shape[2]] = img
            _img_data = np.transpose(_img_data, [1, 2, 0])
    
            visual = np.zeros((result.shape[0] * 2, result.shape[1] * 2, 3))
            visual[result.shape[0]:, :result.shape[1], :] = result
            visual[result.shape[0]:, result.shape[1]:, :] = _label
            visual[:result.shape[0], result.shape[1]:, :] = heat_map
            visual[:result.shape[0], :result.shape[1], :] = _img_data

            if not os.path.exists('./visual'):
                os.makedirs('./visual')

            name = label_img_name.split('/')[-1]
            #cv2.imwrite('./visual/'+name,visual)

    id2name = ['road', 'sidewalk',  'building',  'wall',  'fence',  'pole',  'traffic_light', 'traffic_sign',  'vegetation',  'terrain',  'sky', 'person',  'rider',  'car',  'truck',  'bus', 'train',  'motorcycle',  'bicycle']

    for c in range(num_class):
        temp = tp[c] / (denom[c] + 1e-6)
        writer.add_scalar(id2name[c], temp, epoch)
    writer.add_scalar('Aug-Validation', iou, epoch)
    logger.info('Epoch[%d] Aug-Validation-%s=%f', epoch, miou, iou)


class Solver(object):
    def __init__(self, symbol, ctx=None,
                 begin_epoch=0, end_epoch=None,
                 arg_params=None, aux_params=None, opt_states=None, num_class=21):
        self.symbol = symbol
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.begin_epoch = begin_epoch
        self.end_epoch = end_epoch
        self.arg_params = arg_params
        self.aux_params = aux_params
        self.opt_states = opt_states
        self.num_class = num_class

        self.mpool = multiprocessing.Pool(50)

    def _reset_bind(self):
        """Internal function to reset binded state."""
        self.binded = False
        self._exec_group = None
        self._data_shapes = None
        self._label_shapes = None

    def fit(self, prefix="",start_evl=100,train_data=None, eval_data=None, eval_metric='acc',
            validate_metric=None, work_load_list=None,
            max_data_shapes=None, max_label_shapes=None,
            epoch_end_callback=None, batch_end_callback=None,
            fixed_param_regex=None, initializer=None, optimizer=None,
            optimizer_params=None, logger=None, kvstore='local', p=None):
        writer = SummaryWriter(os.path.join('/',*prefix.split('/')[:-1]))

        if logger is None:
            logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        if validate_metric is None:
            validate_metric = eval_metric

        # set module
        mod = MutableModule(symbol=self.symbol,
                            data_names=[x[0] for x in train_data.provide_data],
                            label_names=[x[0] for x in train_data.provide_label],
                            logger=logger,
                            context=self.ctx,
                            preload_opt_states=self.opt_states,
                            work_load_list=work_load_list,
                            max_data_shapes=max_data_shapes,
                            max_label_shapes=max_label_shapes,
                            fixed_param_regex=fixed_param_regex
                            )

        mod.bind(data_shapes=max_data_shapes, label_shapes=max_label_shapes)
        self.arg_params['oc_W_weight'] = mx.nd.zeros([256,256,1,1])
        #self.arg_params['nonlocal_conv44'] = mx.nd.zeros([1024,1024,1,1])
        mod.init_params(initializer=initializer,
                        arg_params=self.arg_params,
                        aux_params=self.aux_params,
                        allow_missing=True)
        mod.init_optimizer(kvstore=kvstore,
                           optimizer=optimizer,
                           optimizer_params=optimizer_params)

        metric_tra = []
        metric_lst = []
        eval_metric1 = metrics.AccWithIgnoreMetric(ignore_label=255)
        metric_tra.append(eval_metric1)
        metric_lst.append(metrics.IoUMetric(ignore_label=255, label_num=self.num_class, name='IOU'))
        validate_metric = mx.metric.CompositeEvalMetric(metrics=metric_lst)
        eval_metric = mx.metric.CompositeEvalMetric(metrics=metric_tra)

        # training loop
        for epoch in range(self.begin_epoch, self.end_epoch):
            tic = time.time()
            eval_metric.reset()


            # evaluation
            print('score')
            if eval_data and (epoch>=start_evl) and (epoch%2==0):
                if (p.test_gpus is not None) and epoch%10==0:
                    arg_params, aux_params = mod.get_params()
                    eval_IOU(epoch, p, arg_params, aux_params, self.mpool,
                             	logger, writer, visualize=False)
                else:
                    res = mod.score(eval_data, validate_metric)
                    for name, val in res:
                        logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)
                        writer.add_scalar(name, val, epoch)
                    validate_metric.reset()


            for nbatch, data in enumerate(train_data, 1):
                mod.forward(data_batch=data, is_train=True)
                mod.backward()
                mod.update_metric(eval_metric, data.label)
                mod.update()

                if batch_end_callback is not None:
                    batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch, eval_metric=eval_metric)
                    batch_end_callback(batch_end_params)

            # one epoch training is finished
            for name, val in eval_metric.get_name_value():
                logger.info('Epoch[%d] Train-%s=%f', epoch, name, val)
            toc = time.time()
            logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc - tic))
            if epoch_end_callback is not None:
                arg_params, aux_params = mod.get_params()
                if epoch > 100 and (epoch%20)==0:
                    mod.save_checkpoint(prefix, epoch+1, True)
                mod.set_params(arg_params, aux_params)

            # one epoch training is finished, reset train set
            train_data.reset()
            eval_data.reset()
