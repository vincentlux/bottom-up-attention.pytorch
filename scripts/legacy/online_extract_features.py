import argparse
import os
import sys
import torch
# import tqdm
import cv2
import numpy as np
sys.path.append('detectron2')

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.structures import Instances

# from utils.progress_bar import ProgressBar
from frcnn_ext_models import add_config
from frcnn_ext_models.bua.layers.nms import nms

from tqdm import tqdm
from typing import List
# import ray
# from ray.actor import ActorHandle

PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
TEST_SCALES = (600,)
TEST_MAX_SIZE = 1000


class Pack(dict):
    def __getattr__(self, name):
        return self[name]

    def clone_dict(self, x):
        for k, v in list(x.items()):
            self[k] = v

    def add(self, **kwargs):
        for k, v in list(kwargs.items()):
            self[k] = v

    def copy(self):
        pack = Pack()
        for k, v in list(self.items()):
            if type(v) is list:
                pack[k] = list(v)
            else:
                pack[k] = v
        return pack

def load_vocab_file(filename):
    objects = []
    with open(filename) as f:
        for i in f.readlines():
            objects.append(i.strip())
    id_objects_map = {i: v for i, v in enumerate(objects)}
    return id_objects_map

def switch_extract_mode(mode):
    if mode == 'roi_feats':
        switch_cmd = ['MODEL.BUA.EXTRACTOR.MODE', 1]
    elif mode == 'bboxes':
        switch_cmd = ['MODEL.BUA.EXTRACTOR.MODE', 2]
    elif mode == 'bbox_feats':
        switch_cmd = ['MODEL.BUA.EXTRACTOR.MODE', 3, 'MODEL.PROPOSAL_GENERATOR.NAME', 'PrecomputedProposals']
    else:
        print('Wrong extract mode! ')
        exit()
    return switch_cmd

def set_min_max_boxes(min_max_boxes):
    if min_max_boxes == 'min_max_default':
        return []
    try:
        min_boxes = int(min_max_boxes.split(',')[0])
        max_boxes = int(min_max_boxes.split(',')[1])
    except:
        print('Illegal min-max boxes setting, using config default. ')
        return []
    cmd = ['MODEL.BUA.EXTRACTOR.MIN_BOXES', min_boxes,
            'MODEL.BUA.EXTRACTOR.MAX_BOXES', max_boxes]
    return cmd


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_config(args, cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(switch_extract_mode(args.extract_mode))
    cfg.merge_from_list(set_min_max_boxes(args.min_max_boxes))
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def get_image_blob(im, pixel_means):
    """Converts an image into a network input.
    Arguments:
        im (ndarray): a color image
    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    pixel_means = np.array([[pixel_means]])
    dataset_dict = {}
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= pixel_means

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    for target_size in TEST_SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > TEST_MAX_SIZE:
            im_scale = float(TEST_MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)

    dataset_dict["image"] = torch.from_numpy(im).permute(2, 0, 1)
    dataset_dict["im_scale"] = im_scale

    return dataset_dict

def normalize_box_feats(boxes, im_h, im_w):
    '''
    input: 10 * 1d array of len 4 (xmin, ymin, xmax, ymax); img height; img width
    output: np array with shape (num_boxes, 8)
    8: (xmin, ymin, xmax, ymax, xcent, ycent, wbox, hbox) normalized to -1,1
    '''
    # print(f'img width:{im_w} img height:{im_h}')
    # print(f'box 0: {boxes[0]}')
    # print(f'boxes: {boxes}')
    assert(np.all(boxes[:, 0] <= im_w) and np.all(boxes[:, 2] <= im_w))
    assert(np.all(boxes[:, 1] <= im_h) and np.all(boxes[:, 3] <= im_h))
    feats = np.zeros((boxes.shape[0], 6))

    feats[:, 0] = boxes[:, 0] * 2.0 / im_w - 1  # xmin
    feats[:, 1] = boxes[:, 1] * 2.0 / im_h - 1  # ymin
    feats[:, 2] = boxes[:, 2] * 2.0 / im_w - 1  # xmax
    feats[:, 3] = boxes[:, 3] * 2.0 / im_h - 1  # ymax
    # feats[:, 4] = (feats[:, 0] + feats[:, 2]) / 2  # xcenter
    # feats[:, 5] = (feats[:, 1] + feats[:, 3]) / 2  # ycenter
    feats[:, 4] = feats[:, 2] - feats[:, 0]  # box width
    feats[:, 5] = feats[:, 3] - feats[:, 1]  # box height
    return feats


class FRCNNExtractor(object):
    def __init__(self, config_file, mode='caffe', extract_mode='roi_feats', min_max_boxes='10,50'):
        args = {}
        args['config_file'] = config_file
        args['mode'] = mode
        args['extract_mode'] = extract_mode
        args['min_max_boxes'] = min_max_boxes
        args['eval_only'] = True
        args = Pack(args)
        cfg = setup(args)

        self.model = DefaultTrainer.build_model(cfg)
        DetectionCheckpointer(self.model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=False
        )
        self.model.eval()
        self.cfg = cfg
        self.vg_objects = load_vocab_file('objects_vocab.txt')

    def post_process(self, cfg, im, dataset_dict, boxes, scores, features_pooled, attr_scores=None):
        MIN_BOXES = cfg.MODEL.BUA.EXTRACTOR.MIN_BOXES
        MAX_BOXES = cfg.MODEL.BUA.EXTRACTOR.MAX_BOXES
        CONF_THRESH = cfg.MODEL.BUA.EXTRACTOR.CONF_THRESH

        dets = boxes[0] / dataset_dict['im_scale']
        scores = scores[0]
        feats = features_pooled[0]

        max_conf = torch.zeros((scores.shape[0])).to(scores.device)
        for cls_ind in range(1, scores.shape[1]):
                cls_scores = scores[:, cls_ind]
                keep = nms(dets, cls_scores, 0.3)
                max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep],
                                                cls_scores[keep],
                                                max_conf[keep])

        keep_boxes = torch.nonzero(max_conf >= CONF_THRESH).flatten()
        if len(keep_boxes) < MIN_BOXES:
            keep_boxes = torch.argsort(max_conf, descending=True)[:MIN_BOXES]
        elif len(keep_boxes) > MAX_BOXES:
            keep_boxes = torch.argsort(max_conf, descending=True)[:MAX_BOXES]
        image_feat = feats[keep_boxes]
        image_bboxes = dets[keep_boxes]
        # image_objects_conf = np.max(scores[keep_boxes].numpy()[:,1:], axis=1)
        image_objects = np.argmax(scores[keep_boxes].numpy()[:,1:], axis=1)

        image_h = np.size(im, 0)
        image_w = np.size(im, 1)
        loc_feat = normalize_box_feats(image_bboxes.numpy(), image_h, image_w)
        feat = np.concatenate((image_feat.numpy(), loc_feat), axis=1)


        objects = ' '.join([self.vg_objects[i] for i in image_objects])

        info = {
            'objects': objects,
            'img_feat': feat
        }

        return info

    def batch_extract_feat(self, imgs: List[np.ndarray], batch_size: int = 1):
        if batch_size > 1:
            raise NotImplementedError
        img_feat_list = []

        for img in tqdm(imgs):
            # im = cv2.imread(im_file)
            dataset_dict = get_image_blob(img, self.cfg.MODEL.PIXEL_MEAN)
            # extract roi features
            if self.cfg.MODEL.BUA.EXTRACTOR.MODE == 1:
                attr_scores = None
                with torch.set_grad_enabled(False):
                    if self.cfg.MODEL.BUA.ATTRIBUTE_ON:
                        boxes, scores, features_pooled, attr_scores = self.model([dataset_dict])
                    else:
                        boxes, scores, features_pooled = self.model([dataset_dict])
                boxes = [box.tensor.cpu() for box in boxes]
                scores = [score.cpu() for score in scores]
                features_pooled = [feat.cpu() for feat in features_pooled]
                if attr_scores is not None:
                    attr_scores = [attr_score.cpu() for attr_score in attr_scores]

                img_feat = self.post_process(self.cfg, img, dataset_dict,
                    boxes, scores, features_pooled, attr_scores)
                img_feat_list.append(img_feat)

        return img_feat_list



if __name__ == "__main__":
    extractor = FRCNNExtractor('configs/bua-caffe/extract-bua-caffe-r101.yaml')
    # img_dir = '/home/vincent/proj/soco/soco/soco-image-sparta/data/coco/train2014'
    img_dir = './datasets/demo'
    img_path_list = os.listdir(img_dir)
    budget = 10
    imgs = []
    for i, img in enumerate(img_path_list):
        imgs.append(cv2.imread(os.path.join(img_dir, img)))
        if i == budget:
            break

    results = extractor.batch_extract_feat(imgs)
