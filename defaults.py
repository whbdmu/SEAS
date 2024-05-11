# -*- coding: UTF-8 -*-
# @Author: Yimin Jiang

from yacs.config import CfgNode as CN

_C = CN()

# -------------------------------------------------------------------------------------------------------------------- #
# Dataset                                                                                                              #
# -------------------------------------------------------------------------------------------------------------------- #
_C.DATASET = CN()
# Choose one from {'CUHK_SYSU', 'PRW'} as the dataset
_C.DATASET.TYPE = 'CUHK_SYSU'
_C.DATASET.PATH = ''
# Number of images per batch
_C.DATASET.BATCH_SIZE = 5
# -------------------------------------------------------------------------------------------------------------------- #
# SOLVER                                                                                                               #
# -------------------------------------------------------------------------------------------------------------------- #
_C.SOLVER = CN()

_C.SOLVER.MAX_EPOCHS = 20

# Learning rate settings
_C.SOLVER.OPTIMIZER = 'Adam'
_C.SOLVER.BASE_LR = 0.0001
_C.SOLVER.WEIGHT_DECAY = 0.0
_C.SOLVER.BIAS_LR_FACTOR = 1.0
_C.SOLVER.BIAS_DECAY = 0.0
_C.SOLVER.SGD_MOMENTUM = 0.0

# The epoch milestones to decrease the learning rate by GAMMA
_C.SOLVER.LR_DECAY_MILESTONES = [8, 14]
_C.SOLVER.LR_DECAY_GAMMA = 0.1
_C.SOLVER.WARMUP_FACTOR = 0.001
_C.SOLVER.WARMUP_EPOCHS = 1

# Set to negative value to disable gradient clipping
_C.SOLVER.CLIP_GRADIENTS = 10.0
_C.SOLVER.AMP = True
# -------------------------------------------------------------------------------------------------------------------- #
# Evaluator                                                                                                            #
# -------------------------------------------------------------------------------------------------------------------- #
_C.EVAL = CN()
# The period to evaluate the model during training
_C.EVAL.PERIOD = 1
_C.EVAL.DETECTION_IOU_THRESHOLD = 0.5
# If set to -1, the standard threshold will be used
_C.EVAL.SEARCH_IOU_THRESHOLD = 0.5
_C.EVAL.TOP_K = [1, 5, 10]
# -------------------------------------------------------------------------------------------------------------------- #
# Misc                                                                                                                 #
# -------------------------------------------------------------------------------------------------------------------- #
# The device loading the model
_C.DEVICE = 'cuda'
# Set seed to negative to fully randomize everything
_C.SEED = 1
# -------------------------------------------------------------------------------------------------------------------- #
# Model                                                                                                                #
# -------------------------------------------------------------------------------------------------------------------- #
_C.MODEL = CN()

_C.MODEL.TRANSFORM = CN()
_C.MODEL.TRANSFORM.MIN_SIZE = 900
_C.MODEL.TRANSFORM.MAX_SIZE = 1500
_C.MODEL.TRANSFORM.IMG_MEAN = [0.485, 0.456, 0.406]
_C.MODEL.TRANSFORM.IMG_STD = [0.229, 0.224, 0.225]

# Choose one from {'SOLIDER', 'ConvNeXt', 'ResNet'} as the backbone
_C.MODEL.BACKBONE = 'ConvNeXt'

_C.MODEL.RPN = CN()
_C.MODEL.RPN.ANCHOR_SIZE = ((32, 64, 128, 256, 512),)
_C.MODEL.RPN.ANCHOR_RATIO = ((0.5, 1.0, 2.0),)
# NMS threshold used on RoIs
_C.MODEL.RPN.NMS_THRESH = 0.7
# Number of anchors per image used to train RPN
_C.MODEL.RPN.BATCH_SIZE_TRAIN = 256
# Target fraction of foreground examples per RPN minibatch
_C.MODEL.RPN.POS_FRAC_TRAIN = 0.5
# Overlap threshold for an anchor to be considered foreground (if >= POS_THRESH_TRAIN)
_C.MODEL.RPN.POS_THRESH_TRAIN = 0.7
# Overlap threshold for an anchor to be considered background (if < NEG_THRESH_TRAIN)
_C.MODEL.RPN.NEG_THRESH_TRAIN = 0.3
# Number of top scoring RPN RoIs to keep before applying NMS
_C.MODEL.RPN.PRE_NMS_TOPN_TRAIN = 12000
_C.MODEL.RPN.PRE_NMS_TOPN_TEST = 6000
# Number of top scoring RPN RoIs to keep after applying NMS
_C.MODEL.RPN.POST_NMS_TOPN_TRAIN = 2000
_C.MODEL.RPN.POST_NMS_TOPN_TEST = 300

_C.MODEL.DETECTION = CN()
_C.MODEL.DETECTION.SAMPLE = CN()
_C.MODEL.DETECTION.SAMPLE.POS_THRESH = 0.5
_C.MODEL.DETECTION.SAMPLE.NEG_THRESH = 0.5
_C.MODEL.DETECTION.SAMPLE.BATCH_SIZE = 128
_C.MODEL.DETECTION.SAMPLE.POS_FRAC = 0.5
_C.MODEL.DETECTION.FEAT_MAP_SIZE = (14, 14)
_C.MODEL.DETECTION.QUALITY = False
_C.MODEL.DETECTION.BOX_REG_LOSS_TYPE = 'smooth_l1'
_C.MODEL.DETECTION.QUALITY_LOSS_TYPE = 'iou'
_C.MODEL.DETECTION.POST_PROCESS = CN()
_C.MODEL.DETECTION.POST_PROCESS.SCORE_THRESH = 0.5
_C.MODEL.DETECTION.POST_PROCESS.NMS_THRESH = 0.4
_C.MODEL.DETECTION.POST_PROCESS.DETECTIONS_PER_IMAGE = 300

_C.MODEL.REID = CN()
_C.MODEL.REID.SAMPLE = CN()
_C.MODEL.REID.SAMPLE.POS_THRESH = 0.7
_C.MODEL.REID.SAMPLE.NEG_THRESH = 0.3
_C.MODEL.REID.SAMPLE.BATCH_SIZE = 128
_C.MODEL.REID.SAMPLE.POS_FRAC = 0.5
_C.MODEL.REID.FEAT_MAP_SIZE = (24, 12)
_C.MODEL.REID.DIM_IDENTITY = 1024  # dimensions of both the feature embedding and the feature representation
_C.MODEL.REID.FEAT_MAP_USED = 'OriginalAndDownsample'  # {'Downsample', 'OriginalAndDownsample'} feature map used
_C.MODEL.REID.EMBEDDING = 'MGE'  # {'MGE', 'GFE'} embedding type
_C.MODEL.REID.EMBEDDING_MGE = CN()
_C.MODEL.REID.EMBEDDING_MGE.NUM_BRANCHES = 3
_C.MODEL.REID.EMBEDDING_MGE.DROP_PATH = 0.1
_C.MODEL.REID.BNR = CN()
_C.MODEL.REID.BNR.MAPPING = 'MBN'  # {'MBN', 'BN', 'Identity'} the mapping type of BNR loss
_C.MODEL.REID.BNR.MAPPING_MBN_MOMENTUM = 0.9
_C.MODEL.REID.EXTRACTOR = 'TripleConvHead'  # {'TripleConvHead', 'BackboneHead', 'Identity', None} noise extractor type
_C.MODEL.REID.DENOISER = 'CrossAttention'  # {'CrossAttention', 'LinearProjection', None}
_C.MODEL.REID.DENOISER_CA = CN()
_C.MODEL.REID.DENOISER_CA.POOL_TYPE = 'Max'  # {'Max', 'Avg'} type of adaptive pooling function for fg noise map
_C.MODEL.REID.DENOISER_CA.POOL_SIZE = (7, 7)  # output size of the pooling function
_C.MODEL.REID.DENOISER_CA.POS_EMB = 'Learnable2D'  # {'Learnable2D', 'Learnable1D', None} position embed of noise map
_C.MODEL.REID.DENOISER_CA.LAYER_NORM = True  # layer normalize the noise map
_C.MODEL.REID.DENOISER_CA.PD_NUM_HEADS = 8
_C.MODEL.REID.DENOISER_CA.PD_DIM_FFN = 2048
_C.MODEL.REID.DENOISER_CA.PD_DROPOUT = 0.0
_C.MODEL.REID.LOSS = CN()
_C.MODEL.REID.LOSS.LUT_SIZE = 5532
_C.MODEL.REID.LOSS.CQ_SIZE = 5000
_C.MODEL.REID.LOSS.MOMENTUM = 0.5
_C.MODEL.REID.LOSS.SCALAR = 30.0
_C.MODEL.REID.LOSS.MARGIN = 0.25
_C.MODEL.REID.LOSS.WEIGHT_SOFTMAX = 1.0
_C.MODEL.REID.LOSS.WEIGHT_TRIPLET = 1.0

_C.MODEL.LOSS_WEIGHT = CN()
_C.MODEL.LOSS_WEIGHT.RPN_REG = 1.0
_C.MODEL.LOSS_WEIGHT.RPN_CLS = 1.0
_C.MODEL.LOSS_WEIGHT.PROPOSAL_REG = 10.0
_C.MODEL.LOSS_WEIGHT.PROPOSAL_CLS = 1.0
_C.MODEL.LOSS_WEIGHT.PROPOSAL_QLT = 1.0
_C.MODEL.LOSS_WEIGHT.BOX_BNR = 1.0
_C.MODEL.LOSS_WEIGHT.BOX_REID = 1.0

# Choose one from {'v1', 'v2'}
_C.MODEL.PARAM_INIT = 'v1'


def get_default_cfg():
    return _C.clone()
