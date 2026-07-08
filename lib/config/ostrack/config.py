from easydict import EasyDict as edict
import yaml

"""
Default config for OSTrack.
"""
cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.PRETRAIN_FILE = "mae_pretrain_vit_base.pth"
cfg.MODEL.EXTRA_MERGER = False
cfg.MODEL.RETURN_INTER = False
cfg.MODEL.RETURN_STAGES = []
cfg.MODEL.NUM_OBJECT_QUERIES = 1

cfg.MODEL.MOTION = edict()
cfg.MODEL.MOTION.ENABLE = False
cfg.MODEL.MOTION.VERSION = "v1"
cfg.MODEL.MOTION.HISTORY_LEN = 1
cfg.MODEL.MOTION.HIDDEN_DIM = 16
cfg.MODEL.MOTION.NUM_HEADS = 4
cfg.MODEL.MOTION.NUM_LAYERS = 1
cfg.MODEL.MOTION.RING_INNER = 0
cfg.MODEL.MOTION.RING_OUTER = 2
cfg.MODEL.MOTION.DROPOUT = 0.0
cfg.MODEL.MOTION.TOKEN_GATE = False
cfg.MODEL.MOTION.TOKEN_GATE_ALPHA = 0.5
cfg.MODEL.MOTION.PROMPT_ENABLE = False
cfg.MODEL.MOTION.PROMPT_NUM = 4
cfg.MODEL.MOTION.SCORE_FUSION = False
cfg.MODEL.MOTION.SCORE_FUSION_ALPHA = 1.0
cfg.MODEL.MOTION.SCORE_FUSION_RESIDUAL_SCALE = 1.0
cfg.MODEL.MOTION.SCORE_FUSION_RELIABILITY = False
cfg.MODEL.MOTION.COUNTERFACTUAL_ENABLE = False
cfg.MODEL.MOTION.COUNTERFACTUAL_MODE = "batch_roll"
cfg.MODEL.MOTION.COUNTERFACTUAL_ERASE_SCALE = 1.0

cfg.MODEL.EXIST = edict()
cfg.MODEL.EXIST.ENABLE = False
cfg.MODEL.EXIST.THRESHOLD = 0.5

# MODEL.BACKBONE
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = "vit_base_patch16_224"
cfg.MODEL.BACKBONE.STRIDE = 16
cfg.MODEL.BACKBONE.MID_PE = False
cfg.MODEL.BACKBONE.SEP_SEG = False
cfg.MODEL.BACKBONE.CAT_MODE = "direct"
cfg.MODEL.BACKBONE.MERGE_LAYER = 0
cfg.MODEL.BACKBONE.ADD_CLS_TOKEN = False
cfg.MODEL.BACKBONE.OUT_INDICES = []

# Kept for compatibility with shared FocusTrack helpers.
cfg.MODEL.DECODER = edict()
cfg.MODEL.DECODER.ADD_DECODER = False
cfg.MODEL.DECODER.DECODER_DIM = 384
cfg.MODEL.DECODER.NUM_LAYERS = 3
cfg.MODEL.DECODER.PRETRAIN_FILE = None

# MODEL.HEAD
cfg.MODEL.HEAD = edict()
cfg.MODEL.HEAD.BOX_HEAD = edict()
cfg.MODEL.HEAD.BOX_HEAD.TYPE = "CENTER"
cfg.MODEL.HEAD.BOX_HEAD.NUM_CHANNELS = 256

cfg.MODEL.HEAD.CLS_HEAD = edict()
cfg.MODEL.HEAD.CLS_HEAD.USE_CLS_TOKEN = False
cfg.MODEL.HEAD.CLS_HEAD.AVERAGE_POOL = False
cfg.MODEL.HEAD.CLS_HEAD.AVERAGE_POOL_TYPE = "avg_pool"
cfg.MODEL.HEAD.CLS_HEAD.HEAD_TYPE = "linear"
cfg.MODEL.HEAD.CLS_HEAD.NUM_CLASSES = 2

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 500
cfg.TRAIN.LR_DROP_EPOCH = 400
cfg.TRAIN.BATCH_SIZE = 16
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.FOCALLOSS_WEIGHT = 1.0
cfg.TRAIN.MOTION_WEIGHT = 0.0
cfg.TRAIN.FUSION_WEIGHT = 0.0
cfg.TRAIN.FUSION_BETA = 0.5
cfg.TRAIN.MOTION_BG_WEIGHT = 0.0
cfg.TRAIN.MOTION_MARGIN_WEIGHT = 0.0
cfg.TRAIN.MOTION_CENTER_WEIGHT = 0.0
cfg.TRAIN.MOTION_MARGIN = 0.2
cfg.TRAIN.MOTION_CENTER_TEMP = 0.1
cfg.TRAIN.MOTION_CF_WEIGHT = 0.0
cfg.TRAIN.MOTION_CF_BG_WEIGHT = 0.0
cfg.TRAIN.MOTION_CF_REL_WEIGHT = 0.0
cfg.TRAIN.MOTION_CF_MARGIN = 0.2
cfg.TRAIN.MOTION_CF_REL_TEMP = 0.2
cfg.TRAIN.MOTION_CF_WARMUP_EPOCH = 0
cfg.TRAIN.MOTION_CF_RAMP_EPOCH = 0
cfg.TRAIN.EXIST_WEIGHT = 0.0
cfg.TRAIN.CE_WEIGHT = 0.0
cfg.TRAIN.FOCALMASK_WEIGHT = 0.0
cfg.TRAIN.FREEZE_LAYERS = [0]
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.VAL_EPOCH_INTERVAL = 20
cfg.TRAIN.VAL_START_EPOCH = 1
cfg.TRAIN.VAL_EVERY_EPOCH_AFTER = 0
cfg.TRAIN.FINAL_VAL_EPOCHS = 5
cfg.TRAIN.TEST_AFTER_VAL = False
cfg.TRAIN.TEST_DATASET_NAME = "antiuav410_test"
cfg.TRAIN.TEST_THREADS = 6
cfg.TRAIN.TEST_NUM_GPUS = 1
cfg.TRAIN.TEST_SKIP_MISSING_SEQ = False
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.AMP = False
cfg.TRAIN.SAVE_INTERVAL = 40
cfg.TRAIN.EXTRA_SAVE_EPOTH = []
cfg.TRAIN.TRAIN_SECOND_STAGE = False
cfg.TRAIN.DROP_PATH_RATE = 0.1

# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1

# DATA
cfg.DATA = edict()
cfg.DATA.SAMPLER_MODE = "causal"
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200

cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
cfg.DATA.TRAIN.POS_PROB = 1.0

cfg.DATA.VAL = edict()
cfg.DATA.VAL.DATASETS_NAME = ["GOT10K_votval"]
cfg.DATA.VAL.DATASETS_RATIO = [1]
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 10000

cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.SIZE = 320
cfg.DATA.SEARCH.FACTOR = 5.0
cfg.DATA.SEARCH.CENTER_JITTER = 4.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5
cfg.DATA.SEARCH.NUMBER = 1
cfg.DATA.SEARCH.RANDOM_JITTER_RATIO = 0.2
cfg.DATA.SEARCH.RANDOM_CENTER_JITTER = 0
cfg.DATA.SEARCH.RANDOM_SCALE_JITTER = 0.1

cfg.DATA.SEARCH.GRID = edict()
cfg.DATA.SEARCH.GRID.TYPE = "qp"
cfg.DATA.SEARCH.GRID.USE_GRID = False
cfg.DATA.SEARCH.GRID.SHAPE = [17, 17]

cfg.DATA.SEARCH.GRID.GENERATOR = edict()
cfg.DATA.SEARCH.GRID.GENERATOR.BANDWIDTH_SCALE = 64
cfg.DATA.SEARCH.GRID.GENERATOR.ZOOM_FACTOR = 1.5

cfg.DATA.SEARCH.GRID.GENERATOR.LOSS = edict()
cfg.DATA.SEARCH.GRID.GENERATOR.LOSS.NAMES = "asap_ts_weight"
cfg.DATA.SEARCH.GRID.GENERATOR.LOSS.WEIGHTS = [1, 1]

cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0
cfg.DATA.TEMPLATE.USE_GRID = False

# TEST
cfg.TEST = edict()
cfg.TEST.EPOCH = 500
cfg.TEST.TEMPLATE = edict()
cfg.TEST.TEMPLATE.FACTOR = 2.0
cfg.TEST.TEMPLATE.SIZE = 128
cfg.TEST.TEMPLATE.GRID = edict()
cfg.TEST.TEMPLATE.GRID.USE_GRID = False

cfg.TEST.SEARCH = edict()
cfg.TEST.SEARCH.SIZE = 320
cfg.TEST.SEARCH.FACTOR = 5.0
cfg.TEST.SEARCH.USE_GRID = False
cfg.TEST.SEARCH.GRID = edict()
cfg.TEST.SEARCH.GRID.SHAPE = [17, 17]
cfg.TEST.SEARCH.GRID.GENERATOR = edict()
cfg.TEST.SEARCH.GRID.GENERATOR.TYPE = "qp"
cfg.TEST.SEARCH.GRID.GENERATOR.BANDWIDTH_SCALE = 64
cfg.TEST.SEARCH.GRID.GENERATOR.ZOOM_FACTOR = 1.5
cfg.TEST.SEARCH.GRID.GENERATOR.LOSS = edict()
cfg.TEST.SEARCH.GRID.GENERATOR.LOSS.NAMES = "asap_ts_weight"
cfg.TEST.SEARCH.GRID.GENERATOR.LOSS.WEIGHTS = [1, 1]

cfg.TEST.USE_REGION_ADJUST = False
cfg.TEST.MOTION_VERIFY = False
cfg.TEST.MOTION_VERIFY_ALPHA = 0.5
cfg.TEST.MAX_SEARCH_FACTOR = 8
cfg.TEST.ENLARGE_STEP = 1.0
cfg.TEST.T_LOGITS = 0.8
cfg.TEST.T_SCORE = 0.5


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename, base_cfg=None):
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        if base_cfg is not None:
            _update_config(base_cfg, exp_config)
        else:
            _update_config(cfg, exp_config)
