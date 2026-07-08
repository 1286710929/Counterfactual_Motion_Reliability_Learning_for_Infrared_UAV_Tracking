from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.ostrack.config import cfg, update_config_from_file


def parameters(yaml_name: str, test_epoch=None):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir

    yaml_file = os.path.join(prj_dir, "experiments/ostrack/%s.yaml" % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg

    params.template_factor = cfg.TEST.TEMPLATE.FACTOR
    params.template_size = cfg.TEST.TEMPLATE.SIZE
    params.search_factor = cfg.TEST.SEARCH.FACTOR
    params.search_size = cfg.TEST.SEARCH.SIZE

    epoch = cfg.TEST.EPOCH if test_epoch is None else test_epoch
    params.checkpoint = os.path.join(
        save_dir,
        "checkpoints/train/ostrack/%s/OSTrack_ep%04d.pth.tar" % (yaml_name, epoch),
    )

    params.save_all_boxes = False
    return params
