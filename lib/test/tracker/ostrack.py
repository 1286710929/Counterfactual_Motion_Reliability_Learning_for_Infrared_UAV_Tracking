import os

import cv2
import torch

from lib.models.ostrack import build_ostrack
from lib.test.tracker.basetracker import BaseTracker
from lib.test.tracker.data_utils import Preprocessor
from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box


def _load_tracker_checkpoint(checkpoint_path):
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


class OSTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(OSTrack, self).__init__(params)
        network = build_ostrack(params.cfg, training=False)
        network.load_state_dict(_load_tracker_checkpoint(self.params.checkpoint)["net"], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        self.feat_sz = self.cfg.TEST.SEARCH.SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                os.makedirs(self.save_dir, exist_ok=True)
            else:
                self._init_visdom(None, 1)

        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.search_history = []

    def initialize(self, image, info: dict):
        z_patch_arr, resize_factor, z_amask_arr = sample_target(
            image,
            info["init_bbox"],
            self.params.template_factor,
            output_sz=self.params.template_size,
        )
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.state = info["init_bbox"]
        self.frame_id = 0
        self.search_history = []
        if self.save_all_boxes:
            all_boxes_save = info["init_bbox"] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        h, w, _ = image.shape
        self.frame_id += 1
        crop_center_box = self.state
        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image,
            crop_center_box,
            self.params.search_factor,
            output_sz=self.params.search_size,
        )

        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        search_history = self._get_search_history_tensor(search.tensors.device)
        with torch.no_grad():
            out_dict = self.network.forward(
                template=self.z_dict1.tensors,
                search=search.tensors,
                search_history=search_history,
                training=False,
            )

        response_map = out_dict.get("motion_score_map", out_dict["score_map"])
        response = self.output_window * response_map
        if "motion_score_map" not in out_dict and self.cfg.TEST.MOTION_VERIFY and "motion_map" in out_dict:
            response = response * (1.0 + self.cfg.TEST.MOTION_VERIFY_ALPHA * out_dict["motion_map"])

        pred_boxes = self.network.box_head.cal_bbox(response, out_dict["size_map"], out_dict["offset_map"])
        pred_boxes = pred_boxes.view(-1, 4)
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), h, w, margin=10)
        self._push_search_history(search.tensors.detach())
        output_state = self.state
        if self._predict_absent(out_dict):
            output_state = [0, 0, 0, 0]

        if self.debug:
            if not self.use_visdom:
                x1, y1, bw, bh = self.state
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_bgr, (int(x1), int(y1)), (int(x1 + bw), int(y1 + bh)), (255, 0, 0), 2)
                cv2.imwrite(os.path.join(self.save_dir, "%04d.jpg" % self.frame_id), image_bgr)
            else:
                self.visdom.register((image, info["gt_bbox"].tolist(), self.state), "Tracking", 1, "Tracking")
                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), "image", 1, "search_region")
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), "image", 1, "template")
                self.visdom.register(out_dict["score_map"].view(self.feat_sz, self.feat_sz), "heatmap", 1, "score_map")
                self.visdom.register(response.view(self.feat_sz, self.feat_sz), "heatmap", 1, "score_map_hann")

                if "removed_indexes_s" in out_dict and out_dict["removed_indexes_s"]:
                    removed_indexes_s = [x.cpu().numpy() for x in out_dict["removed_indexes_s"]]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), "image", 1, "masked_search")

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        output = {"target_bbox": output_state}
        if getattr(self.params, "return_motion_visualization", False):
            output["search_patch"] = x_patch_arr
            output["score_map"] = out_dict["score_map"].detach().float().squeeze().cpu().numpy()
            output["motion_score_map"] = response_map.detach().float().squeeze().cpu().numpy()
            if "motion_map" in out_dict:
                output["motion_map"] = out_dict["motion_map"].detach().float().squeeze().cpu().numpy()
            if "motion_score_map" in out_dict:
                output["fused_score_map"] = out_dict["motion_score_map"].detach().float().squeeze().cpu().numpy()
            if "motion_reliability" in out_dict:
                output["motion_reliability"] = out_dict["motion_reliability"].detach().float().view(-1).mean().item()
            output["crop_center_box"] = crop_center_box
            output["resize_factor"] = resize_factor

        if self.save_all_boxes:
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            output["all_boxes"] = all_boxes.view(-1).tolist()
        return output

    def _predict_absent(self, out_dict):
        if "exist_prob" not in out_dict:
            return False
        threshold = float(getattr(self.cfg.MODEL.EXIST, "THRESHOLD", 0.5))
        return out_dict["exist_prob"].detach().view(-1).mean().item() < threshold

    def _get_search_history_tensor(self, device):
        if len(self.search_history) == 0:
            return None
        history_len = self.cfg.MODEL.MOTION.HISTORY_LEN
        history = torch.stack(self.search_history[-history_len:], dim=1).to(device)
        return history

    def _push_search_history(self, search_tensor):
        if not getattr(self.cfg.MODEL.MOTION, "ENABLE", False):
            return
        self.search_history.append(search_tensor.cpu())
        history_len = self.cfg.MODEL.MOTION.HISTORY_LEN
        if len(self.search_history) > history_len:
            self.search_history = self.search_history[-history_len:]

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return OSTrack
