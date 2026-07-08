# CMRTrack Best Anti-UAV410 Result Package

This folder records the best CMRTrack result used in the paper. The result is not manually written: it is copied from the original test log, raw prediction files, evaluation cache, and checkpoint.

## Main Result

Tracker: `ostrack`

Config: `ostrack_motion_v166_single`

Checkpoint: `OSTrack_ep0030.pth.tar`

Dataset: `antiuav410_test`

Original metric line:

```text
ostrack_ostrack_motion_v166_single_030 | 67.26 | 86.03 | 58.98 | 89.91 | 86.16 | 68.48
```

Metric order:

```text
AUC | OP50 | OP75 | Precision | Norm Precision | mSA
```

Paper rounded result:

```text
AUC 67.3, Precision 89.9, Norm Precision 86.2, mSA 68.5
```

## Folder Contents

```text
github/
  checkpoints/
    README.md
  configs/
    ostrack_motion_v166_single.yaml
  experiments/
    ostrack/
  lib/
  tracking/
  eval/
    antiuav410_test_eval_data.pkl
    antiuav410_test_metrics.log
    ostrack-ostrack_motion_v166_single.log
  raw_predictions/
    antiuav410/
  test_code/
    run_cmrtrack_antiuav410_test.sh
  MANIFEST.sha256
```

The code in `lib/`, `tracking/`, and `experiments/` has been trimmed for the main CMRTrack experiment. It keeps only the test-time OSTrack/CMRTrack model, Anti-UAV410 evaluation code, utility functions required by inference, and the `ostrack_motion_v166_single` configuration. Training code, unrelated tracker variants, unrelated experiment YAML files, notebooks, demos, and visualization scripts are not included.

The checkpoint file `OSTrack_ep0030.pth.tar` is not committed to this GitHub repository because it is about 1.1 GB. Please download it from the GitHub Release page and place it under `checkpoints/`. Details are provided below and in `checkpoints/README.md`.

## Checkpoint

Download the best checkpoint from the release page:

```text
https://github.com/1286710929/Counterfactual_Motion_Reliability_Learning_for_Infrared_UAV_Tracking/releases/download/v1.0.0/OSTrack_ep0030.pth.tar
```

Then place it at:

```text
checkpoints/OSTrack_ep0030.pth.tar
```

SHA256:

```text
f3236be0aa911550a100c7f7940885148b4bc498d417f661e7ebb3b70d49d737
```

You can verify the downloaded checkpoint with:

```bash
sha256sum checkpoints/OSTrack_ep0030.pth.tar
```

## Reproduce Evaluation

Run from this package:

```bash
cd /home/cyh/Tracking/FocusTrack/github
bash test_code/run_cmrtrack_antiuav410_test.sh
```

Or run from the original project root:

```bash
bash github/test_code/run_cmrtrack_antiuav410_test.sh
```

The package includes the testing code under `tracking/` and `lib/`. The script links the packaged checkpoint to the default checkpoint path expected by `tracking/test.py`, then evaluates epoch 30 on `antiuav410_test`. The reproduced metric log is appended to:

```text
eval/reproduced_antiuav410_test_metrics.log
```

The default Anti-UAV410 dataset path is:

```text
/home/cyh/Anti_UAV/datasets/Anti-UAV410
```

To use another dataset location, set:

```bash
export ANTIUAV410_PATH=/path/to/Anti-UAV410
```

The script automatically uses `/home/cyh/miniconda3/envs/focustrack5090/bin/python` when it exists. To use another Python environment, set:

```bash
export PYTHON_BIN=/path/to/python
```

The original raw prediction files are stored in:

```text
raw_predictions/antiuav410/
```
