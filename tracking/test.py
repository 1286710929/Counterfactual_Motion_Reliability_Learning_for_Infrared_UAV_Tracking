import os
import sys
import argparse
from datetime import datetime

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.environment import env_settings
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker, trackerlist
from lib.test.analysis.plot_results import (
    check_and_load_precomputed_results,
    generate_formatted_report,
    get_auc_curve,
    get_prec_curve,
    get_tracker_display_name,
    merge_multiple_runs,
)
import torch

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def _format_tracker_metrics(trackers, dataset, dataset_name, tracker_name, tracker_param,
                            merge_results=True, skip_missing_seq=False):
    eval_data = check_and_load_precomputed_results(
        trackers,
        dataset,
        dataset_name,
        tracker_param,
        tracker_name,
        force_evaluation=True,
        skip_missing_seq=skip_missing_seq,
    )

    if merge_results:
        eval_data = merge_multiple_runs(eval_data)

    tracker_names = eval_data['trackers']
    valid_sequence = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)
    scores = {}

    threshold_set_overlap = torch.tensor(eval_data['threshold_set_overlap'])
    ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])
    auc_curve, auc = get_auc_curve(ave_success_rate_plot_overlap, valid_sequence)
    scores['AUC'] = auc
    scores['OP50'] = auc_curve[:, threshold_set_overlap == 0.50].view(-1)
    scores['OP75'] = auc_curve[:, threshold_set_overlap == 0.75].view(-1)

    ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])
    _, precision = get_prec_curve(ave_success_rate_plot_center, valid_sequence)
    scores['Precision'] = precision

    ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])
    _, norm_precision = get_prec_curve(ave_success_rate_plot_center_norm, valid_sequence)
    scores['Norm Precision'] = norm_precision

    if 'state_accuracy' in eval_data:
        state_accuracy = torch.tensor(eval_data['state_accuracy'], dtype=torch.float64)
        valid_state_accuracy = state_accuracy[valid_sequence, :]
        if (valid_state_accuracy >= 0.0).any():
            scores['mSA'] = valid_state_accuracy.mean(0) * 100.0

    tracker_disp_names = [get_tracker_display_name(trk) for trk in tracker_names]
    report_text = generate_formatted_report(tracker_disp_names, scores, table_name=dataset_name)
    header = 'Reporting results over {} / {} sequences'.format(
        valid_sequence.long().sum().item(), valid_sequence.shape[0])

    return '{}\n{}'.format(header, report_text)


def _default_metrics_log_path(tracker_name, tracker_param, dataset_name):
    settings = env_settings()
    return os.path.join(settings.result_plot_path, tracker_name, tracker_param, dataset_name, 'metrics.log')


def _write_metrics_log(log_path, report_text, args):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a') as f:
        f.write('\n' + '=' * 80 + '\n')
        f.write('Time: {}\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        if args is not None:
            f.write('Tracker: {}  Param: {}  Dataset: {}  Run ID: {}\n'.format(
                args.tracker_name, args.tracker_param, args.dataset_name, args.run_id))
        f.write(report_text)
        if not report_text.endswith('\n'):
            f.write('\n')


def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0,
                num_gpus=8, test_epoch=None, analysis=True, analysis_log=None, merge_results=True,
                skip_missing_seq=False, args=None):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot, tnl2k).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """

    dataset = get_dataset(dataset_name)

    if sequence is not None:
        dataset = [dataset[sequence]]
    
    trackers = trackerlist(tracker_name, tracker_param, dataset_name, run_id)
    run_dataset(dataset, trackers, debug, threads, num_gpus=num_gpus)

    if analysis and not debug:
        report_text = _format_tracker_metrics(
            trackers,
            dataset,
            dataset_name,
            tracker_name,
            tracker_param,
            merge_results=merge_results,
            skip_missing_seq=skip_missing_seq,
        )
        print(report_text)

        log_path = analysis_log or _default_metrics_log_path(tracker_name, tracker_param, dataset_name)
        _write_metrics_log(log_path, report_text, args)
        print('Metrics log saved to {}'.format(log_path))


def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    
    def parse_run_id(run_id):
        try:
            return [int(x) for x in run_id.split(',')]
        except ValueError:
            return [int(run_id)]
    
    parser.add_argument('--tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('--tracker_param', type=str, help='Name of config file.')
    parser.add_argument('--run_id', type=parse_run_id, default=None, help='The run id.')
    parser.add_argument('--dataset_name', type=str, default='otb', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--test_epoch', type=int, default=None)
    parser.add_argument('--analysis', type=int, choices=[0, 1], default=1,
                        help='Compute and print metrics after running the tracker.')
    parser.add_argument('--analysis_log', type=str, default=None,
                        help='Path to append the metric report. Defaults to output/test/result_plots/<tracker>/<param>/<dataset>/metrics.log')
    parser.add_argument('--merge_results', type=int, choices=[0, 1], default=1,
                        help='Merge multiple run ids when reporting metrics.')
    parser.add_argument('--skip_missing_seq', type=int, choices=[0, 1], default=0,
                        help='Skip missing result files when computing metrics.')
    

    args = parser.parse_args()

    seq_name = args.sequence
    
    # try:
    #     seq_name = int(args.sequence)
    # except:
    #     seq_name = args.sequence

    if args.run_id is None and args.test_epoch is None:
        print("Please specify either run_id or test_epoch")
    if args.run_id is None and args.test_epoch is not None:
         args.run_id = args.test_epoch
    if args.test_epoch is None and args.run_id is not None:
         args.test_epoch = args.run_id
    
    print("dataset_name is %s" %args.dataset_name)
    print("tracker_param is %s" %args.tracker_param)
    
    run_tracker(args.tracker_name, args.tracker_param, args.run_id, args.dataset_name, seq_name, args.debug,
                args.threads, num_gpus=args.num_gpus, test_epoch=args.test_epoch,
                analysis=bool(args.analysis), analysis_log=args.analysis_log,
                merge_results=bool(args.merge_results), skip_missing_seq=bool(args.skip_missing_seq),
                args=args)
if __name__ == '__main__':
    main()
