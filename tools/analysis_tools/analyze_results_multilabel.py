# Modified analyze_results.py to support multi-label results
# Drop-in usage:
#   python analyze_results_multilabel.py <config> <results.pkl> --out-dir <dir> [--topk 1000] [--threshold 0.5]
# Behavior:
#   * Works for both single-label and multi-label.
#   * For multi-label, compares GT vs. PRED after converting to multi-hot with a probability threshold.
#   * Uses UniversalVisualizer to draw GT & Pred on images (works for both cases).

import argparse
import os.path as osp
from pathlib import Path
from typing import List, Tuple

import mmcv
import mmengine
import numpy as np
import torch
from mmengine import DictAction

from mmpretrain.datasets import build_dataset
from mmpretrain.structures import DataSample
from mmpretrain.visualization import UniversalVisualizer


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze test results (supports multi-label).')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('result', help='test result json/pkl file')
    parser.add_argument('--out-dir', required=True, help='dir to store output files')
    parser.add_argument('--topk', default=20, type=int, help='Number of images to export per bucket')
    parser.add_argument('--rescale-factor', '-r', type=float,
                        help='image rescale factor if output is too large/small')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='probability threshold for multi-label (if logits are provided, sigmoid will be applied)')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='override settings in config; key=value format')
    args = parser.parse_args()
    return args


# -------- helpers --------

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _is_multilabel_like(gt_label) -> bool:
    """Heuristic: list/array with length > 1 or nested indicates multi-label."""
    if gt_label is None:
        return False
    if isinstance(gt_label, (list, tuple)):
        return len(gt_label) != 1
    arr = _to_numpy(gt_label)
    return arr.ndim == 1 and arr.size != 1


def _indices_from_any(x) -> List[int]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    arr = _to_numpy(x).squeeze()
    if arr.ndim == 0:
        return [int(arr)]
    # multi-hot or index list
    uniq = np.unique(arr)
    if set(uniq.tolist()).issubset({0, 1}) and arr.size > 1:
        return np.where(arr.astype(int) == 1)[0].tolist()
    return [int(v) for v in arr.tolist()]


def _to_multihot(indices: List[int], num_classes: int) -> np.ndarray:
    mh = np.zeros((num_classes,), dtype=np.int32)
    for i in indices:
        if 0 <= i < num_classes:
            mh[i] = 1
    return mh


def _scores_to_multihot(scores: np.ndarray, thr: float, treat_as_logits: bool) -> np.ndarray:
    s = scores.astype(np.float32)
    if treat_as_logits:
        s = 1.0 / (1.0 + np.exp(-s))  # sigmoid
    return (s >= thr).astype(np.int32)


def save_imgs(result_dir, folder_name, results, dataset, rescale_factor=None):
    full_dir = osp.join(result_dir, folder_name)
    mmengine.mkdir_or_exist(full_dir)

    vis = UniversalVisualizer()
    vis.dataset_meta = {'classes': dataset.CLASSES}

    dump_infos = []
    for data_sample in results:
        data_info = dataset.get_data_info(data_sample.sample_idx)
        if 'img' in data_info:
            img = data_info['img']
            name = str(data_sample.sample_idx)
        elif 'img_path' in data_info:
            img = mmcv.imread(data_info['img_path'], channel_order='rgb')
            name = Path(data_info['img_path']).name
        else:
            raise ValueError('Cannot load images from the dataset infos.')
        if rescale_factor is not None:
            img = mmcv.imrescale(img, rescale_factor)

        vis.visualize_cls(img, data_sample, out_file=osp.join(full_dir, name + '.png'))

        dump = {}
        for k, v in data_sample.items():
            dump[k] = v.tolist() if isinstance(v, torch.Tensor) else v
        dump_infos.append(dump)

    mmengine.dump(dump_infos, osp.join(full_dir, folder_name + '.json'))


def main():
    args = parse_args()

    cfg = mmengine.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Build dataset (no transforms needed)
    cfg.test_dataloader.dataset.pipeline = []
    dataset = build_dataset(cfg.test_dataloader.dataset)
    num_classes = len(getattr(dataset, 'CLASSES', [])) or None

    raw_results = list(mmengine.load(args.result))

    # Convert raw dicts to DataSample for visualization, and compute success/fail robustly.
    samples: List[DataSample] = []
    eval_tuples: List[Tuple[np.ndarray, np.ndarray, float, int]] = []  # (gt_mh, pred_mh, max_score, sample_idx)

    for result in raw_results:
        # Build sample for visualize
        ds = DataSample()
        ds.set_metainfo({'sample_idx': result['sample_idx']})
        if 'gt_label' in result:
            ds.set_gt_label(result['gt_label'])
        if 'pred_label' in result:
            ds.set_pred_label(result['pred_label'])
        if 'pred_score' in result:
            ds.set_pred_score(result['pred_score'])
        samples.append(ds)

        # Prepare arrays for success/fail decision
        gt_idx = _indices_from_any(result.get('gt_label', None))
        pred_idx = _indices_from_any(result.get('pred_label', None))

        # Determine number of classes if not known
        if num_classes is None:
            # fall back to pred_score length if available, else max index + 1
            ps = result.get('pred_score', None)
            if ps is not None:
                num_classes = int(_to_numpy(ps).reshape(-1).shape[0])
            else:
                max_idx = max(gt_idx + pred_idx) if (gt_idx or pred_idx) else 0
                num_classes = max_idx + 1

        gt_mh = _to_multihot(gt_idx, num_classes)

        # Build pred multi-hot: prefer pred_score + threshold to be robust
        pred_score = result.get('pred_score', None)
        if pred_score is not None:
            arr = _to_numpy(pred_score).reshape(-1)
            # Heuristic to decide if logits or probs
            treat_as_logits = not (arr.min() >= 0.0 and arr.max() <= 1.0)
            pred_mh = _scores_to_multihot(arr, args.threshold, treat_as_logits)
            max_score = (1.0 / (1.0 + np.exp(-arr))).max() if treat_as_logits else arr.max()
        else:
            pred_mh = _to_multihot(pred_idx, num_classes)
            max_score = float(pred_mh.max())

        eval_tuples.append((gt_mh, pred_mh, max_score, int(result['sample_idx'])))

    # Sort by confidence (descending)
    order = np.argsort([-t[2] for t in eval_tuples]).tolist()
    samples = [samples[i] for i in order]
    eval_tuples = [eval_tuples[i] for i in order]

    success, fail = [], []
    for (gt_mh, pred_mh, _, sample_idx), ds in zip(eval_tuples, samples):
        if np.array_equal(gt_mh, pred_mh):
            success.append(ds)
        else:
            fail.append(ds)

    success = success[:args.topk]
    fail = fail[:args.topk]

    save_imgs(args.out_dir, 'success', success, dataset, args.rescale_factor)
    save_imgs(args.out_dir, 'fail', fail, dataset, args.rescale_factor)


if __name__ == '__main__':
    main()
