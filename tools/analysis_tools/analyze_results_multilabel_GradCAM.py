# Modified analyze_results.py to support multi-label results + optional Grad-CAM
# Output: PNGs arranged as:
#   First row = GT/PRED overlay (converted to BGR for cv2.imwrite)
#   Second row = all CAM overlays (horizontally concatenated)

import argparse
import os.path as osp
from pathlib import Path
from typing import List, Tuple

import cv2
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
        description='Analyze test results (supports multi-label + Grad-CAM).')
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
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to classification checkpoint for Grad-CAM. If omitted, CAM is skipped.')
    parser.add_argument('--target-layer', type=str, default='backbone.layer4.2',
                        help='Module path for CAM, e.g., backbone.layer4.2 or backbone.layer3.5')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Square input size for CAM preprocessing')
    parser.add_argument('--cam-alpha', type=float, default=0.45,
                        help='Blend factor for heatmap overlay (0~1)')
    parser.add_argument('--cam-mode', type=str, default='top1', choices=['top1','all_pos'],
                        help='top1: CAM for highest-prob class; all_pos: CAM for all predicted-positive classes (hstack)')
    args = parser.parse_args()
    return args


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _indices_from_any(x) -> List[int]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    arr = _to_numpy(x).squeeze()
    if arr.ndim == 0:
        return [int(arr)]
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
        s = 1.0 / (1.0 + np.exp(-s))
    return (s >= thr).astype(np.int32)


def _normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = np.maximum(cam, 0)
    cam -= cam.min() if cam.size > 0 else 0
    cam_max = cam.max() if cam.size > 0 else 1.0
    if cam_max > 0:
        cam /= cam_max
    return cam


def _preprocess_for_cam(img_rgb: np.ndarray, img_size: int, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    img = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))[None, ...]
    return torch.from_numpy(img)


def _get_module_by_path(module: torch.nn.Module, path: str) -> torch.nn.Module:
    cur = module
    for p in path.split('.'):
        cur = getattr(cur, p)
    return cur


def _grad_cam(model, target_layer, inp, class_idx, device):
    fmap = []
    grads = []
    def fwd_hook(_, __, output): fmap.append(output)
    def bwd_hook(_, grad_input, grad_output): grads.append(grad_output[0])
    handle_f = target_layer.register_forward_hook(fwd_hook)
    handle_b = target_layer.register_full_backward_hook(bwd_hook)
    model.zero_grad(set_to_none=True)
    inp = inp.to(device)
    logits = model(inp, mode='tensor')
    score = logits[0, class_idx] if logits.ndim == 2 else logits.squeeze()[class_idx]
    score.backward(retain_graph=True)
    A = fmap[0].detach().cpu()
    dA = grads[0].detach().cpu()
    weights = dA.mean(dim=(2, 3), keepdim=True)
    cam = (weights * A).sum(dim=1, keepdim=False)[0]
    cam = _normalize_cam(cam.numpy())
    handle_f.remove(); handle_b.remove()
    return cam


def _build_model_for_cam(cfg, checkpoint, device):
    from mmpretrain.apis import init_model
    model = init_model(cfg, checkpoint, device=device)
    model.eval()
    return model


def save_imgs(result_dir, folder_name, results, dataset, rescale_factor=None,
              cam_ctx=None, cam_mode='top1', cam_thr=0.5):
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
            continue
        if rescale_factor is not None:
            img = mmcv.imrescale(img, rescale_factor)

        # First row: GT/PRED overlay
        left_path = osp.join(full_dir, name + '.png')
        vis.visualize_cls(img, data_sample, out_file=left_path)
        base_img = mmcv.imread(left_path, channel_order='rgb')

        cam_row = None
        if cam_ctx is not None:
            model, target_layer, img_size, mean, std, cam_alpha, device, class_names = cam_ctx
            scores_np = None
            pred_score = getattr(data_sample, 'pred_score', None)
            if isinstance(pred_score, torch.Tensor):
                scores_np = pred_score.detach().cpu().numpy().reshape(-1)
                if not (scores_np.min() >= 0.0 and scores_np.max() <= 1.0):
                    scores_np = 1.0 / (1.0 + np.exp(-scores_np))
            classes_for_cam = []
            if cam_mode == 'all_pos' and scores_np is not None:
                classes_for_cam = np.where(scores_np >= cam_thr)[0].tolist() or [int(np.argmax(scores_np))]
            elif cam_mode == 'all_pos':
                classes_for_cam = _indices_from_any(getattr(data_sample, 'pred_label', None)) or [0]
            else:
                classes_for_cam = [int(np.argmax(scores_np))] if scores_np is not None else [0]
            inp = _preprocess_for_cam(img, img_size, mean, std)
            overlays = []
            for cid in classes_for_cam:
                cam = _grad_cam(model, target_layer, inp, cid, device)
                cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
                heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
                overlay = (cam_alpha * heat + (1.0 - cam_alpha) * cv2.cvtColor(img, cv2.COLOR_RGB2BGR)).astype(np.uint8)
                label = class_names[cid] if class_names and cid < len(class_names) else f'class {cid}'
                cv2.putText(overlay, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2, cv2.LINE_AA)
                overlays.append(overlay)
            if overlays:
                cam_row = np.hstack(overlays)

        if cam_row is not None:
            W = max(base_img.shape[1], cam_row.shape[1])
            def _pad_w(im, W):
                if im.shape[1] == W: return im
                return np.pad(im, ((0,0),(0,W-im.shape[1]),(0,0)), mode='constant')
            base_img = _pad_w(cv2.cvtColor(base_img, cv2.COLOR_RGB2BGR), W)
            cam_row = _pad_w(cam_row, W)
            combo = np.vstack([base_img, cam_row])
        else:
            combo = cv2.cvtColor(base_img, cv2.COLOR_RGB2BGR)

        out_path = osp.join(full_dir, name + '.png')
        cv2.imwrite(out_path, combo)

        dump = {}
        for k, v in data_sample.items():
            dump[k] = v.tolist() if isinstance(v, torch.Tensor) else v
        dump_infos.append(dump)

    mmengine.dump(dump_infos, osp.join(full_dir, folder_name + '.json'))


def main():
    args = parse_args()
    cfg = mmengine.Config.fromfile(args.config)
    if args.cfg_options: cfg.merge_from_dict(args.cfg_options)
    mean = np.array(cfg.get('data_preprocessor', {}).get('mean', [123.675, 116.28, 103.53]), dtype=np.float32)
    std = np.array(cfg.get('data_preprocessor', {}).get('std', [58.395, 57.12, 57.375]), dtype=np.float32)
    test_ds_cfg = cfg.test_dataloader.dataset.copy(); test_ds_cfg['pipeline'] = []
    cfg.test_dataloader.dataset = test_ds_cfg
    dataset = build_dataset(cfg.test_dataloader.dataset)
    class_names = getattr(dataset, 'CLASSES', None)
    raw_results = list(mmengine.load(args.result))
    cam_ctx = None
    if args.checkpoint:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = _build_model_for_cam(cfg, args.checkpoint, device)
        target_layer = _get_module_by_path(model, args.target_layer)
        cam_ctx = (model, target_layer, args.img_size, mean, std, args.cam_alpha, device, class_names)
    samples = []
    eval_tuples = []
    num_classes = None
    for result in raw_results:
        ds = DataSample(); ds.set_metainfo({'sample_idx': result['sample_idx']})
        if 'gt_label' in result: ds.set_gt_label(result['gt_label'])
        if 'pred_label' in result: ds.set_pred_label(result['pred_label'])
        if 'pred_score' in result: ds.set_pred_score(result['pred_score'])
        samples.append(ds)
        gt_idx = _indices_from_any(result.get('gt_label', None))
        pred_idx = _indices_from_any(result.get('pred_label', None))
        if num_classes is None:
            ps = result.get('pred_score', None)
            num_classes = int(_to_numpy(ps).reshape(-1).shape[0]) if ps is not None else max(gt_idx+pred_idx or [0])+1
        gt_mh = _to_multihot(gt_idx, num_classes)
        pred_score = result.get('pred_score', None)
        if pred_score is not None:
            arr = _to_numpy(pred_score).reshape(-1)
            treat_as_logits = not (arr.min() >= 0.0 and arr.max() <= 1.0)
            pred_mh = _scores_to_multihot(arr, args.threshold, treat_as_logits)
            max_score = (1.0 / (1.0 + np.exp(-arr))).max() if treat_as_logits else arr.max()
        else:
            pred_mh = _to_multihot(pred_idx, num_classes)
            max_score = float(pred_mh.max())
        eval_tuples.append((gt_mh, pred_mh, max_score, int(result['sample_idx'])))
    order = np.argsort([-t[2] for t in eval_tuples]).tolist()
    samples = [samples[i] for i in order]
    eval_tuples = [eval_tuples[i] for i in order]
    success, fail = [], []
    for (gt_mh, pred_mh, _, _), ds in zip(eval_tuples, samples):
        (success if np.array_equal(gt_mh, pred_mh) else fail).append(ds)
    success = success[:args.topk]; fail = fail[:args.topk]
    save_imgs(args.out_dir, 'success', success, dataset, args.rescale_factor, cam_ctx,
              cam_mode=args.cam_mode, cam_thr=args.threshold)
    save_imgs(args.out_dir, 'fail', fail, dataset, args.rescale_factor, cam_ctx,
              cam_mode=args.cam_mode, cam_thr=args.threshold)

if __name__ == '__main__':
    main()
