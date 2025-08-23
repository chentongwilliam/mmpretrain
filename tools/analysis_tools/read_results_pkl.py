from mmengine import load

pkl_path = "/home/chentong/IUNA_Trainer_work_dir/Training/training_20250823_224219/Test_20250823_224524/results.pkl"
results = load(pkl_path)

print(type(results), len(results))     # 看整体是 list 还是别的
print(type(results[0]))                # 看单条是什么（dict / DataSample / 其他）
print(results[0])                      # 直接窥探第一条结构

import numpy as np

def to_numpy(x):
    # torch.Tensor
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    # numpy
    if isinstance(x, np.ndarray):
        return x
    # list/tuple/int/float/None 都原样返回
    return x

def unpack_record(rec):
    # 兼容 dict 或具备属性的对象（例如 DataSample）
    get = (lambda k: rec.get(k, None)) if isinstance(rec, dict) else (lambda k: getattr(rec, k, None))
    out = {}
    for k in ("sample_idx", "pred_score", "pred_label", "gt_label"):
        v = get(k)
        out[k] = to_numpy(v)
    return out

rec0 = unpack_record(results[0])
for k, v in rec0.items():
    print(k, type(v), (v.shape if hasattr(v, "shape") else v))


bad_shape_idx = []
empty_pred_idx = []
for i, r in enumerate(results):
    rr = unpack_record(r)
    ps, pl, gl = rr["pred_score"], rr["pred_label"], rr["gt_label"]

    # 记录 pred_label / gt_label 长度不一致的样本
    try:
        len_pl = len(pl) if pl is not None else -1
        len_gl = len(gl) if gl is not None else -1
        if len_pl != len_gl:
            bad_shape_idx.append((i, len_pl, len_gl))
    except TypeError:
        # 某些实现 pred_label/gt_label 可能是标量或 None
        bad_shape_idx.append((i, type(pl), type(gl)))

    # 记录“预测为空”的样本（例如没有任何正例被选中）
    if pl is None or (hasattr(pl, "__len__") and len(pl) == 0):
        empty_pred_idx.append(i)

print("bad_shape_idx (idx, len_pred_label, len_gt_label):", bad_shape_idx[:20])
print("empty_pred_idx (前20):", empty_pred_idx[:20], "总数:", len(empty_pred_idx))
