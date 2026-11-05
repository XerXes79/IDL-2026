"""Check for test-set leakage between the 75x75 and 150x150 clock datasets.

Splits both datasets exactly like the notebook:
  random_split(..., [0.9, 0.1], seed 528)                -> train+val / test
  inside train_model: random_split(..., val_ratio 0.1111, seed 528) -> train / val
and then checks
  1. the labels of both files are in the same order
  2. image i of the 150x150 file (shrunk to 75x75) is image i of the 75x75 file
  3. no 150x150 test image is (closest to) a 75x75 image the model trained on
  4. no test image is a (near) copy of a training image within the same dataset

Usage: python check_leak.py [path/to/A1_data_75.zip] [path/to/A1_data_150.zip]
"""
import io, sys, zipfile
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import random_split

SEED, VAL_RATIO = 528, 0.1111
DUPLICATE = 0.98     # correlation above this between two images of the same file = a copy


def load(path):
    out = {}
    with zipfile.ZipFile(path) as z:
        for p in ["images.npy", "labels.npy"]:
            with io.BufferedReader(z.open(p)) as f:
                out[p] = np.load(f)
    return torch.from_numpy(out["images.npy"]), torch.from_numpy(out["labels.npy"])     # images stay uint8 (less memory)


def split_indices(n):
    """Same indices the notebook gets (random_split only depends on the length and the seed)."""
    rest, test = random_split(range(n), [0.9, 0.1], generator=torch.Generator().manual_seed(SEED))
    n_val = int(len(rest) * VAL_RATIO)
    tr, va = random_split(rest, [len(rest) - n_val, n_val], generator=torch.Generator().manual_seed(SEED))
    rest = torch.tensor(list(rest))
    return rest[list(tr.indices)], rest[list(va.indices)], torch.tensor(list(test))


def normalize(images):
    """Flatten, remove the mean and scale to length 1, so a dot product is the correlation."""
    x = images.flatten(1).float()
    x = x - x.mean(1, keepdim=True)
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)


def best_match(queries, pool, chunk=512):
    """For every query: highest correlation with any image in pool, and which one."""
    sims, idx = [], []
    for q in queries.split(chunk):
        s = q @ pool.T
        v, i = s.max(1)
        sims.append(v); idx.append(i)
    return torch.cat(sims), torch.cat(idx)


path75 = sys.argv[1] if len(sys.argv) > 1 else "data/A1_data_75.zip"
path150 = sys.argv[2] if len(sys.argv) > 2 else "data/A1_data_150.zip"
img75, lab75 = load(path75)
img150, lab150 = load(path150)
n = len(img75)
assert len(img150) == n, f"different sizes: {len(img75)} vs {len(img150)}"
train, val, test = split_indices(n)
print(f"{n} images | train {len(train)} | val {len(val)} | test {len(test)}")
leak = False

# 1. labels in the same order
same_labels = torch.equal(lab75, lab150)
print(f"\n1. labels identical in both files: {same_labels}")
if not same_labels:
    print(f"   {(lab75 != lab150).any(1).sum().item()} of {n} rows differ")

# 2 + 3. for every 150x150 test image (shrunk to 75x75), find the most similar image in the whole 75x75 file.
#        It should be the same index. If it is a train/val index instead, the model has seen that clock.
v75 = normalize(img75)
v150 = normalize(torch.cat([F.avg_pool2d(c.float().unsqueeze(1), 2).squeeze(1) for c in img150.split(1000)]))
own = (v75 * v150).sum(1)
sim, which = best_match(v150[test], v75)
is_self = which == test
in_seen = torch.isin(which, torch.cat([train, val])) & ~is_self
print(f"2. correlation of image i (150, shrunk) with image i (75): min {own.min():.4f} | mean {own.mean():.4f}")
print(f"   test images whose best match in the 75x75 file is the same index: {is_self.sum().item()} of {len(test)}")
print(f"3. test images whose best match is a 75x75 TRAIN/VAL image: {in_seen.sum().item()} of {len(test)}")
if in_seen.any():
    leak = True
    for t, s, c in list(zip(test[in_seen].tolist(), which[in_seen].tolist(), sim[in_seen].tolist()))[:5]:
        print(f"   e.g. 150x150 test image {t} ~ 75x75 training image {s} (corr {c:.3f}), "
              f"labels {lab150[t].tolist()} / {lab75[s].tolist()}")
same_order = bool(is_self.all())
seen = torch.cat([train, val])

# 4. duplicates inside each dataset (test vs train/val)
for name, v in [("75x75", v75), ("150x150", v150)]:
    sim, which = best_match(v[test], v[seen])
    dup = sim > DUPLICATE
    print(f"4. {name}: test images that are (near) copies of a train/val image: {dup.sum().item()} "
          f"(highest correlation {sim.max():.4f})")
    if dup.any():
        leak = True

print("\nRESULT:", "possible leak, see above" if leak or not same_order else
      "no leak: both files have the same images in the same order, so the 75x75 weights never saw the 150x150 test set")




import glob


def clock_test_set(size=75):
    """The test split used by every Tell-the-Time run (seed 528), with the labels as (hour, minute)."""
    data = TellTheTime if size == 75 else TellTheTime_150
    images = torch.from_numpy(data['images.npy']).float().unsqueeze(1) / 255
    labels = torch.from_numpy(data['labels.npy']).long()
    y = torch.stack([labels[:, 0] % 12, labels[:, 1]], dim=1)
    _, test = random_split(TensorDataset(images, y), [0.9, 0.1], generator=torch.Generator().manual_seed(528))
    return test


def build_from_weights(path):
    """Rebuild a Tell-the-Time model from its weights file alone.
    Reads the number of outputs, whether it is a multi-head model and whether it has the stride-2 stem
    from the saved weights (multi-head = 'mh' in the file name or saved with hour_head / minute_head)."""
    sd = torch.load(path, map_location=device)
    name = os.path.basename(path)
    stem = sd["features.conv_stack.0.weight"].shape == (1, 1, 2, 2)
    size = 150 if (stem or "_150" in name) else 75
    if "hour_head.weight" in sd:
        n_hour, n_minute = sd["hour_head.weight"].shape[0], sd["minute_head.weight"].shape[0]
        multihead = True
    else:
        last = max(int(k.split(".")[2]) for k in sd if k.startswith("classifier.linear_stack.") and k.endswith("weight"))
        n_out = sd[f"classifier.linear_stack.{last}.weight"].shape[0]
        multihead = "_mh_" in name
        n_hour = (1 if n_out == 2 else 12) if multihead else None
        n_minute = n_out - n_hour if multihead else None
    n_out = n_hour + n_minute if multihead else n_out
    model = clock_cnn(n_out, image_size=150 if stem else size)
    if multihead:
        model = MultiHeadCNN(model, n_hour=n_hour)
    if stem:
        model = add_stem(model)
    model.load_state_dict(sd)
    kind = "multihead" if multihead else ("regression" if n_out == 1 else "classification")
    return model, kind, n_out, size


def predict_clock(model, kind, n_out, test_set, batch_size=256):
    """Run the model on the test set. Returns one row per image (true / predicted time, error, ...)."""
    model.to(device).eval()
    rows, idx = [], test_set.indices
    with torch.no_grad():
        start = 0
        for x, y in DataLoader(test_set, batch_size=batch_size, shuffle=False):
            x, y = x.to(device), y.to(device)
            out = model(x)
            true_min = y[:, 0] * 60 + y[:, 1]                                 # 0..719
            conf = torch.full((len(y),), float("nan"), device=device)
            if kind == "classification":
                bin_min = 720 // n_out
                cls = out.argmax(1)
                conf = torch.softmax(out, 1).max(1).values
                pred_min = cls * bin_min + (bin_min - 1) / 2                     # middle of the bin
                correct = cls == true_min // bin_min                           # right class
                raw = [str(c) for c in cls.tolist()]
            else:
                if kind == "regression":
                    pred_min = torch.remainder(out.squeeze(1) * 60, 720)
                    raw = [f"{v:.4f}" for v in out.squeeze(1).tolist()]
                else:                                                         # same decoding as training
                    h_out, m_out = out
                    h_fn = nn.L1Loss() if h_out.shape[1] == 1 else nn.CrossEntropyLoss()
                    m_fn = nn.L1Loss() if m_out.shape[1] == 1 else nn.CrossEntropyLoss()
                    _, pred_h, _ = multihead_loss(out, y, h_fn, m_fn)
                    pred_min = torch.remainder(pred_h * 60, 720)
                    fmt = lambda t: [f"{v:.4f}" for v in t.squeeze(1).tolist()] if t.shape[1] == 1 else [str(v) for v in t.argmax(1).tolist()]
                    raw = [f"{a}|{b}" for a, b in zip(fmt(h_out), fmt(m_out))]                  # hour head | minute head
                    if h_out.shape[1] > 1:
                        conf = torch.softmax(h_out, 1).max(1).values             # confidence of the hour head
                correct = torch.remainder(torch.round(pred_min), 720) == true_min  # exact minute
            err = common_sense_error_min(pred_min / 60, true_min.float() / 60)
            shown = torch.remainder(torch.round(pred_min), 720).long()             # predicted time, rounded to a minute
            pred_hour = torch.div(pred_min, 60, rounding_mode="floor").long()       # hour hand: 2:59.6 -> 2
            for j in range(len(y)):
                rows.append({
                    "index": idx[start + j],
                    "true_time": f"{y[j, 0].item()}:{y[j, 1].item():02d}",
                    "pred_time": f"{shown[j].item() // 60}:{shown[j].item() % 60:02d}",
                    "true_hour": y[j, 0].item(), "true_minute": y[j, 1].item(),
                    "pred_hour": pred_hour[j].item(), "pred_minute": round(pred_min[j].item() % 60, 2),
                    "raw_output": raw[j], "confidence": round(conf[j].item(), 4),
                    "error_min": round(err[j].item(), 2), "correct": int(correct[j].item()),
                    "hour_correct": int(pred_hour[j].item() == y[j, 0].item()),
                })
            start += len(y)
    return rows


def evaluate_all_clock_models(pattern="weights_saved/*TellTheTime*.pt", out_dir="results_saved"):
    """Evaluate every saved Tell-the-Time model on its test set.
    Writes results_saved/predictions/<model>.csv (every test image: true / predicted time, error, correct)
    and results_saved/summary.csv (one row per model). Returns the summary rows.
      accuracy     : classification -> right class (at the model's bin size); regression / multi-head -> exact minute
      hour_accuracy: right hour on the clock
      within_k     : common-sense error <= k minutes (comparable between all models)
      mae_min      : mean common-sense error in minutes (12-hour wrap-around)"""
    os.makedirs(f"{out_dir}/predictions", exist_ok=True)
    test_sets, summary = {}, []
    for path in sorted(glob.glob(pattern)):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            model, kind, n_out, size = build_from_weights(path)
        except Exception as e:                                                   # not a clock model of this shape
            print(f"skipped {name}: {type(e).__name__}: {str(e)[:80]}")
            continue
        if size not in test_sets:
            test_sets[size] = clock_test_set(size)
        rows = predict_clock(model, kind, n_out, test_sets[size])
        with open(f"{out_dir}/predictions/{name}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        errs = torch.tensor([r["error_min"] for r in rows])
        s = {"model": name, "kind": kind, "outputs": n_out, "size": size, "n_test": len(rows),
             "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 4),
             "hour_accuracy": round(sum(r["hour_correct"] for r in rows) / len(rows), 4),
             **{f"within_{k}": round((errs <= k).float().mean().item(), 4) for k in (1, 5, 15, 30)},
             "mae_min": round(errs.mean().item(), 2), "median_err_min": round(errs.median().item(), 2)}
        summary.append(s)
        print(f"{name:<60} {kind:<14} acc {s['accuracy']:.3f} | hour {s['hour_accuracy']:.3f} | "
              f"<=5 min {s['within_5']:.3f} | MAE {s['mae_min']:.1f} min")
    if summary:
        with open(f"{out_dir}/summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary[0].keys())
            writer.writeheader()
            writer.writerows(summary)
    return summary

summary = evaluate_all_clock_models()      # predictions per model in results_saved/predictions/, overview in results_saved/summary.csv