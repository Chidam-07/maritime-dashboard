"""Run the model on one scene and turn the output into absolute tracks."""
import numpy as np
import torch

from . import config as C


def xy_to_lonlat(x, y):
    lon = C.ORIGIN_LON + np.asarray(x) / C.METERS_PER_DEG_LON
    lat = C.ORIGIN_LAT + np.asarray(y) / C.METERS_PER_DEG_LAT
    return lon, lat


@torch.no_grad()
def predict_scene(model, data, device):
    data = data.to(device)
    preds, _, _ = model(data.x, data.edge_index, data.edge_attr)  # (N, T, 2) displacement from ref
    return preds.cpu().numpy()


def build_tracks(data, pred_disp):
    """All arrays are absolute positions in metres (same frame as the AIS pipeline)."""
    ref = data.ref_pos.cpu().numpy()                              # (N, 2)
    hist = data.x[:, :, :2].cpu().numpy() + ref[:, None, :]       # (N, obs, 2)
    actual = data.y.cpu().numpy() + ref[:, None, :]               # (N, pred, 2)
    pred = pred_disp + ref[:, None, :]                            # (N, pred, 2)
    return {"mmsi": data.mmsi.cpu().numpy(), "ref": ref, "hist": hist, "actual": actual, "pred": pred}
