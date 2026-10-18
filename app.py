"""Vessel trajectory review dashboard.  Run:  streamlit run app.py"""
from __future__ import annotations

import json
import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import folium
import streamlit as st
import torch
from folium.plugins import Fullscreen, MeasureControl, MiniMap
from streamlit_folium import st_folium

from src import config as C
from src.dataset import MaritimeGraphDataset
from src.inference import build_tracks, predict_scene, xy_to_lonlat
from src.metrics import per_vessel_errors, summarize
from src.model import DynamicSpatioTemporalGATv2

BLUE, RED, GREY = "#1f5fd6", "#d62839", "#8a8f98"

st.set_page_config(page_title="Maritime vessel trajectory prediction", layout="wide")


# ---------------------------------------------------------------- data / model
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DynamicSpatioTemporalGATv2(
        in_features=5, edge_dim=2, hidden_dim=64, heads=4,
        pred_len=C.PRED_LEN, dist_thresh=C.DIST_THRESH,
    )
    model.load_state_dict(torch.load(C.CHECKPOINT, map_location=device))
    return model.to(device).eval(), device


@st.cache_resource
def load_dataset():
    return MaritimeGraphDataset(C.SCENE_DIR, C.OBS_LEN, C.PRED_LEN, C.DIST_THRESH)


@st.cache_data
def load_manifest():
    with open(C.MANIFEST) as f:
        return json.load(f)


def day_splits(days):
    """Reproduce the train/validation day split used in training."""
    shuffled = list(days)
    random.seed(C.SPLIT_SEED)
    random.shuffle(shuffled)
    n_val = max(1, int(C.VAL_FRACTION * len(shuffled)))
    return {d: ("validation" if i < n_val else "training") for i, d in enumerate(shuffled)}


@st.cache_data(show_spinner="Running model on scene...")
def run_scene(idx: int):
    model, device = load_model()
    data = load_dataset()[idx]
    return build_tracks(data, predict_scene(model, data, device))


@st.cache_data(show_spinner="Evaluating every scene of the day...")
def evaluate_day(day: str, threshold: float):
    errs = [per_vessel_errors(t["actual"], t["pred"])
            for t in (run_scene(i) for i in load_manifest()[day])]
    return summarize(np.concatenate(errs, axis=0), threshold), len(errs)


# ---------------------------------------------------------------- figures
def _hover(mmsi, label, steps):
    return [f"MMSI {mmsi}<br>{label}<br>t+{C.DT_SECONDS * (k + 1)} s" for k in range(steps)]


LEGEND_HTML = """
<div style="position: fixed; bottom: 28px; left: 12px; z-index: 9999; background: rgba(255,255,255,.92);
            padding: 8px 12px; border-radius: 6px; font: 13px/1.6 sans-serif; box-shadow: 0 1px 4px rgba(0,0,0,.3);">
  <span style="color:#1f5fd6">&#9679;</span> Actual path<br>
  <span style="color:#d62839">&#9679;</span> Predicted path<br>
  <span style="color:#1c2430">&#9679;</span> Last observed position<br>
  <span style="color:#8a8f98">&#9473;</span> Final-point error
</div>
"""


def folium_map(t, sel, show_hist):
    allx = np.concatenate([t["actual"][sel, :, 0].ravel(), t["pred"][sel, :, 0].ravel(), t["hist"][sel, :, 0].ravel()])
    ally = np.concatenate([t["actual"][sel, :, 1].ravel(), t["pred"][sel, :, 1].ravel(), t["hist"][sel, :, 1].ravel()])
    lon, lat = xy_to_lonlat(allx, ally)

    m = folium.Map(location=[float(lat.mean()), float(lon.mean())], zoom_start=12, tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="Street").add_to(m)
    folium.TileLayer("CartoDB positron", name="Light").add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite",
    ).add_to(m)

    g_hist = folium.FeatureGroup("Observed history", show=show_hist).add_to(m)
    g_act = folium.FeatureGroup("Actual path").add_to(m)
    g_pred = folium.FeatureGroup("Predicted path").add_to(m)
    g_err = folium.FeatureGroup("Final-point error").add_to(m)
    g_ref = folium.FeatureGroup("Last observed position").add_to(m)

    def ll(arr):
        lo, la = xy_to_lonlat(arr[:, 0], arr[:, 1])
        return [[float(a), float(b)] for a, b in zip(la, lo)]

    def coord_tip(mmsi, label, p):
        return f"<b>MMSI {mmsi}</b> · {label}<br>Lat {p[0]:.5f}°<br>Lon {p[1]:.5f}°"

    def draw(group, pts, color, mmsi, label):
        folium.PolyLine(pts, color=color, weight=3, opacity=0.9).add_to(group)
        for k, p in enumerate(pts[1:], start=1):
            last = k == len(pts) - 1
            folium.CircleMarker(
                p, radius=7 if last else 4, color=color, fill=True, fill_color=color, fill_opacity=1, weight=1,
                tooltip=coord_tip(mmsi, f"{label}, t+{k * C.DT_SECONDS} s", p),
            ).add_to(group)

    for n in sel:
        mmsi, ref = int(t["mmsi"][n]), t["ref"][n]
        act = ll(np.vstack([ref, t["actual"][n]]))
        prd = ll(np.vstack([ref, t["pred"][n]]))
        hist = ll(t["hist"][n])
        folium.PolyLine(hist, color=BLUE, weight=2, opacity=0.4, dash_array="4 6").add_to(g_hist)
        for i, p in enumerate(hist[:-1]):
            folium.CircleMarker(
                p, radius=3, color=BLUE, fill=True, fill_color=BLUE, fill_opacity=0.5, weight=1,
                tooltip=coord_tip(mmsi, f"observed, t-{(len(hist) - 1 - i) * C.DT_SECONDS} s", p),
            ).add_to(g_hist)
        draw(g_act, act, BLUE, mmsi, "actual")
        draw(g_pred, prd, RED, mmsi, "predicted")
        err_m = float(np.linalg.norm(t["pred"][n][-1] - t["actual"][n][-1]))
        folium.PolyLine([act[-1], prd[-1]], color=GREY, weight=1.5, dash_array="5 5",
                        tooltip=f"MMSI {mmsi} · final error {err_m:.0f} m").add_to(g_err)
        folium.CircleMarker(
            act[0], radius=7, color="#ffffff", weight=2, fill=True, fill_color="#1c2430", fill_opacity=1,
            tooltip=coord_tip(mmsi, "last observed", act[0]),
        ).add_to(g_ref)

    m.fit_bounds([[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]], padding=(30, 30))
    Fullscreen().add_to(m)
    MiniMap(toggle_display=True, position="bottomright").add_to(m)
    MeasureControl(primary_length_unit="meters", position="topleft").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(LEGEND_HTML))
    return m


def local_figure(t, sel, show_hist):
    fig, seen = go.Figure(), set()

    def add(x, y, color, group, label, mode="lines+markers", dash=None, width=2.5, opacity=1.0, size=7, text=None):
        fig.add_trace(go.Scatter(
            x=x, y=y, mode=mode, line=dict(color=color, width=width, dash=dash),
            marker=dict(size=size, color=color), opacity=opacity, name=label, legendgroup=group,
            showlegend=group not in seen, text=text, hoverinfo="text" if text else "x+y",
        ))
        seen.add(group)

    for n in sel:
        m, ref = int(t["mmsi"][n]), t["ref"][n]
        if show_hist:
            add(t["hist"][n][:, 0], t["hist"][n][:, 1], BLUE, "hist", "Observed history",
                mode="lines", dash="dot", opacity=0.5, text=f"MMSI {m} observed")
        a, p = np.vstack([ref, t["actual"][n]]), np.vstack([ref, t["pred"][n]])
        add(a[:, 0], a[:, 1], BLUE, "act", "Actual path", text=f"MMSI {m} actual")
        add(p[:, 0], p[:, 1], RED, "pred", "Predicted path", text=f"MMSI {m} predicted")
        add([a[-1, 0], p[-1, 0]], [a[-1, 1], p[-1, 1]], GREY, "gap", "Final-point error",
            mode="lines", width=1.5, text=f"MMSI {m} final error")
    add(t["ref"][sel, 0], t["ref"][sel, 1], "#1c2430", "ref", "Last observed position",
        mode="markers", size=10, text=[f"MMSI {int(t['mmsi'][n])}" for n in sel])
    fig.update_layout(
        height=640, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="East (m)", scaleanchor="y"), yaxis=dict(title="North (m)"),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)"),
    )
    return fig


def error_figure(err, mmsi, threshold):
    x = (np.arange(err.shape[1]) + 1) * C.DT_SECONDS
    fig = go.Figure()
    for n in range(err.shape[0]):
        fig.add_trace(go.Scatter(x=x, y=err[n], mode="lines", line=dict(color=RED, width=1),
                                 opacity=0.3, showlegend=False,
                                 hovertemplate=f"MMSI {int(mmsi[n])}<br>%{{x}} s: %{{y:.0f}} m<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=err.mean(0), mode="lines+markers", name="Mean over vessels",
                             line=dict(color=RED, width=3)))
    fig.add_hline(y=threshold, line=dict(color=GREY, dash="dash"),
                  annotation_text=f"accuracy threshold {threshold:.0f} m")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="Seconds ahead", yaxis_title="Position error (m)")
    return fig


# ---------------------------------------------------------------- sidebar
if not C.MANIFEST.exists() or not C.CHECKPOINT.exists():
    st.error(f"Missing files. Expected scenes at `{C.SCENE_DIR}` (with day_to_scenes.json) "
             f"and the checkpoint at `{C.CHECKPOINT}`.")
    st.stop()

manifest = load_manifest()
splits = day_splits(manifest.keys())
days = sorted(manifest)
default_day = next(i for i, d in enumerate(days) if splits[d] == "validation")

st.sidebar.header("Scene")
day = st.sidebar.selectbox("Day", days, index=default_day)
scene_id = st.sidebar.selectbox("Scene", manifest[day], key=f"scene_{day}",
                                format_func=lambda i: f"Scene {i:05d}")

tracks = run_scene(scene_id)
all_mmsi = [int(m) for m in tracks["mmsi"]]

st.sidebar.header("Display")
chosen = st.sidebar.multiselect("Vessels", all_mmsi, default=all_mmsi, key=f"ves_{scene_id}")
show_hist = st.sidebar.checkbox("Show observed history", value=True)
threshold = st.sidebar.slider("Accuracy threshold (m)", 25, 500, 100, step=25,
                              help="A predicted point counts as accurate if it is within this distance of the true position.")

sel = [all_mmsi.index(m) for m in chosen]
if not sel:
    st.info("Select at least one vessel in the sidebar.")
    st.stop()

# ---------------------------------------------------------------- main
st.title("Maritime vessel trajectory prediction")
st.caption(f"{day} · scene {scene_id:05d} · {len(all_mmsi)} vessels · "
           f"{C.OBS_LEN * C.DT_SECONDS // 60} min observed, {C.PRED_LEN * C.DT_SECONDS // 60} min predicted")

err_all = per_vessel_errors(tracks["actual"], tracks["pred"])
s = summarize(err_all[sel], threshold)

c1, c2, c3 = st.columns(3)
c1.metric("Average error (ADE)", f"{s['ade']:.1f} m")
c2.metric("Final-point error (FDE)", f"{s['fde']:.1f} m")
c3.metric(f"Accuracy within {threshold} m", f"{s['accuracy']:.1f} %")

tab_map, tab_local, tab_err, tab_tbl = st.tabs(["Map", "Local plane", "Error over time", "Per vessel"])
with tab_map:
    st_folium(folium_map(tracks, sel, show_hist), height=640, use_container_width=True,
              returned_objects=[], key=f"map_{scene_id}_{show_hist}_{'-'.join(map(str, sel))}")
with tab_local:
    st.plotly_chart(local_figure(tracks, sel, show_hist), use_container_width=True)
with tab_err:
    st.plotly_chart(error_figure(err_all[sel], tracks["mmsi"][sel], threshold), use_container_width=True)
with tab_tbl:
    df = pd.DataFrame({
        "MMSI": [all_mmsi[n] for n in sel],
        "ADE (m)": err_all[sel].mean(1).round(1),
        "FDE (m)": err_all[sel, -1].round(1),
        f"Accurate points (% within {threshold} m)": ((err_all[sel] <= threshold).mean(1) * 100).round(1),
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()
st.subheader(f"Whole day: {day}")
if st.button("Evaluate every scene of this day"):
    ds, n_scenes = evaluate_day(day, float(threshold))
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("ADE", f"{ds['ade']:.1f} m")
    d2.metric("FDE", f"{ds['fde']:.1f} m")
    d3.metric(f"Accuracy within {threshold} m", f"{ds['accuracy']:.1f} %")
    d4.metric("Scenes / vessels", f"{n_scenes} / {ds['vessels']}")

st.caption("Blue: actual path. Red: predicted path. Black dot: last observed position, where the prediction starts. "
           "ADE is the mean error over all 12 predicted points, FDE the error at the last one. "
           "Accuracy is the share of predicted points within the chosen threshold of the true position.")
