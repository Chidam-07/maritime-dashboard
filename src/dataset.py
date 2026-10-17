"""Scene dataset, identical to the training notebook except it also returns MMSI ids."""
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class MaritimeGraphDataset(Dataset):
    def __init__(self, scene_dir, obs_len=8, pred_len=12, dist_thresh=3500.0):
        self.scene_paths = sorted(list(Path(scene_dir).glob("*.parquet")))
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.dist_thresh = dist_thresh

    def __len__(self):
        return len(self.scene_paths)

    def __getitem__(self, idx):
        scene_df = pl.read_parquet(self.scene_paths[idx])
        mmsis = scene_df.select("MMSI").unique().sort("MMSI")["MMSI"].to_list()
        N = len(mmsis)

        coords = np.zeros((self.seq_len, N, 5), dtype=np.float32)
        for n_i, mmsi in enumerate(mmsis):
            v_data = scene_df.filter(pl.col("MMSI") == mmsi).sort("time_dt")
            coords[:, n_i, :] = v_data.select(["x", "y", "vx", "vy", "cog"]).to_numpy()[: self.seq_len]

        traj = torch.from_numpy(coords)
        ref_pos = traj[self.obs_len - 1, :, :2]

        obs_traj = traj[: self.obs_len].clone()
        obs_traj[:, :, 0] -= ref_pos[:, 0].unsqueeze(0)
        obs_traj[:, :, 1] -= ref_pos[:, 1].unsqueeze(0)
        node_features = obs_traj.permute(1, 0, 2).contiguous()

        future_pos = traj[self.obs_len :, :, :2].clone()
        future_disp = future_pos - ref_pos.unsqueeze(0)
        y = future_disp.permute(1, 0, 2).contiguous()

        current_x, current_y = ref_pos[:, 0], ref_pos[:, 1]
        dx = current_x.unsqueeze(1) - current_x.unsqueeze(0)
        dy = current_y.unsqueeze(1) - current_y.unsqueeze(0)
        dist_matrix = torch.sqrt(dx**2 + dy**2)

        edge_indices, edge_attrs = [], []
        for i in range(N):
            for j in range(N):
                if i == j:
                    edge_indices.append([i, j])
                    edge_attrs.append([0.0, 0.0])
                elif dist_matrix[i, j] <= self.dist_thresh:
                    edge_indices.append([i, j])
                    norm_dist = dist_matrix[i, j].item() / self.dist_thresh
                    rel_bearing = torch.atan2(dy[j, i], dx[j, i]).item() / np.pi
                    edge_attrs.append([norm_dist, rel_bearing])

        return Data(
            x=node_features,
            edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_attrs, dtype=torch.float32),
            y=y, ref_pos=ref_pos, num_nodes=N,
            mmsi=torch.tensor(mmsis, dtype=torch.long),
        )
