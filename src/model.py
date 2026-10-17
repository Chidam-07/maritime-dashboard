"""Model definition, identical to the training notebook."""
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class SharedBiLSTMEncoder(nn.Module):
    def __init__(self, in_features=5, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.proj(torch.cat([h_n[-2], h_n[-1]], dim=-1))


class DynamicSpatioTemporalGATv2(nn.Module):
    def __init__(self, in_features=5, edge_dim=2, hidden_dim=64, heads=4, pred_len=12, dist_thresh=3500.0):
        super().__init__()
        self.model_name = "Model_B_DynamicGAT_Fixed_TeacherForced"
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.dist_thresh = dist_thresh
        self.temporal_enc = SharedBiLSTMEncoder(in_features, hidden_dim)
        self.step_gat = GATv2Conv(
            hidden_dim, hidden_dim // heads, heads=heads, edge_dim=edge_dim,
            concat=True, dropout=0.1, add_self_loops=False,
        )
        self.gat_norm = nn.LayerNorm(hidden_dim)
        self.gat_act = nn.ELU()
        self.decoder_cell = nn.LSTMCell(2 + hidden_dim, hidden_dim)
        self.out_head = nn.Linear(hidden_dim, 2)

    def _update_edge_attr(self, curr_pos, edge_index):
        src, tgt = edge_index[0], edge_index[1]
        diff = curr_pos[tgt] - curr_pos[src]
        dist = torch.norm(diff, p=2, dim=-1, keepdim=True)
        norm_dist = dist / self.dist_thresh
        bearing = torch.atan2(diff[:, 1:2], diff[:, 0:1]) / torch.pi
        is_self = (src == tgt).unsqueeze(-1)
        norm_dist = torch.where(is_self, torch.zeros_like(norm_dist), norm_dist)
        bearing = torch.where(is_self, torch.zeros_like(bearing), bearing)
        return torch.cat([norm_dist, bearing], dim=-1)

    def forward(self, x, edge_index, initial_edge_attr, target_y=None, teacher_forcing_ratio=0.0):
        num_nodes = x.size(0)
        device = x.device
        h_temp = self.temporal_enc(x)
        hx, cx = h_temp, torch.zeros((num_nodes, self.hidden_dim), device=device)
        step_disp = torch.zeros((num_nodes, 2), device=device)
        curr_cum = torch.zeros((num_nodes, 2), device=device)
        preds, alpha_trajectory = [], []

        for t in range(self.pred_len):
            edge_attr_t = initial_edge_attr if t == 0 else self._update_edge_attr(curr_cum, edge_index)
            h_spat_raw, (edge_idx_att, alpha_t) = self.step_gat(
                hx, edge_index, edge_attr=edge_attr_t, return_attention_weights=True
            )
            h_spat = self.gat_norm(self.gat_act(h_spat_raw) + hx)
            alpha_trajectory.append(alpha_t.unsqueeze(1))
            cell_in = torch.cat([step_disp, h_spat], dim=-1)
            hx, cx = self.decoder_cell(cell_in, (hx, cx))
            step_disp_pred = self.out_head(hx)
            curr_cum_pred = curr_cum + step_disp_pred
            preds.append(curr_cum_pred.unsqueeze(1))

            if target_y is not None and teacher_forcing_ratio > 0.0:
                force_mask = (torch.rand(num_nodes, device=device) < teacher_forcing_ratio).unsqueeze(-1)
                gt_cum = target_y[:, t, :]
                gt_disp = target_y[:, t, :] if t == 0 else target_y[:, t, :] - target_y[:, t - 1, :]
                curr_cum = torch.where(force_mask, gt_cum, curr_cum_pred)
                step_disp = torch.where(force_mask, gt_disp, step_disp_pred)
            else:
                curr_cum, step_disp = curr_cum_pred, step_disp_pred

        return torch.cat(preds, dim=1), (edge_idx_att, torch.cat(alpha_trajectory, dim=1)), hx
