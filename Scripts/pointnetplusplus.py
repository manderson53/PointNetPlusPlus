import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import fps

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        """
        Args:
            npoint: number of centroids to sample with FPS
            radius: radius for ball query neighborhood
            nsample: max number of neighbors to group
            in_channel: number of input feature channels per point (excluding xyz)
            mlp: list of output sizes for shared MLP layers
        """
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        self.mlps = nn.ModuleList()
        last_channel = in_channel + 3  # Add 3 for relative xyz coordinates

        for out_channel in mlp:
            self.mlps.append(nn.Conv2d(last_channel, out_channel, kernel_size=1))
            self.mlps.append(nn.BatchNorm2d(out_channel))
            self.mlps.append(nn.Identity())
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        Forward pass for the SA module.
        Args:
            xyz: (B, 3, N) tensor of point coordinates
            points: (B, D, N) tensor of point features, or None
        Returns:
            new_xyz: (B, 3, npoint) coordinates of sampled centroids
            new_points: (B, mlp[-1], npoint) aggregated local features
        """
        B, _, N = xyz.shape

        # print("Input xyz shape:", xyz.shape)  # should be [B, 3, N]
        # assert xyz.shape[2] >= self.npoint, f"Input N={xyz.shape[2]} less than npoint={self.npoint}"

        # 1. Sample centroids with FPS
        xyz_trans = xyz.permute(0, 2, 1)  # (B, N, 3)
        # try using torch_cluster fps for speed
        centroids_idx = farthest_point_sample(xyz_trans, self.npoint)  # (B, npoint)
        new_xyz = torch.gather(xyz_trans, 1, centroids_idx.unsqueeze(-1).repeat(1, 1, 3))  # (B, npoint, 3)
        #centroids_idx = fast_farthest_point_sample(xyz_trans, self.npoint)
        # new_xyz = torch.gather(xyz_trans, 1, centroids_idx.unsqueeze(-1).expand(-1, -1, 3))  # (B, npoint, 3)


        # 2. Group points within radius
        group_idx = query_ball_point(self.radius, self.nsample, xyz_trans, new_xyz)  # (B, npoint, nsample)

        # 3. Group xyz and features
        grouped_xyz = index_points(xyz_trans, group_idx)  # (B, npoint, nsample, 3)
        grouped_xyz -= new_xyz.unsqueeze(2)  # normalize coordinates by centroid

        if points is not None:
            points_trans = points.permute(0, 2, 1)  # (B, N, D)
            grouped_points = index_points(points_trans, group_idx)  # (B, npoint, nsample, D)
            new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)  # (B, npoint, nsample, 3+D)
        else:
            new_points = grouped_xyz  # (B, npoint, nsample, 3)

        # 4. Apply shared MLP and max pooling
        new_points = new_points.permute(0, 3, 2, 1)  # (B, 3+D, nsample, npoint)
        for layer in self.mlps:
            new_points = F.relu(layer(new_points))
        new_points = torch.max(new_points, 2)[0]  # max pooling over nsample dim (B, mlp[-1], npoint)

        return new_xyz.permute(0, 2, 1), new_points  # (B, 3, npoint), (B, mlp[-1], npoint)

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            self.mlp_bns.append(nn.Identity())

            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Args:
            xyz1: (B, 3, N) coordinates of target points to propagate to (dense)
            xyz2: (B, 3, S) coordinates of source points with features (sparser)
            points1: (B, D1, N) features to be concatenated (skip connection, may be None)
            points2: (B, D2, S) features to interpolate from source

        Returns:
            new_points: (B, mlp[-1], N) propagated and refined features
        """

        B, _, N = xyz1.shape
        _, _, S = xyz2.shape

        if S == 1:
            # If only one source point, repeat its features for all target points
            interpolated_points = points2.repeat(1, 1, N)
        else:
            # Compute pairwise squared distances between target and source points
            dists = square_distance(xyz1.permute(0, 2, 1), xyz2.permute(0, 2, 1))  # (B, N, S)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # Use 3 nearest neighbors

            dist_recip = 1.0 / (dists + 1e-8)  # inverse distances for weighting
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm  # normalize weights

            points2 = points2.permute(0, 2, 1)  # (B, S, D2)

            # Weighted sum of neighbor features
            interpolated_points = torch.sum(index_points(points2, idx) * weight.unsqueeze(-1), dim=2).permute(0, 2, 1)
            # Shape back to (B, D2, N)

        if points1 is not None:
            new_points = torch.cat([interpolated_points, points1], dim=1)  # concat skip features
        else:
            new_points = interpolated_points

        # Pass concatenated features through shared MLP layers
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))

        return new_points

class PointNet2Seg(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Set Abstraction layers: sample and encode features hierarchically
        self.sa1 = PointNetSetAbstraction(npoint=1024, radius=0.1, nsample=32, in_channel=0, mlp=[64, 64, 128])
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=0.2, nsample=32, in_channel=128, mlp=[128, 128, 256])
        
        # Feature Propagation layers: upsample and refine features to original resolution
        self.fp1 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256, mlp=[256, 128])

        # Final classification layers
        self.conv1 = nn.Conv1d(128, 128, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)


        self.conv2 = nn.Conv1d(128, num_classes, kernel_size=1)

    def forward(self, xyz):
        """
        xyz: input point cloud coordinates (B, 3, N)
        returns: per-point class logits (B, N, num_classes)
        """

        B, _, N = xyz.shape

        # 1) Hierarchical feature extraction through SA layers
        l1_xyz, l1_points = self.sa1(xyz, None)        # (B, 3, 1024), (B, 128, 1024)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # (B, 3, 256), (B, 256, 256)

        # 2) Feature Propagation: upsample and refine features
        l1_points = self.fp1(l1_xyz, l2_xyz, l1_points, l2_points) # (B, 256, 1024)
        l0_points = self.fp2(xyz, l1_xyz, None, l1_points)       # (B, 128, N)

        # 3) Final classification head
        x = F.relu(self.bn1(self.conv1(l0_points)))  # (B, 128, N)

        x = self.drop1(x)


        x = self.conv2(x)  # (B, num_classes, N)

        # transpose to (B, N, num_classes) for loss and evaluation
        x = x.transpose(1, 2).contiguous()

        return x
    

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: [B, N, 3] pointcloud
        npoint: int, number of points to sample
    Return:
        centroids: [B, npoint] indices of sampled points
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10   # initialize large distance
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  # initial farthest
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid_xyz = xyz[batch_indices, farthest, :].view(B,1,3)  # [B,1,3]
        dist = torch.sum((xyz - centroid_xyz)**2, dim=-1)         # [B, N]
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=1)[1]
    return centroids

def fast_farthest_point_sample(xyz, npoint):
    """
    Efficient batched FPS using torch_cluster.fps.

    Args:
        xyz (torch.Tensor): (B, N, 3) input points
        npoint (int): number of centroids to sample
    Returns:
        torch.Tensor: (B, npoint) indices of sampled points
    """
    B, N, _ = xyz.shape
    flat_xyz = xyz.reshape(B * N, 3)             # Flatten to (B*N, 3)
    batch = torch.arange(B, device=xyz.device).repeat_interleave(N)
    ratio = npoint / N                           # Fraction of points to sample
    indices = fps(flat_xyz, batch=batch, ratio=ratio, random_start=True)
    # If you want exactly npoint per batch, slice accordingly:
    indices = indices.view(B, -1)[:, :npoint]     # (B, npoint)
    return indices



def square_distance(src, dst):
    """
    Calculate squared Euclidean distance between each two points.
    src: [B, N, C]
    dst: [B, M, C]
    Output:
        dist: [B, N, M]
    """

    B, N, _ = src.shape
    _, M, _ = dst.shape
    
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # [B, N, M]
    dist += torch.sum(src ** 2, dim=-1).view(B, N, 1)
    dist += torch.sum(dst ** 2, dim=-1).view(B, 1, M)
    return dist


def query_ball_point(radius, nsample, xyz, new_xyz):
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]

    sqrdists = square_distance(new_xyz, xyz)  # [B, S, N]

    group_idx = torch.arange(N, device=xyz.device).view(1, 1, N).repeat(B, S, 1)  # [B, S, N]
    group_idx[sqrdists > radius**2] = N  # Mark invalid neighbors with N

    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]  # Take first nsample neighbors after sorting

    # Replace all invalid indices (equal to N) with zero - safe fallback index
    invalid_mask = group_idx == N
    valid_first_idx = group_idx[:, :, 0].clone()
    valid_first_idx[valid_first_idx == N] = 0

    group_first_exp = valid_first_idx.unsqueeze(-1).expand_as(group_idx)
    group_idx[invalid_mask] = group_first_exp[invalid_mask]

    # Final safety clamp (optional but good)
    group_idx = group_idx.clamp(0, N - 1)

    return group_idx.long()

def index_points(points, idx):
    """
    Input:
        points: (B, N, C)
        idx: (B, S, nsample)
    Return:
        new_points: (B, S, nsample, C)
    """
    B = points.shape[0]
    S = idx.shape[1]
    nsample = idx.shape[2]

    batch_indices = torch.arange(B, device=points.device).view(B, 1, 1).repeat(1, S, nsample)
    new_points = points[batch_indices, idx, :]
    return new_points

