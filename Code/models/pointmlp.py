
import torch
import torch.nn as nn
import torch.nn.functional as F
from .efficient_kan import kan
# from torch import einsum
# from einops import rearrange, repeat


from pointnet2_ops import pointnet2_utils


def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class LocalGrouper(nn.Module):
    def __init__(self, channel, groups, kneighbors, use_xyz=True, normalize="center", **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channel=3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1,1,1,channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]

        # fps_idx = torch.multinomial(torch.linspace(0, N - 1, steps=N).repeat(B, 1).to(xyz.device), num_samples=self.groups, replacement=False).long()
        # fps_idx = farthest_point_sample(xyz, self.groups).long()
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.groups).long()  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        new_points = index_points(points, fps_idx)  # [B, npoint, d]

        idx = knn_point(self.kneighbors, xyz, new_xyz)
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, k, 3]
        grouped_points = index_points(points, idx)  # [B, npoint, k, d]
        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz],dim=-1)  # [B, npoint, k, d+3]
        if self.normalize is not None:
            if self.normalize =="center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize =="anchor":
                mean = torch.cat([new_points, new_xyz],dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]
            std = torch.std((grouped_points-mean).reshape(B,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
            grouped_points = (grouped_points-mean)/(std + 1e-5)
            grouped_points = self.affine_alpha*grouped_points + self.affine_beta

        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        return new_xyz, new_points


class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)


class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class PreExtraction(nn.Module):
    def __init__(self, channels, out_channels,  blocks=1, groups=1, res_expansion=1, bias=True,
                 activation='relu', use_xyz=True):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PreExtraction, self).__init__()
        in_channels = 3+2*channels if use_xyz else 2*channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion,
                                bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        x = self.transfer(x)
        batch_size, _, _ = x.size()
        x = self.operation(x)  # [b, d, k]
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu'):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):  # [b, d, g]
        return self.operation(x)

class ScanBranchFC(nn.Module):
    def __init__(self, input_dim=4, hidden_dims=[32,64,8], output_dim=8, activation='gelu'):
        super(ScanBranchFC, self).__init__()
        
        layers = []
        last_dim = input_dim
        act_fn = get_activation(activation)

        # 构建MLP层
        for hidden_dim in hidden_dims:
            layers.append(nn.Conv1d(last_dim, hidden_dim, kernel_size=1))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_fn)
            last_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Conv1d(last_dim, output_dim, kernel_size=1))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # 检查输入通道维度
        if x.size(1) != 4:
            raise ValueError(f"Expected input channel size 4, but got {x.size(1)}")
            
        return self.mlp(x)

class ScanBranchFutureKAN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=[8,16,8], output_dim=1):
        super(ScanBranchFutureKAN, self).__init__()
        kan_layers_strcture = [input_dim] + hidden_dim + [output_dim]
        self.kan_layers = kan.KAN(kan_layers_strcture)
        self.ln = nn.LayerNorm(output_dim)

    def forward(self, x):
       B, C, N = x.size()
       x = x.permute(0, 2, 1)  # [B, N, C]
       x_reshaped = x.reshape(-1, C)
       processed_x = self.kan_layers(x_reshaped)
       output = processed_x.reshape(B, N, -1)
       output = self.ln(output)
       output = output.permute(0, 2, 1)  # [B, output_dim, N]
       return output

    
class MultiScaleConv2(nn.Module):
    def __init__(self, in_channels, out_channels, num_scales=5, kernel_size=8, Dropout=0.3):
        super(MultiScaleConv2, self).__init__()
        self.num_scales = num_scales
        #并行多尺度卷积层 (使用ModuleList更清晰)
        self.convs = nn.ModuleList()
        for i in range(num_scales):
            kernel_size = 3 + i * 2
            padding = (kernel_size - 1) // 2
            self.convs.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True)
            )
        self.pooling = nn.MaxPool1d(kernel_size=8, padding=0)
        self.bn = nn.BatchNorm1d(out_channels * num_scales)
        self.dropout = nn.Dropout(Dropout)
        self.act = nn.GELU()

    def forward(self, x):
        pooled_outs = []
        for conv in self.convs:
            out = conv(x)
            out = self.pooling(out)
            pooled_outs.append(out)
        outs = torch.cat(pooled_outs, dim=1) # 形状: [B, 16 * 5, N/8] = [B, 80, N/8]
        outs = self.bn(outs)
        outs = self.act(outs)
        outs = self.dropout(outs)
        return outs
    
class MAB(nn.Module):
    """ Multihead Attention Block """
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        # Q, K 均需要是 [B, N, D] 格式
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        # 拆分多头
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        # 计算注意力
        A = torch.softmax(Q_.bmm(K_.transpose(1,2)) / self.dim_V**0.5, 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class PMA(nn.Module):
    """ Pooling by Multihead Attention """
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.randn(1, num_seeds, dim))
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        # X: [B, N, D]
        # self.S: [1, num_seeds, D] -> 广播为 [B, num_seeds, D]
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class ScanBranch(nn.Module):
    def __init__(self, KAN_dim=5, hidden_dim=512, output_dim=256):
        super(ScanBranch, self).__init__()
        self.output_dim = output_dim
        num_heads = 4
        
        self.KAN = ScanBranchFutureKAN(input_dim=5, hidden_dim=[8,16,8], output_dim=KAN_dim)
        self.bn = nn.BatchNorm1d(KAN_dim)

        # --- 1. 输入嵌入层 ---
        # 将 5 维输入点映射到模型的内部维度 hidden_dim
        self.embedding = nn.Linear(KAN_dim, hidden_dim)

        # --- 2. 编码器 (ISAB 块) ---
        # MAB 用于点集内部的自注意力
        self.encoder = nn.ModuleList([
            MAB(hidden_dim, hidden_dim, hidden_dim, num_heads, ln=True),
            MAB(hidden_dim, hidden_dim, hidden_dim, num_heads, ln=True)
        ])

        # --- 3. 解码器 (PMA 块) ---
        # PMA 用于从点集中池化信息，我们只需要一个全局特征，所以 num_seeds=1
        self.decoder = PMA(hidden_dim, num_heads, num_seeds=1, ln=True)

        # --- 4. 最终的输出层 ---
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.GELU()
        )

        self.bn2 = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: [B, 5, N]

        # 1. 通过 KAN 层处理输入点云
        x = self.KAN(x)
        x = self.bn(x)
        x = self.dropout(x)  # [B, KAN_dim, N]
        
        # 调整维度以适应 Linear 和 Attention 层: [B, N, 5]
        x = x.permute(0, 2, 1)

        # 1. 输入嵌入
        x = self.embedding(x)  # [B, N, hidden_dim]

        # 2. 通过编码器进行自注意力计算
        for mab_layer in self.encoder:
            # 对于自注意力，Q 和 K 是相同的
            x = mab_layer(x, x)

        # 3. 通过解码器进行池化，得到全局特征
        x = self.decoder(x)   # [B, 1, hidden_dim]
        
        # 移除序列长度为 1 的维度
        x = x.squeeze(1)      # [B, hidden_dim]

        # 4. 通过最后的FC层映射到目标维度
        x = self.output_fc(x) # [B, output_dim]
        x = self.bn2(x)
        return x


class Model(nn.Module):
    def __init__(self, points=1024, class_num=40, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="center",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[2, 2, 2, 2], EnableSideBranch=True, **kwargs):
        super(Model, self).__init__()
        self.stages = len(pre_blocks)
        self.class_num = class_num
        self.points = points
        self.embedding = ConvBNReLU1D(3, embed_dim, bias=bias, activation=activation)
        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        anchor_points = self.points
        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce
            # append local_grouper_list
            local_grouper = LocalGrouper(last_channel, anchor_points, kneighbor, use_xyz, normalize)  # [b,g,k,d]
            self.local_grouper_list.append(local_grouper)
            # append pre_block_list
            pre_block_module = PreExtraction(last_channel, out_channel, pre_block_num, groups=groups,
                                             res_expansion=res_expansion,
                                             bias=bias, activation=activation, use_xyz=use_xyz)
            self.pre_blocks_list.append(pre_block_module)
            # append pos_block_list
            pos_block_module = PosExtraction(out_channel, pos_block_num, groups=groups,
                                             res_expansion=res_expansion, bias=bias, activation=activation)
            self.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel

        final_future_dim = last_channel

        if EnableSideBranch:
            self.scan_branch = ScanBranch(output_dim=last_channel)
            self.attention = nn.Sequential(
                nn.Linear(last_channel * 2, last_channel),
                nn.Sigmoid()
            )
            self.drop = nn.Dropout(0.3)

        self.act = get_activation(activation)
        self.classifier = nn.Sequential(
            nn.Linear(final_future_dim, 256),
            nn.BatchNorm1d(256),
            self.act,
            nn.Dropout(0.3),
            nn.Linear(256, self.class_num),
            nn.Sigmoid()
        )

    def forward(self, x, scan, EnableSideBranch=True):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = self.embedding(x)  # B,D,N
        for i in range(self.stages):
            # Give xyz[b, p, 3] and fea[b, p, d], return new_xyz[b, g, 3] and new_fea[b, g, k, d]
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))  # [b,g,3]  [b,g,k,d]
            x = self.pre_blocks_list[i](x)  # [b,d,g]
            x = self.pos_blocks_list[i](x)  # [b,d,g]

        pointmlp_features = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)  # [b,d]

        if EnableSideBranch:
            scan = scan.permute(0, 2, 1)
            scan_features = self.scan_branch(scan)
            final_features = torch.cat([pointmlp_features, scan_features], dim=1)
            attention_weights = self.attention(torch.cat([pointmlp_features, scan_features], dim=1))
            final_features = pointmlp_features + attention_weights * scan_features
            final_features = self.act(final_features)
            final_features = self.drop(final_features)
        else:
            final_features = pointmlp_features

        x = self.classifier(final_features)
        return x




def pointMLP(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=64, groups=1, res_expansion=1.0,
                   activation="gelu", bias=False, use_xyz=False, normalize="anchor",
                   dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                   k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], **kwargs)


def pointMLPElite(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=32, groups=1, res_expansion=0.25,
                   activation="gelu", bias=False, use_xyz=False, normalize="anchor",
                   dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                   k_neighbors=[24,24,24,24], reducers=[2, 2, 2, 2], **kwargs)

if __name__ == '__main__':
    data = torch.rand(2, 3, 1024)
    print("===> testing pointMLP ...")
    model = pointMLP()
    out = model(data)
    print(out.shape)

