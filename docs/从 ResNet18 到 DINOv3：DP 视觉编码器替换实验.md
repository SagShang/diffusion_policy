## 1. 实验目标

本轮实验只改 DP 的视觉前端，不改扩散策略主体。目标有三项：

- 验证将视觉编码器从 ResNet18 替换为 DINOv3 `vits16` 是否能提升成功率。
- 验证“冻结 backbone + 只训练 adapter”是否足够。
- 在效果不足时，比较两种轻量微调路线：
  `partial unfreeze` 与 `LoRA`。

实验范围固定如下：

- policy：DP
- 数据设置：`demo_clean`
- 示范数：`50`
- 任务：`adjust_bottle`、`stack_blocks_two`
- 评测：每个主结果按 `100` 次 rollout 统计成功率

## 2. 核心结论

先给最终结论，便于快速定位：

- 冻结版 DINOv3 可以稳定接入 DP，但效果不够强。`adjust_bottle` 只达到 `64.0%`。
- 当输入升级到 `320`、adapter 改为 `spatial_softmax`、并允许轻量微调后，DINOv3 开始明显超过原先视觉基线。
- `adjust_bottle` 上，`partial unfreeze` 与 `LoRA` 都达到 `100.0%`。
- `stack_blocks_two` 上，冻结版 `adapter224 = 19.0%`，`partial unfreeze = 47.0%`，`LoRA = 42.0%`，三者都高于既有 ResNet18 基线约 `7%`。
- 从成本看，冻结版 DINOv3 并不更贵；真正增加成本的是 `320 + partial unfreeze / LoRA`，而且主要增加的是 step 时间，不是显存峰值。

## 3. 代码改动概览

本轮实现涉及以下核心文件：

- `policy/DP/diffusion_policy/model/vision/dinov3_adapter.py`
- `policy/DP/diffusion_policy/model/vision/model_getter.py`
- `policy/DP/diffusion_policy/model/vision/multi_image_obs_encoder.py`
- `policy/DP/diffusion_policy/workspace/robotworkspace.py`
- `policy/DP/diffusion_policy/config/robot_dp_14_dinov3_vits16_adapter224.yaml`
- `policy/DP/diffusion_policy/config/robot_dp_14_dinov3_vits16_unfreeze320.yaml`
- `policy/DP/diffusion_policy/config/robot_dp_14_dinov3_vits16_lora320.yaml`

功能分工如下：

| 文件                           | 作用                                               |
| ---------------------------- | ------------------------------------------------ |
| `dinov3_adapter.py`          | 封装 DINOv3 backbone、图像预处理、层融合、adapter、LoRA、部分解冻逻辑 |
| `model_getter.py`            | 新增 `get_dinov3_adapter` 工厂函数                     |
| `multi_image_obs_encoder.py` | 支持共享一个 backbone、按 `camera_key` 选择相机 adapter      |
| `robotworkspace.py`          | 将优化器参数拆成 base/backbone/LoRA 三组，分别设学习率            |

本轮没有修改 DP 的以下部分：

- 扩散 U-Net 结构
- 动作维度与动作头
- 噪声调度器
- 时序窗口设置

因此，效果变化可以主要归因于视觉编码器与其适配方式。

## 4. 实现细节

这一节按“输入如何进 DINOv3，再如何变成 DP 使用的条件向量”来写。为了可复现，下面的数学描述和代码实现逐项对应。

### 4.1 符号约定

设一帧 RGB 图像为

$$
x \in [0,1]^{3 \times H \times W}.
$$

本项目中 `D435` 头相机的原始分辨率为

$$
H = 240,\quad W = 320.
$$

DP 使用 `n_obs_steps = 3` 帧观测。  
本实验的低维状态与动作都使用双臂关节驱动目标的同一参数化：

$$
q_t =
\left[
q^{L}_{t,1:6},\,
g^{L}_t,\,
q^{R}_{t,1:6},\,
g^{R}_t
\right]
\in \mathbb{R}^{14},
$$

其中：

- `L / R` 分别表示左臂与右臂
- $q^{L}_{t,1:6}, q^{R}_{t,1:6}$ 是两侧 6 关节驱动目标
- $g^{L}_t, g^{R}_t$ 是两侧归一化 gripper 开合量

动作向量也采用同构表示：

$$
a_t =
\left[
a^{L}_{t,1:6},\,
\alpha^{L}_t,\,
a^{R}_{t,1:6},\,
\alpha^{R}_t
\right]
\in \mathbb{R}^{14}.
$$

视觉编码器输出单帧向量

$$
v_t \in \mathbb{R}^{512}.
$$

因此，单帧最终观测特征为

$$
o_t = [v_t, q_t] \in \mathbb{R}^{526}.
$$

DP 看到的条件向量为前 `3` 帧拼接：

$$
g = [o_{t-2}, o_{t-1}, o_t] \in \mathbb{R}^{1578}.
$$

### 4.2 图像预处理

对应代码：`DINOv3ImagePreprocessor`。

给定长边目标尺寸 `L`，先按长边等比例缩放：

$$
s = \frac{L}{\max(H, W)},
$$

$$
H_r = \mathrm{round}(H s), \quad W_r = \mathrm{round}(W s).
$$

然后将高宽 pad 到 patch size `p = 16` 的倍数：

$$
\hat H = 16 \left\lceil \frac{H_r}{16} \right\rceil,\quad
\hat W = 16 \left\lceil \frac{W_r}{16} \right\rceil.
$$

归一化采用 DINOv3 对应的 ImageNet 统计量：

$$
\mu = (0.485, 0.456, 0.406), \quad
\sigma = (0.229, 0.224, 0.225),
$$

$$
x_n = \frac{x_r - \mu}{\sigma}.
$$

实现上的关键点：

- resize 使用 `bicubic`
- `antialias=True`
- pad 为最小零填充，左右和上下尽量对称
- `MultiImageObsEncoder` 里必须设 `imagenet_norm=False`
  因为 DINO 预处理内部已经归一化，不能再做一次

对本项目的 `D435` 输入，两个主配置对应的实际尺寸为：

| 配置 | 原图 | resize 后 | pad 后 | patch 网格 |
| --- | --- | --- | --- | --- |
| `adapter224` | `320 x 240` | `224 x 168` | `224 x 176` | `14 x 11` |
| `unfreeze320` / `lora320` | `320 x 240` | `320 x 240` | `320 x 240` | `20 x 15` |

也就是说：

- `224` 版本会发生一次最小 padding
- `320` 版本对 `D435` 输入不需要额外 padding

### 4.3 DINOv3 输出到帧特征的映射

#### 4.3.1 token 处理

对应代码：`third_party/dinov3/dinov3/models/vision_transformer.py` 的 `get_intermediate_layers(..., reshape=True, norm=True)`。

DINOv3 `vits16` 的 token 结构为：

$$
[t_{\text{cls}}, t_{\text{reg}}^{(1)}, t_{\text{reg}}^{(2)}, t_{\text{reg}}^{(3)}, t_{\text{reg}}^{(4)}, t_{\text{patch}}^{(1)}, \dots, t_{\text{patch}}^{(N)}].
$$

其中前 `5` 个 token 正好是：

- `1` 个 CLS token
- `4` 个 register token

当前实现没有手写“切掉前 5 个 token 再 reshape”，而是直接调用 DINOv3 官方接口：

$$
\texttt{get\_intermediate\_layers(..., reshape=True)}
$$

该接口内部已经执行了

$$
[t_{\text{patch}}^{(1)}, \dots, t_{\text{patch}}^{(N)}]
$$

的提取与 reshape，因此与“去掉前 `5` 个 token 再恢复成 feature map”是等价的。

#### 4.3.2 层融合

对应代码：`DINOv3AdapterModel._fuse_feature_maps`。

设取到的最后 `K` 层 patch feature map 为

$$
F^{(l_1)}, F^{(l_2)}, \dots, F^{(l_K)}, \quad
F^{(l_k)} \in \mathbb{R}^{C \times h \times w}.
$$

本轮最佳配置使用：

$$
K = 4.
$$

融合方式为可学习 softmax 加权：

$$
\alpha = \mathrm{softmax}(a), \quad a \in \mathbb{R}^{K},
$$

$$
F = \sum_{k=1}^{K} \alpha_k F^{(l_k)}.
$$

对 `dinov3_vits16`，有

$$
C = 384.
$$

因此融合后的张量为

$$
F \in \mathbb{R}^{384 \times h \times w}.
$$

#### 4.3.3 相机 adapter

对应代码：`DINOv3CameraAdapter`。

adapter 先做通道压缩和局部建模。若记 `d_a = 256`，则：

$$
Z = \phi(F) \in \mathbb{R}^{d_a \times h \times w},
$$

其中 `\phi` 由以下顺序组成：

1. `1 x 1 Conv(384 -> 256)`
2. `GroupNorm`
3. `GELU`
4. `3 x 3 Depthwise Conv`
5. `1 x 1 Pointwise Conv`
6. `GroupNorm`
7. `GELU`

本轮出现过两种 pooling。

`attention` 版本：

$$
\beta_i = \mathrm{softmax}(w^\top z_i), \quad
z_i \in \mathbb{R}^{256},
$$

$$
u = \sum_i \beta_i z_i \in \mathbb{R}^{256}.
$$

`spatial_softmax` 版本：

$$
P_c(u, v) = \mathrm{softmax}(Z_c(u, v)),
$$

$$
\bar x_c = \sum_{u,v} P_c(u,v) x_v,\quad
\bar y_c = \sum_{u,v} P_c(u,v) y_u,
$$

$$
u = [\bar x_1, \bar y_1, \dots, \bar x_{256}, \bar y_{256}] \in \mathbb{R}^{512}.
$$

最后统一通过

$$
v = W \cdot \mathrm{LayerNorm}(u) + b
$$

投到单帧输出维度

$$
v \in \mathbb{R}^{512}.
$$

这一点很关键：

- `attention pooling` 时，adapter 中间向量维度是 `256`
- `spatial_softmax` 时，中间向量维度是 `512`
- 但两者最后都被映射成 `512` 维帧特征

### 4.4 多相机支持

对应代码：`MultiImageObsEncoder.forward` 与 `DINOv3AdapterModel.set_camera_keys`。

若系统有多个 RGB 观测，当前实现使用：

$$
\text{共享 backbone } B_\theta,
$$

$$
\text{每个相机独立 adapter } A_{\phi^{(k)}}.
$$

因此第 `k` 个相机的帧特征为

$$
v_t^{(k)} = A_{\phi^{(k)}}(B_\theta(x_t^{(k)})).
$$

在代码中通过 `camera_key` 区分 adapter。  
本轮主实验使用的 `default_task_14` 实际观测为：

- `head_cam`
- `agent_pos`

因此虽然多相机能力已经实现，但这两组主实验实际只用到了 `head_cam`。

### 4.5 微调策略

#### 4.5.1 冻结版 `adapter224`

配置文件：`robot_dp_14_dinov3_vits16_adapter224.yaml`

- `resize_long_edge = 224`
- `pooling = attention`
- `freeze_backbone = True`
- `fuse_layers` 未显式设置，实际为单层输出
- `batch_size = 128`

这版的训练参数只来自：

$$
\Theta_{\text{train}} = \Theta_{\text{adapter}} \cup \Theta_{\text{DP}}.
$$

#### 4.5.2 部分解冻 `unfreeze320`

配置文件：`robot_dp_14_dinov3_vits16_unfreeze320.yaml`

- `resize_long_edge = 320`
- `pooling = spatial_softmax`
- `fuse_layers = 4`
- `trainable_blocks = 2`
- `trainable_norm = True`
- `backbone_lr_scale = 0.1`
- `batch_size = 64`

`dinov3_vits16` 一共有 `12` 个 transformer blocks。  
这里解冻的是最后 `2` 个 block，即 1-based 编号下的第 `11`、`12` 层。

若记 backbone 参数为 `\Theta_B`，adapter 参数为 `\Theta_A`，DP 主体参数为 `\Theta_P`，则训练集为：

$$
\Theta_{\text{train}} = \Theta_A \cup \Theta_P \cup \Theta_{B,\text{last2}} \cup \Theta_{\text{norm}}.
$$

#### 4.5.3 LoRA `lora320`

配置文件：`robot_dp_14_dinov3_vits16_lora320.yaml`

- `resize_long_edge = 320`
- `pooling = spatial_softmax`
- `fuse_layers = 4`
- `lora_rank = 8`
- `lora_alpha = 16`
- `lora_dropout = 0.05`
- `lora_blocks = 4`
- `lora_targets = ("attn.qkv", "attn.proj")`
- `batch_size = 64`

LoRA 注入最后 `4` 个 block，即 1-based 编号下的第 `9` 到 `12` 层。

对任意被替换的线性层

$$
y = Wx,
$$

LoRA 后变为

$$
y = Wx + \frac{\alpha}{r} BAx,
$$

其中

$$
r = 8,\quad \alpha = 16.
$$

当前实现中，原线性层权重冻结，只训练低秩增量矩阵 `A` 和 `B`。

### 4.6 与 DP 的连接方式

DP 本体没有改，只换了视觉编码器。

训练时，`DiffusionUnetImagePolicy.compute_loss` 做的事可以写成：

1. 对观测和动作做归一化
2. 取前 `n_obs_steps = 3` 帧，经过 obs encoder 得到条件向量 `g`
3. 对动作轨迹 `a_{1:H}` 加噪，`H = 8`
4. 采样扩散步 $\tau \sim \mathcal{U}\{0,\dots,99\}$
5. 训练网络预测噪声

即目标函数为

$$
\epsilon \sim \mathcal{N}(0, I),
$$

$$
\tilde a_\tau = \sqrt{\bar\alpha_\tau} a + \sqrt{1-\bar\alpha_\tau}\epsilon,
$$

$$
\mathcal{L} =
\mathbb{E}_{a,g,\epsilon,\tau}
\left[
\left\|
\hat\epsilon_\theta(\tilde a_\tau, \tau, g) - \epsilon
\right\|_2^2
\right].
$$

也就是说，视觉编码器唯一职责是把每帧图像变成 DP 可以消费的 `512` 维表示；DP 的扩散训练方式本身不变。

### 4.7 优化器参数组

对应代码：`RobotWorkspace._create_optimizer` 与 `DINOv3AdapterModel.get_optimizer_param_groups`。

当前实现把可训练参数拆成三组：

$$
\Theta_{\text{base}},\quad \Theta_{\text{lora}},\quad \Theta_{\text{backbone}}.
$$

学习率分别为：

$$
\eta_{\text{base}} = 10^{-4},
$$

$$
\eta_{\text{lora}} = 10^{-4} \cdot \texttt{lora\_lr\_scale},
$$

$$
\eta_{\text{backbone}} = 10^{-4} \cdot \texttt{backbone\_lr\_scale}.
$$

在 `unfreeze320` 中：

$$
\eta_{\text{backbone}} = 10^{-5}.
$$

这一点直接对应配置里的：

- `lr = 1e-4`
- `backbone_lr_scale = 0.1`

如果不做这种分组，而让解冻后的 backbone 和 DP 主体共用同一学习率，训练稳定性会明显变差。

## 5. 实验配置

### 5.1 硬件与软件

| 项目 | 配置 |
| --- | --- |
| GPU | `8 x NVIDIA GeForce RTX 4090 48GB` |
| Driver | `535.261.03` |
| Python | `3.10.19` |
| PyTorch | `2.4.1+cu121` |
| Conda 环境 | `RoboTwin` |

### 5.2 机器人、动作与传感器配置

本轮 `demo_clean` 配置实际使用的 embodiment 为 `aloha-agilex`，且是同构双臂模式：

$$
\texttt{embodiment} = [\texttt{aloha-agilex}],
\qquad
\texttt{dual\_arm\_embodied} = \mathrm{True}.
$$

机器人与观测配置如下：

| 项目 | 实际设置 |
| --- | --- |
| 机器人 embodiment | `aloha-agilex` |
| 左臂关节 | `fl_joint1` 到 `fl_joint6` |
| 右臂关节 | `fr_joint1` 到 `fr_joint6` |
| 左右夹爪 | `fl_joint7/8`、`fr_joint7/8` mimic |
| 低维状态维度 | `14 = 6 + 1 + 6 + 1` |
| 动作维度 | `14 = 6 + 1 + 6 + 1` |
| 双臂基座姿态 | `robot_pose = [0, -0.65, 0.0, 0.707, 0, 0, 0.707]` |
| 头相机 | `D435`, `320 x 240`, `fovy = 37°` |
| 腕部相机 | 左右各一台 `D435`, `320 x 240` |
| DP 实际使用的图像输入 | 仅 `head_cam` |

静态相机中，与本实验最相关的是头相机：

$$
p_{\text{head}} = (-0.032,\,-0.45,\,1.35),
$$

$$
f_{\text{head}} = (0,\,0.6,\,-0.8),
\qquad
\ell_{\text{head}} = (-1,\,0,\,0).
$$

原始采集还会保存 `front_camera`、`left_camera`、`right_camera` 的 RGB 与标定信息，但 DP 在 `RobotImageDataset` 中只读取：

$$
\{\texttt{head\_camera},\ \texttt{state},\ \texttt{action}\}.
$$

因此本报告的视觉比较虽然建立在多机位原始采集之上，但训练输入严格是单头相机。

### 5.3 原始数据目录与 zarr 结构

当前工作区中的原始数据目录是软链接。例如：

```bash
data/adjust_bottle/demo_clean -> /data1/datasets/RoboTwin/RoboTwin2.0/dataset/adjust_bottle/aloha-agilex_clean_50
```

每个任务的 `demo_clean` 原始目录结构如下：

```text
data/{task_name}/demo_clean/
├── data/
│   ├── episode0.hdf5
│   ├── ...
│   └── episode49.hdf5
├── _traj_data/
│   ├── episode0.pkl
│   ├── ...
│   └── episode49.pkl
├── instructions/
│   ├── episode0.json
│   ├── ...
│   └── episode49.json
└── scene_info.json
```

其中：

- `data/episode{k}.hdf5` 是逐帧观测与关节目标
- `_traj_data/episode{k}.pkl` 是规划轨迹缓存
- `scene_info.json` 记录每个 episode 的对象实例和臂选择占位符
- `instructions/episode{k}.json` 由任务模板和 `scene_info` 自动生成，包含 `seen/unseen` 两组语言描述

单个原始 `episode{k}.hdf5` 至少包含以下关键字段：

```text
/observation/head_camera/{rgb,intrinsic_cv,extrinsic_cv,cam2world_gl}
/observation/front_camera/{rgb,intrinsic_cv,extrinsic_cv,cam2world_gl}
/observation/left_camera/{rgb,intrinsic_cv,extrinsic_cv,cam2world_gl}
/observation/right_camera/{rgb,intrinsic_cv,extrinsic_cv,cam2world_gl}
/joint_action/{left_arm,left_gripper,right_arm,right_gripper,vector}
/endpose/{left_endpose,left_gripper,right_endpose,right_gripper}
```

DP 训练前会通过 `policy/DP/process_data.py` 将原始 HDF5 压缩成 zarr，并只保留三类数组：

$$
\texttt{head\_camera} \in \{0,\dots,255\}^{N \times 3 \times 240 \times 320},
$$

$$
\texttt{state} \in \mathbb{R}^{N \times 14},
\qquad
\texttt{action} \in \mathbb{R}^{N \times 14}.
$$

若原始 episode 长度为 `F` 帧，转换后的监督样本长度为 `F-1`，其时序对齐关系为：

$$
s_t = q_t,\quad
x_t = I_t,\quad
a_t = q_{t+1},
\qquad t = 0,\dots,F-2.
$$

也就是说，zarr 中每个样本使用当前时刻图像与关节状态，预测下一时刻的 14 维关节目标。  
`episode_ends` 存的是各 episode 在拼接后数组中的累计终止下标：

$$
\texttt{episode\_ends}[e] = \sum_{i=1}^{e}(F_i - 1).
$$

本轮实际生成的 zarr 规模为：

| 数据集 | 数组 key | 总 transition 数 | 备注 |
| --- | --- | --- | --- |
| `adjust_bottle-demo_clean-50.zarr` | `action`, `head_camera`, `state` | `7188` | `50` 个 episode |
| `stack_blocks_two-demo_clean-50.zarr` | `action`, `head_camera`, `state` | `15647` | `50` 个 episode |

### 5.4 场景、随机化与任务协议

`demo_clean` 不是泛指“简单数据”，而是一个几乎关闭域随机化的固定实验配置：

| 项目 | `demo_clean` 设置 |
| --- | --- |
| `random_background` | `False` |
| `cluttered_table` | `False` |
| `random_head_camera_dis` | `0` |
| `random_table_height` | `0` |
| `random_light` | `False` |
| `crazy_random_light_rate` | `0` |
| `collect_head_camera` | `True` |
| `collect_wrist_camera` | `True` |

基础场景由一张桌子和背景墙组成。桌面几何参数固定为：

$$
\text{length} = 1.2,\quad
\text{width} = 0.7,\quad
\text{height} = 0.74.
$$

这意味着本轮实验不考察背景纹理、桌面高度、光照和相机距离扰动，而主要考察视觉编码器替换本身。

两个主任务的初始化与成功判定如下。

`adjust_bottle`：

- bottle 类别从 `model_id ∈ {13,16}` 中采样
- bottle 初始朝向由 `qpose_tag ∈ {0,1}` 决定，进而决定应由左臂还是右臂执行
- 成功条件写成：

$$
\big(qpose\_tag = 0 \land x_{\text{bottle}} < -0.15\big)
\ \lor\
\big(qpose\_tag = 1 \land x_{\text{bottle}} > 0.15\big),
$$

并同时满足

$$
z_{\text{bottle}} > 0.9.
$$

- 评测 step 上限：`400`

`stack_blocks_two`：

- 两个方块尺寸固定，半边长为 `0.025`
- 初始位姿随机采样于

$$
x \in [-0.28, 0.28],\quad
y \in [-0.08, 0.05],\quad
z = 0.741 + 0.025,
$$

同时要求两个方块彼此分离，并避开桌面中央预留区域
- 第一块被放到中心目标位姿

$$
(0,\,-0.13,\,0.75 + \texttt{table\_z\_bias})
$$

- 第二块需要堆叠到第一块上方，成功条件为

$$
\left|
p_{\text{block2}} -
\big[
x_{\text{block1}},
y_{\text{block1}},
z_{\text{block1}} + 0.05
\big]
\right| < (0.025, 0.025, 0.012),
$$

并且左右夹爪均处于打开状态
- 评测 step 上限：`800`

评测协议还有两个容易遗漏的细节：

1. 语言指令使用 `unseen` split，不是训练期的 `seen` 指令。
2. `100` 次评测并不是“连续取 100 个随机种子直接跑”。实际流程是从

$$
\texttt{st\_seed} = 100000 \cdot (1 + \texttt{seed})
$$

开始递增采样，只把能被 expert rollout 成功生成、且 `play_once()` 后满足 `check_success()` 的场景计入分母。随后 policy 在同一场景上执行并统计成功率。

因此，这里的成功率口径是“在 expert 可解的有效测试场景上，policy 的成功比例”。

### 5.5 统一训练超参数

除视觉相关项外，其余 DP 设置保持不变：

| 项目 | 数值 |
| --- | --- |
| `horizon` | `8` |
| `n_obs_steps` | `3` |
| `n_action_steps` | `6` |
| optimizer | `AdamW` |
| base lr | `1e-4` |
| lr scheduler | `cosine` |
| warmup | `500` |
| epoch | `600` |
| checkpoint every | `300` |
| EMA | `True` |

### 5.6 对比配置

| 名称                | 配置文件                                         | 关键差异                                             |
| ----------------- | -------------------------------------------- | ------------------------------------------------ |
| ResNet18 baseline | `robot_dp_14.yaml`                           | 原始视觉编码器                                          |
| `adapter224`      | `robot_dp_14_dinov3_vits16_adapter224.yaml`  | `224 + attention + frozen`                       |
| `unfreeze320`     | `robot_dp_14_dinov3_vits16_unfreeze320.yaml` | `320 + spatial_softmax + fuse4 + last2 unfreeze` |
| `lora320`         | `robot_dp_14_dinov3_vits16_lora320.yaml`     | `320 + spatial_softmax + fuse4 + LoRA(last4)`    |

### 5.7 主实验矩阵

| 任务 | 数据 | 评测次数 | 备注 |
| --- | --- | --- | --- |
| `adjust_bottle` | `demo_clean`, `50` demos | `100` | 单次正式评测 |
| `stack_blocks_two` | `demo_clean`, `50` demos | `100` | 用 `4 x 25` 并行汇总 |

评测 instruction 类型均为 `unseen`。

## 6. 复现步骤

这一节只保留能直接复现实验的最小必要信息。

### 6.1 数据与 checkpoint 路径规则

训练数据路径：

```bash
policy/DP/data/{task_name}-{task_config}-{expert_data_num}.zarr
```

checkpoint 路径：

```bash
policy/DP/checkpoints/{task_name}-{ckpt_setting}-{expert_data_num}-{seed}/{checkpoint_num}.ckpt
```

评测结果路径：

```bash
eval_result/{task_name}/DP/{task_config}/{ckpt_setting}/{timestamp}_{seed}_{pid}/_result.txt
```

### 6.2 训练命令模板

```bash
cd /data/wentao/RoboTwin/policy/DP
bash train.sh <task_name> demo_clean 50 <seed> 14 <gpu_id> <ckpt_setting> <config_name>
```

字段含义：

- `<task_name>`：`adjust_bottle` 或 `stack_blocks_two`
- `<seed>`：训练 seed
- `<gpu_id>`：单卡训练所用 GPU
- `<ckpt_setting>`：checkpoint 目录名
- `<config_name>`：Hydra 配置名

### 6.3 本轮实际训练命令

#### 6.3.1 `adjust_bottle`

ResNet18 baseline：

```bash
cd /data/wentao/RoboTwin/policy/DP
bash train.sh adjust_bottle demo_clean 50 0 14 0 demo_clean robot_dp_14
```

DINOv3 `adapter224`：

```bash
cd /data/wentao/RoboTwin/policy/DP
bash train.sh adjust_bottle demo_clean 50 0 14 0 \
  demo_clean_dinov3_vits16_adapter224 \
  robot_dp_14_dinov3_vits16_adapter224
```

DINOv3 `unfreeze320`：

```bash
cd /data/wentao/RoboTwin/policy/DP
bash train.sh adjust_bottle demo_clean 50 0 14 0 \
  demo_clean_dinov3_vits16_unfreeze320 \
  robot_dp_14_dinov3_vits16_unfreeze320
```

DINOv3 `lora320`：

```bash
cd /data/wentao/RoboTwin/policy/DP
bash train.sh adjust_bottle demo_clean 50 0 14 0 \
  demo_clean_dinov3_vits16_lora320 \
  robot_dp_14_dinov3_vits16_lora320
```

#### 6.3.2 `stack_blocks_two`

ResNet18 baseline：

```bash
cd /data/wentao/RoboTwin/policy/DP
bash train.sh stack_blocks_two demo_clean 50 0 14 0 demo_clean robot_dp_14
```

DINOv3 `adapter224`：

```bash
cd /data/wentao/RoboTwin/policy/DP
bash train.sh stack_blocks_two demo_clean 50 0 14 0 \
  demo_clean_dinov3_vits16_adapter224 \
  robot_dp_14_dinov3_vits16_adapter224
```

DINOv3 `unfreeze320`：

```bash
cd /data/wentao/RoboTwin/policy/DP
bash train.sh stack_blocks_two demo_clean 50 0 14 0 \
  demo_clean_dinov3_vits16_unfreeze320 \
  robot_dp_14_dinov3_vits16_unfreeze320
```

DINOv3 `lora320`：

```bash
cd /data/wentao/RoboTwin/policy/DP
bash train.sh stack_blocks_two demo_clean 50 0 14 0 \
  demo_clean_dinov3_vits16_lora320 \
  robot_dp_14_dinov3_vits16_lora320
```

### 6.4 评测命令模板

正式评测前，需要在 `policy/DP/deploy_policy.yml` 中将 `checkpoint_num` 设为目标轮次，或者在命令行里覆盖：

```bash
--checkpoint_num 300
```

标准评测模板：

```bash
cd /data/wentao/RoboTwin/policy/DP
bash eval.sh <task_name> demo_clean <ckpt_setting> 50 <seed> <gpu_id>
```

### 6.5 本轮实际评测命令

#### 6.5.1 `adjust_bottle`

`adapter224` 使用 `600.ckpt`：

```bash
cd /data/wentao/RoboTwin
CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/DP/deploy_policy.yml --overrides \
  --task_name adjust_bottle \
  --task_config demo_clean \
  --ckpt_setting demo_clean_dinov3_vits16_adapter224 \
  --expert_data_num 50 \
  --seed 0 \
  --checkpoint_num 600
```

`unfreeze320` 与 `lora320` 使用 `300.ckpt`：

```bash
cd /data/wentao/RoboTwin
CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/DP/deploy_policy.yml --overrides \
  --task_name adjust_bottle \
  --task_config demo_clean \
  --ckpt_setting demo_clean_dinov3_vits16_unfreeze320 \
  --expert_data_num 50 \
  --seed 0 \
  --checkpoint_num 300
```

```bash
cd /data/wentao/RoboTwin
CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/DP/deploy_policy.yml --overrides \
  --task_name adjust_bottle \
  --task_config demo_clean \
  --ckpt_setting demo_clean_dinov3_vits16_lora320 \
  --expert_data_num 50 \
  --seed 0 \
  --checkpoint_num 300
```

#### 6.5.2 `stack_blocks_two`

为了控制总时间，本轮实际采用 `4 x 25` 并行评测，而不是单进程 `100` 次串行评测。  
这一点需要直接调用 `script/eval_policy.py`，显式传 `--test_num 25`。

`adapter224`：

```bash
cd /data/wentao/RoboTwin
for seed in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$seed PYTHONWARNINGS=ignore::UserWarning \
  python script/eval_policy.py --config policy/DP/deploy_policy.yml --overrides \
    --task_name stack_blocks_two \
    --task_config demo_clean \
    --ckpt_setting demo_clean_dinov3_vits16_adapter224 \
    --expert_data_num 50 \
    --checkpoint_num 300 \
    --seed $seed \
    --test_num 25
done
```

`unfreeze320`：

```bash
cd /data/wentao/RoboTwin
for seed in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$seed PYTHONWARNINGS=ignore::UserWarning \
  python script/eval_policy.py --config policy/DP/deploy_policy.yml --overrides \
    --task_name stack_blocks_two \
    --task_config demo_clean \
    --ckpt_setting demo_clean_dinov3_vits16_unfreeze320 \
    --expert_data_num 50 \
    --checkpoint_num 300 \
    --seed $seed \
    --test_num 25
done
```

`lora320`：

```bash
cd /data/wentao/RoboTwin
for seed in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$seed PYTHONWARNINGS=ignore::UserWarning \
  python script/eval_policy.py --config policy/DP/deploy_policy.yml --overrides \
    --task_name stack_blocks_two \
    --task_config demo_clean \
    --ckpt_setting demo_clean_dinov3_vits16_lora320 \
    --expert_data_num 50 \
    --checkpoint_num 300 \
    --seed $seed \
    --test_num 25
done
```

最终成功率按四个并行结果目录中的 `_result.txt` 末行恢复。若第 `i` 个目录末行记为 `r_i`，则其成功次数为 `25 r_i`，总成功次数为：

$$
\sum_{i=1}^{4} 25 r_i.
$$

## 7. 实验结果

### 7.1 主结果

| 任务                 | 既有基线口径  | DINOv3 `adapter224` | DINOv3 `unfreeze320` | DINOv3 `lora320` |
| ------------------ | ------- | ------------------- | -------------------- | ---------------- |
| `adjust_bottle`    | 约 `98%` | `64.0%`             | `100.0%`             | `100.0%`         |
| `stack_blocks_two` | 约 `7%`  | `19.0%`             | `47.0%`              | `42.0%`          |

### 7.2 `adjust_bottle`

| 配置            | checkpoint | 成功率      | 结果来源          |
| ------------- | ---------- | -------- | ------------- |
| `adapter224`  | `600.ckpt` | `64.0%`  | `_result.txt` |
| `unfreeze320` | `300.ckpt` | `100.0%` | `_result.txt` |
| `lora320`     | `300.ckpt` | `100.0%` | `_result.txt` |

已核对的结果文件：

- `eval_result/adjust_bottle/DP/demo_clean/demo_clean_dinov3_vits16_adapter224/2026-03-13 19:30:50/_result.txt`
- `eval_result/adjust_bottle/DP/demo_clean/demo_clean_dinov3_vits16_unfreeze320/2026-03-15 01:28:22/_result.txt`
- `eval_result/adjust_bottle/DP/demo_clean/demo_clean_dinov3_vits16_lora320/2026-03-15 02:03:22/_result.txt`

### 7.3 `stack_blocks_two`

| 配置 | checkpoint | 100 次成功率 | 汇总细节 |
| --- | --- | --- | --- |
| `adapter224` | `300.ckpt` | `19.0%` | `2/25 + 6/25 + 7/25 + 4/25` |
| `unfreeze320` | `300.ckpt` | `47.0%` | `8/25 + 14/25 + 10/25 + 15/25` |
| `lora320` | `300.ckpt` | `42.0%` | `9/25 + 11/25 + 12/25 + 10/25` |

三组配置的 `100` 次统计值都来自修复后的唯一结果目录。每个目录各执行 `25` 次 rollout，并以 `_result.txt` 末行比例值恢复成功次数后求和。

### 7.4 结果解释

从实验结果可以直接读出四点：

- 冻结版 DINOv3 能工作，但视觉能力还没有充分转化为控制能力。
- 即便完全冻结 backbone，`adapter224` 在 `stack_blocks_two` 上也已经达到 `19.0%`，说明 DINOv3 语义特征本身开始发挥作用。
- 真正有效的是“更高分辨率 + 更适合控制的 adapter + 轻量微调”的组合。
- 在 harder task `stack_blocks_two` 上，收益呈现清晰阶梯：约 `7% -> 19% -> 42%~47%`，因此这轮实验的核心价值在于验证“更强视觉 backbone + 合理适配 + 轻量微调”确实能持续推高高难任务成功率。

## 8. 成本分析

### 8.1 历史真实墙钟时间

该口径取训练目录时间戳作为开始时间，checkpoint 文件修改时间作为结束时间。

| 任务                 | 配置                | checkpoint | 观察到的墙钟时间 |
| ------------------ | ----------------- | ---------- | -------- |
| `adjust_bottle`    | ResNet18 baseline | `300.ckpt` | `4h 43m` |
| `adjust_bottle`    | ResNet18 baseline | `600.ckpt` | `6h 13m` |
| `adjust_bottle`    | `adapter224`      | `300.ckpt` | `1h 03m` |
| `adjust_bottle`    | `adapter224`      | `600.ckpt` | `2h 04m` |
| `adjust_bottle`    | `unfreeze320`     | `300.ckpt` | `2h 24m` |
| `stack_blocks_two` | ResNet18 baseline | `300.ckpt` | `6h 23m` |
| `stack_blocks_two` | ResNet18 baseline | `600.ckpt` | `9h 39m` |
| `stack_blocks_two` | `adapter224`      | `300.ckpt` | `2h 20m` |
| `stack_blocks_two` | `unfreeze320`     | `300.ckpt` | `5h 13m` |
| `stack_blocks_two` | `lora320`         | `300.ckpt` | `6h 14m` |

这组数字反映“真实训练体验”，但不是严格受控 benchmark。

### 8.2 单步训练 benchmark

该口径在当前机器上重新执行真实训练 step，并统计：

- batch 读取与 `postprocess`
- forward
- backward
- `optimizer.step()`
- `ema.step()`
- `torch.cuda.max_memory_allocated()`

#### 8.2.1 实际训练配置口径

| 配置                | batch size | step 时间  | steps / epoch | 估算 `300` epoch 纯训练时间 | 峰值显存 `allocated` | 吞吐                |
| ----------------- | ---------- | -------- | ------------- | -------------------- | ---------------- | ----------------- |
| ResNet18 baseline | `128`      | `0.419s` | `55`          | `1.92h`              | `15.84 GiB`      | `305.7 samples/s` |
| `adapter224`      | `128`      | `0.232s` | `55`          | `1.06h`              | `5.35 GiB`       | `552.4 samples/s` |
| `unfreeze320`     | `64`       | `0.275s` | `110`         | `2.52h`              | `7.48 GiB`       | `232.4 samples/s` |
| `lora320`         | `64`       | `0.333s` | `110`         | `3.05h`              | `8.79 GiB`       | `192.1 samples/s` |

这一组数据说明：

- `adapter224` 相比 ResNet18 更快、更省显存
- `unfreeze320` 的主要代价是 step 变慢和 batch 减半
- `lora320` 是当前成本最高的配置

#### 8.2.2 统一 `batch_size = 64` 的受控对比

| 配置 | 总参数量 | 可训练参数量 | step 时间 | 峰值显存 `allocated` | 吞吐 |
| --- | --- | --- | --- | --- | --- |
| ResNet18 baseline | `96.80M` | `96.80M` | `0.223s` | `8.67 GiB` | `287.0 samples/s` |
| `adapter224` | `107.83M` | `86.22M` | `0.124s` | `3.41 GiB` | `515.6 samples/s` |
| `unfreeze320` | `108.09M` | `90.04M` | `0.275s` | `7.48 GiB` | `232.4 samples/s` |
| `lora320` | `108.16M` | `86.56M` | `0.333s` | `8.79 GiB` | `192.1 samples/s` |

这里最重要的结论是：

- DINOv3 替换 ResNet18 不等于“显存暴涨”
- 冻结版 DINOv3 实际上更轻
- 真正的代价来自 `320` 分辨率和微调后的更慢计算

## 9. 复现注意事项

### 9.1 `stack_blocks_two` 并行评测目录覆盖

历史版本的 `script/eval_policy.py` 保存目录只精确到秒。如果多个并行评测进程在同一秒启动，会写到同一个目录里，导致 `_result.txt` 被覆盖。

这个问题已在本轮修复。当前目录名由时间戳微秒、`seed` 与 `pid` 共同构成：

$$
\text{run\_id} = \text{timestamp}_{\mu s} \Vert \text{seed} \Vert \text{pid}.
$$

因此并行评测不会再互相覆盖。本报告中的 `stack_blocks_two` 最终 `100` 次结果基于修复后的重新评测：

- `adapter224`: `0.08 + 0.24 + 0.28 + 0.16 = 19/100`
- `unfreeze320`: `0.32 + 0.56 + 0.40 + 0.60 = 47/100`
- `lora320`: `0.36 + 0.44 + 0.48 + 0.40 = 42/100`

### 9.2 `checkpoint_num` 必须和评测结果一致

本轮主结果对应：

- `adjust_bottle`
  `adapter224 = 600.ckpt`，`unfreeze320 = 300.ckpt`，`lora320 = 300.ckpt`
- `stack_blocks_two`
  `adapter224 = 300.ckpt`，`unfreeze320 = 300.ckpt`，`lora320 = 300.ckpt`

如果评测时仍使用 `deploy_policy.yml` 默认的 `600`，会得到错误结果。

### 9.3 不要重复做 ImageNet 归一化

DINOv3 的均值方差归一化已经在 `DINOv3ImagePreprocessor` 中实现。  
因此在 `MultiImageObsEncoder` 里必须保持：

```yaml
imagenet_norm: False
```

否则会发生 double normalization。

### 9.4 多相机复现时的设置

如果后续扩展到多相机，需要同时满足：

- `share_rgb_model: True`
- 观测字典里包含多个 `rgb` key
- 共享一个 DINO backbone
- 每个相机由 `camera_key` 路由到各自 adapter

如果 `share_rgb_model` 改成 `False`，语义会变成每个相机各有一套完整视觉模型，不再是本报告实验的设置。

## 10. 最终建议

如果只考虑本轮实验，推荐默认配置是：

- backbone：`dinov3_vits16`
- 输入：`resize_long_edge = 320`
- pooling：`spatial_softmax`
- 层融合：最后 `4` 层 softmax 融合
- 微调：优先 `partial unfreeze`
- 解冻：最后 `2` 个 block + norm
- 优化器：`backbone_lr_scale = 0.1`

原因很直接：

- `adjust_bottle` 达到 `100.0%`
- `stack_blocks_two` 达到 `47.0%`
- 相比 `LoRA`，当前效果略高，训练成本也更低一些

如果优先级变成“尽量少改 backbone 权重”，则可以退而使用 `lora320`，但按当前实验结果，它不是最优默认项。

---

**注意：上述实验有一个BUG**

DP原代码会在输入dinov3前（训练和推理）会进行一次归一化到-1,1，破坏了dinov3 pretrained的预定义分布，所以才导致实验中需要对backbone进行微调来缓解此问题。当将此归一化操作改为恒等映射之后，完全冻结dinov3 backbone的实验取得了最好的结果。
