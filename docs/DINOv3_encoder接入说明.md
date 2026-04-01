# DINOv3 encoder 接入说明

## 改动概览

- 新增 `diffusion_policy/model/vision/dinov3_encoder.py`
  - 封装 DINOv3 `vits16` backbone
  - 内置长边缩放、patch 对齐 padding、ImageNet 归一化
  - 使用最后 `4` 层 patch feature 做 softmax 融合
  - 通过轻量 adapter + `spatial_softmax` 输出 `512` 维特征
  - adapter 支持通过 `use_residual_gating` 控制是否启用残差门控
- 新增 encoder 配置组：
  - `diffusion_policy/config/encoder/resnet18.yaml`
  - `diffusion_policy/config/encoder/dinov3_vits16.yaml`
- `robot_dp_16.yaml` 改为从 `encoder` 配置组读取视觉编码器
- `RobotImageDataset` 新增 `image_normalizer` 配置，支持：
  - `range`: 图像归一化到 `[-1, 1]`
  - `identity`: 图像保持在 `[0, 1]`

## 如何切换到 DINOv3

默认配置仍然是 `ResNet18`。

使用 DINOv3 时，在配置里选择：

```yaml
defaults:
  - _self_
  - task: default_task_16
  - encoder: dinov3_vits16
```

或直接在命令行覆盖：

```bash
python train.py --config-name=robot_dp_16 encoder=dinov3_vits16
```

## 双重归一化修复

DINOv3 这条链路明确避开了双重归一化：

- 数据集侧使用 `task.dataset.image_normalizer=identity`
  - 图像保持 `[0,1]`
- `MultiImageObsEncoder` 侧使用 `imagenet_norm=false`
  - 不再额外做一层 `torchvision.Normalize`
- DINOv3 encoder 内部自行完成：
  - resize
  - pad 到 patch size 的整数倍
  - ImageNet mean/std 归一化

也就是说，DINOv3 只在 encoder 内部做一次真正需要的视觉归一化。

## 依赖与权重

- DINOv3 仓库位于 `third_party/dinov3`
- 默认权重路径：

```text
data/pretrained/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
```

如果拉下代码后子模块未初始化，执行：

```bash
git submodule update --init --recursive
```

## 说明

这次实现只接入了冻结 backbone 的 DINOv3 encoder，没有顺手改训练器参数分组，也没有把 LoRA / partial unfreeze 一起带进来，目的是先把当前项目里的基础接入做稳、做干净。
