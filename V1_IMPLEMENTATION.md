# 频域交叉注意力红外SR：当前实现与训练说明

所有代码、日志、检查点、环境和缓存均位于本项目。网络结构与原bicubic退化公式未改。

## 默认正式实验

- train/IR：1000张；val/IR：200张独立验证图；不用RGB。
- 训练：HR随机128×128 → PyTorch抗混叠bicubic×4下采样 → LR/HR同步25%水平翻转 → 从(90,-90,180,-180)中等概率旋转。LR为32×32。±180等价，实际180度概率50%，无0度。
- batch=4，accumulation=1，200 epochs，50000次成功更新；不丢弃尾批。AMP fp16，loss计算FP32；梯度裁剪1。
- AdamW lr=2e-4，warmup=500成功更新，余弦至1e-6；EMA=0.999。当前损失L1=1、梯度=0、边缘=0，未额外引入新损失。
- 模型642837参数：浅层卷积、4个Haar频率交互块（HF为Q、LL为K/V）、长残差、两级PixelShuffle、bicubic图像残差。
- 每10 epoch与最终epoch使用EMA评价全部200张整图；每10 epoch覆盖last.pt，结束也保存。默认无需分块。
- best.pt：同次验证PSNR、SSIM的全验证集逐图均值都严格超过各自历史最高值才保存。首次有效验证建立best；任意一项平局或下降不更新。即使没有更新best，两项独立历史最高值仍更新。训练验证不计算BRISQUE/NIQE/PIQE，后续test.py加--with-nr才计算。
- NR指标整图、不裁边；PSNR/SSIM裁4像素边。SR先clamp到[0,1]，直接浮点评价，PNG仅用于显示；NIQE后端自身的取整等内部处理按pyiqa默认实现。显示拼图从左到右为GT、bicubic、SR、绝对误差×4。
- best只由EMA的PSNR和SSIM决定；无参考指标用于训练后的补充评价。训练和独立评价共用utils/psnr_ssim_calculate.py，PSNR按每图MSE计算后取算术平均，SSIM使用11x11、sigma=1.5的高斯窗口有效区域。精确匹配沿用原实现的有限160dB上限。

## 文件职责

| 文件 | 作用 |
|---|---|
| config.py | 路径、网络、增强、损失及训练手工参数 |
| dataset.py | 固定量程读取、裁剪、同步增强；训练器传入种子后按epoch和样本索引确定增强 |
| scripts/prepare_data.py | 仅bicubic在线退化 |
| network.py | 频率交互网络，保持原结构 |
| utils/training_components.py | AdamW、warmup/cosine、EMA |
| utils/runtime.py | DataLoader、AMP更新、严格best选择、完整检查点及恢复 |
| utils/psnr_ssim_calculate.py | 独立PSNR/SSIM计算、固定[0,1]量程、默认裁4像素边 |
| utils/evaluation.py | 训练/独立推理共享整图评价、逐图CSV、汇总JSON与PNG |
| utils/brisque_niqe_piqe_calculate.py | pyiqa三项NR指标，文件/张量接口 |
| train.py | 训练、验证、日志及保存调度 |
| test.py | 加载EMA，对HR在线生成LR后进行全图重建评价 |
| tests/test_training_runtime.py | CUDA精确恢复、AMP溢出跳步、PSNR/SSIM严格历史最优规则测试 |

检查点含原模型、EMA、AdamW、scheduler、GradScaler、成功/跳过计数、epoch/样本位置、Python/NumPy/Torch/CUDA RNG、best状态、配置、数据文件SHA256和源码SHA256。调度按成功更新推进，溢出重试同一组数据。恢复拒绝配置、数据、源码或关键依赖变化；只加载可信本项目检查点。

## 命令（PowerShell）

```powershell
cd D:\IRSR_model\My_IRSR_model_new
# 首次正式启动；如果已由助手启动，不要重复执行
.\.venv\Scripts\python.exe -B train.py

# 停止后从最近的last继续（不要同时运行两个进程）
.\.venv\Scripts\python.exe -B train.py --resume checkpoints/frequency_v1_psnr_ssim/last.pt

# 全部200张验证图，EMA全图推理与逐图导出
.\.venv\Scripts\python.exe -B test.py --checkpoint checkpoints/frequency_v1_psnr_ssim/best.pt --output results/frequency_v1_psnr_ssim/final_best

# 后续追加无参考评价（不会更改best）
.\.venv\Scripts\python.exe -B test.py --checkpoint checkpoints/frequency_v1_psnr_ssim/best.pt --output results/frequency_v1_psnr_ssim/post_nr --with-nr

# 其他独立HR集：增加 --input 路径；这里输入的是HR，不是已经降采样的LR
# 短程测试使用独立run-name，避免覆盖已有检查点
.\.venv\Scripts\python.exe -B train.py --smoke --run-name my_check
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v

# 查看后台正式训练控制台日志
Get-Content results/frequency_v1_psnr_ssim/console.log -Tail 20 -Wait
```

环境说明：新建项目.venv隔离安装了pyiqa依赖，通过previous_runtime.pth只读复用旧项目.venv中的CUDA Torch等已有包，因此本机不要删除旧环境。安装没有修改旧环境；requirements-lock.txt记录本次依赖版本。迁移机器需重新安装CUDA版Torch等依赖，不能直接搬用Windows虚拟环境。

## 验证与限制

16项单元/集成测试通过；真实8张训练、2张整图验证的两epoch端到端通过；独立推理复算与训练评价完全一致；CUDA FP16中途恢复与连续训练参数、EMA、优化器和损失逐项一致。短程结果仅验证流程，不能证明模型质量。

1200张均可解码且非常量，训练/验证未发现相同文件内容。这不证明场景层面完全独立；相邻视频帧泄漏仍须按数据来源核实。

已有旧EDSR/旧模型使用过MATLAB风格bicubic，当前仍为新项目原有PyTorch抗混叠bicubic。不同退化、增强、EMA及训练预算的结果不能直接归因于网络优劣。

第一次正式best/last常规保存将在epoch10产生。自然崩溃最多损失最近10epoch进度；--max-updates用于计划性停止，会额外保存last且不改变完整学习率预算。evaluation_failed.pt可用于重新执行失败验证。
