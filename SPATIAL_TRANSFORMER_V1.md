# 上采样前空间残差 Transformer 第一版

## 实现与设计边界
原生LR -> stem -> 4个频率交互块（RGB可选） -> 2个空间残差组 -> body_conv + stem长残差 -> 两级PixelShuffle -> SR残差 + bicubic。
训练HR256/LR64时，空间残差组始终为[B,96,64,64]；不是在256×256上做注意力。
每组2层：普通8×8窗口、移位4像素的8×8窗口；4头；预LayerNorm；相对位置偏置；MLP为96→192→96、GELU；每层注意力与MLP都有残差。
组输出为 x + 0.1*Conv3x3(两层输出)。0.1固定，非可学习近零门控。Q/K/V均来自当前融合后的空间特征。
无dropout、无BatchNorm，无额外损失或训练数据变化。该版本是已有空间建模组件的组合适配，不是已证明的论文创新，也不是HAT/PFT的复现。

依据：SwinIR的残差组/移位窗口 https://arxiv.org/abs/2108.10257 ; HAT的跨窗口信息交换 https://github.com/XPixelGroup/HAT ; HiT-SR的窗口上下文研究 https://arxiv.org/abs/2407.05878 ; PFT的注意力聚合研究 https://openaccess.thecvf.com/content/CVPR2025/html/Long_Progressive_Focused_Transformer_for_Single_Image_Super-Resolution_CVPR_2025_paper.html 。也检索了2026年的生成式SR研究，但其真实退化与感知生成目标不同于当前bicubic红外保真重建，不直接引入。
当前只实现普通/移位窗口，未实现HAT通道注意力/重叠交叉注意力、HiT分层大窗口或PFT跨层注意力传播。

## 修改文件
- spatial_transformer.py：独立空间模块、移位边界掩码、训练激活重计算、推理窗口批处理。
- network.py：频率块之后接模块；输入补边满足频率窗口与空间窗口的最小公倍数；新增层最后初始化。
- config.py：SPATIAL_*参数，新实验名。
- train.py：--spatial-groups，参数量日志，检查点源码指纹包含新模块。
- test.py：从检查点恢复全部空间结构参数并校验新模块源码。
- tests/check_spatial_transformer.py：定向回归测试。
数据、损失、优化器、训练预算、EMA和best准则保持原有配置。RGB开关独立。

## 已验证
结果见results/spatial_transformer_checks/checks.json。
关闭空间组与原版本（纯IR和RGB两种）初始权重、输出逐值一致；开启时原有层初始值不变。
窗口批处理与整批输出一致；移位注意力无循环边界串扰；重计算与普通计算输出、梯度一致。
奇数/小尺寸/非方形输入可恢复准确x4尺寸；新增层真实数据AMP梯度有限且非零。
完整优化器、EMA、scaler、调度器和随机状态恢复后下一步逐值一致。
真实1024×1280整图推理通过，测试进程峰值约1.20GiB。
训练入口8张训练/2张验证、2轮共4次成功更新已完成，中间保存退出再恢复；best、last、EMA全图指标和NR指标及图像导出通过。
短程分数不代表最终模型效果，没有启动200轮正式训练。

参数：新增468816；RGB总3522433，纯IR总3030117。
训练激活重计算use_reentrant=False；推理仅按独立注意力窗口分批128，仍是全图推理，不引入图像切块拼接。

## 实验指令（PowerShell，项目根目录）
```powershell
Set-Location D:\IRSR_model\My_IRSR_model_new
$py = '.\.venv\Scripts\python.exe'
# 先运行当前RGB架构 + 空间组，200epoch，其余与最近RGB实验一致。
& $py -B train.py --rgb-guidance --channels 96 --num-blocks 4 --hr-patch-size 256 --spatial-groups 2 --run-name rgb96_f4_spatial2x2_hr256_v1
# 此次新实验自己的断点续训
& $py -B train.py --rgb-guidance --channels 96 --num-blocks 4 --hr-patch-size 256 --spatial-groups 2 --run-name rgb96_f4_spatial2x2_hr256_v1 --resume checkpoints/rgb96_f4_spatial2x2_hr256_v1/last.pt
# 训练完成后测试集全图评价
& $py -B test.py --checkpoint checkpoints/rgb96_f4_spatial2x2_hr256_v1/best.pt --input D:\IRSR_model\dataset	est\IR --output results/rgb96_f4_spatial2x2_hr256_v1/test_best --with-nr
```
先与最近RGB实验的同协议验证/测试结果比较：val40.90452256/.98087549，test40.99579037/.98148978；最终同时检查逐图增益、轮廓ROI、参数和速度，不能只看最好的一张。
如做全套2×2消融，独立实验名分别运行RGB on/off × spatial-groups 0/2；全部从头同种子、同预算。已有旧纯IR实验微批量不同，严格对照应重跑同入口。空间增强有效不等于RGB变得有效，后续仍需错配/关闭RGB的内容依赖诊断。

## 原实验保护
原代码完整快照：_backups/before_spatial_transformer_20261014（Python入口和所需源码）。旧权重/评价未修改。
本次结构和源码改变，不能用旧权重--resume新结构；新实验从头训练。旧检查点应在对应旧源码快照下评价，不能关闭严格哈希校验强行混用。
