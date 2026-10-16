# RGB高频引导V1：实现与运行

## 信息流

输入lr [B,1,64,64]，rgb [B,3,256,256]。RGB使用不抽取Haar，丢弃LL，三个高频拼成9通道；原两层3×3+GELU及1×1残差投影保持不变，输出96×256×256。接三组Conv3×3(stride=2,padding=1)+GELU，得到96×128×128、96×64×64、96×32×32。

RGB特征每张图只编码一次，供四个频率块共用。每块先完成原IR LL->HF交叉注意力，再由IR三个HF作为Q、RGB高频特征作为K/V，使用独立参数的4头4×4窗口注意力。三个方向各有一个可学习gamma，初始0.01，共12个门控。LL不直接接收RGB注入。RGB输入必须和LR同批次、同设备，宽高为LR的4倍；不隐式resize。处理非窗口整除尺寸时两路按对应物理范围补边。

原IR各层先初始化，再创建新增模块，保证同种子下原IR初始化与纯IR模式相同。总参数3,053,617，纯IR 2,561,301；新增492,316参数。关闭引导不是测试时把RGB归零，而是实例化不含新增分支的独立控制模型。

## 改动文件

- network.py：独立RGB->HF注意力、方向门控、双输入forward、两路补边；原IR子路径保留。
- rgb_high_frequency_encoder.py：添加RGBGuidanceEncoder及三层步长2压缩。
- config.py：96通道4块，RGB开关，gamma初值0.01，RGB编码推理tile256；batch1 accumulation4。
- dataset.py：抽取rgb_path_for供读取和指纹共用；原裁剪、增强不变。
- utils/runtime.py：训练传RGB，数据指纹记录RGB文件路径和哈希。
- utils/evaluation.py：EMA整图验证传RGB，评价协议记录RGB开关及编码tile。
- train.py：RGB开关CLI、RGB跨划分重复检查、新模块源码哈希。
- test.py：从检查点恢复RGB架构配置，验证新模块源码并加载EMA。

损失保持L1，AdamW/warmup/cosine/EMA及best规则不变。全部旧权重与旧结果保留，Compare_model未修改。修改前源码位于results/rgb_guidance_v1_checks/source_before_integration。新架构必须开启新实验；旧纯IR检查点不能当作新模型完整断点恢复，未实施权重热启动。

## 低显存整图验证

只有局部RGB编码器在eval模式分块：默认HR核心256、halo16，并保持8像素采样相位对齐。其依赖偏移[-9,+10]，halo足够。裁取每块核心输出后拼接，不做图像缩放或加权混合。红外主干和窗口注意力仍对完整图像执行。CPU及4050 FP32对照检查误差为0（特定测试输入），数学运算等价，其他设备数值舍入可能不同。此方式不修正原数据可能存在的跨模态配准误差。

## 验证

unit_verification.json：同种子IR权重一致、全部参数/输入RGB梯度有限、RGB改变输出、门控归零精确回退、关闭引导匹配原架构、局部编码等价、完整CPU恢复后的下一步模型/EMA完全一致、RGB文件进入指纹。

gpu_tile_verification.json：4050 FP32局部编码整图/分块一致。

checkpoints/rgb_guidance_v1_final__smoke：8训练/2验证图、4次成功更新，先训练1步保存后恢复到4步；FP16 AMP运行成功。results/rgb_guidance_v1_checks/test_export：EMA独立验证入口、PSNR/SSIM及三个无参考指标和PNG导出通过。smoke只验证管线，不代表重建质量。

## PowerShell指令

```powershell
Set-Location D:\IRSR_model\My_IRSR_model_new
$py = '.\.venv\Scripts\python.exe'

# 正式RGB引导实验：200epoch，50000次成功更新，从头训练
& $py -B .\train.py --rgb-guidance --channels 96 --num-blocks 4 --hr-patch-size 256 --run-name rgb_guided96_blocks4_hr256_v1

# 同一新实验断点续训
& $py -B .\train.py --rgb-guidance --channels 96 --num-blocks 4 --hr-patch-size 256 --run-name rgb_guided96_blocks4_hr256_v1 --resume .\checkpoints\rgb_guided96_blocks4_hr256_v1\last.pt

# 完成训练后的测试：自动使用test/IR同级test/RGB
& $py -B .\test.py --checkpoint .\checkpoints\rgb_guided96_blocks4_hr256_v1\best.pt --input D:\IRSR_model\dataset\test\IR --output .\results\rgb_guided96_blocks4_hr256_v1\test_best --with-nr

# 可选：相同新入口、微批次/累积设置下的独立纯IR控制（另一次训练）
& $py -B .\train.py --no-rgb-guidance --channels 96 --num-blocks 4 --hr-patch-size 256 --run-name ir_control96_blocks4_hr256_v1
```

一次只启动一个正式训练。模型成效用验证集选择方案，测试集用于固定方案后的最终报告。RGB引导方法使用额外HR RGB信息，和纯IR方法比较时需要明确这一输入差异。
