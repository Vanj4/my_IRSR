"""集中配置：修改这里即可更改项目默认参数。
导入本文件不读数据、不创建目录、不启动GPU或训练。
训练、验证和推理统一读取本配置。
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT.parent / "dataset"
TRAIN_ROOT = DATA_ROOT / "train" / "IR"
VALIDATION_ROOT = DATA_ROOT / "val" / "IR"  # 独立验证集，仅使用IR
TEST_ROOT = None        # 请指定实际测试集，不猜测目录
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"
RESULT_ROOT = PROJECT_ROOT / "results"

# 数据：先裁HR，再退化，最后LR/HR同步增强。
SCALE = 4              # 当前V1网络仅支持x4
HR_PATCH_SIZE = 256
RECURSIVE = False
DATA_RANGE = None       # 自动按位深；12/14位数据可显式设置4095/16383
RANDOM_CROP = True
TRAIN_AUGMENT = True
HORIZONTAL_FLIP_PROBABILITY = 0.25
ROTATION_ANGLES = (90, -90, 180, -180)  # 实际180度概率50%，没有0度
VALIDATION_PATCH_SIZE = None             # None为整图；PSNR/SSIM在整图裁边后计算
TEST_PATCH_SIZE = None

# 网络：原生LR输入，单层Haar，低频向高频提供上下文。
CHANNELS = 96
NUM_BLOCKS = 4
ATTENTION_HEADS = 4
WINDOW_SIZE = 4         # 小波子带上的窗口边长
CROSS_ALPHA_INIT = 0.1
USE_RGB_GUIDANCE = True
RGB_GATE_INIT = 0.01    # 每块三个方向的可学习门控初始值
RGB_GUIDE_TILE = 256    # 仅推理时RGB局部编码的HR核心大小；0关闭分块

# 上采样前空间Transformer：2组×2层，0组关闭；频域交互仍为原4块。
SPATIAL_GROUPS = 2
SPATIAL_DEPTH = 2
SPATIAL_HEADS = 4
SPATIAL_WINDOW = 8
SPATIAL_MLP_RATIO = 2.0
SPATIAL_RESIDUAL_SCALE = 0.1
SPATIAL_CHECKPOINT = True   # 训练重计算降低显存；不改变注意力定义
SPATIAL_EVAL_CHUNK = 128    # 推理按独立窗口分批，不是图像分块

# 优化组件：显式实参优先，否则使用下述默认值。
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
ADAM_BETAS = (0.9, 0.999)
TOTAL_UPDATES = 50000
WARMUP_UPDATES = 500
MIN_LEARNING_RATE = 1e-6
EMA_DECAY = 0.999
L1_WEIGHT = 1.0
GRADIENT_WEIGHT = 0.0
EDGE_WEIGHT = 0.0

# 训练入口参数。TOTAL_UPDATES仅为组件单测默认；正式预算按数据量和EPOCHS计算。
DEVICE = "cuda:0"
SEED = 2026
BATCH_SIZE = 1
ACCUMULATION_STEPS = 4
NUM_WORKERS = 0
AMP = "fp16"
GRAD_CLIP = 1.0
PRINT_EVERY = 50
EPOCHS = 200
VALIDATE_EVERY_EPOCHS = 10
SAVE_EVERY_EPOCHS = 10
EVAL_BORDER = 4
IQA_DEVICE = "cpu"
CPU_THREADS = 4
VISUALIZE_COUNT = 4
MAX_OVERFLOW_RETRIES = 20
RUN_NAME = "rgb96_f4_spatial2x2_hr256_v1"
