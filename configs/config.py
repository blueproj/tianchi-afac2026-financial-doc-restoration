"""
AFAC2026 赛题二 - 全局配置文件（主办方复现版）

说明：
- 本工程唯一外部依赖的模型服务为官方 FinixDoc-VL API（见下方 API 配置）。
- output/ logs/ cache/ 均落在本工程目录内，自动创建。
- 评测数据路径通过运行脚本的 --table-dir / --long-dir 命令行参数传入
  （见 README.md），下方 TEST_A_* 仅为未传参时的默认值占位。
"""
import os

# ==================== 路径配置 ====================
# 工程根目录（= submit_code/）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据目录（默认值占位；实际评测请用 --table-dir/--long-dir 显式传入图片目录）
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "AFAC 训练数据集")
TEST_A_DIR = os.path.join(DATA_DIR, "AFAC A榜评测数据集(2)")

# 训练集路径（仅个别模块的 __main__ 自测使用，主链路不依赖）
TRAIN_LONG_DIR = os.path.join(TRAIN_DIR, "finixdocbench_huge_long_100")
TRAIN_TABLE_DIR = os.path.join(TRAIN_DIR, "finixdocbench_huge_table_100")

# 测试集默认路径（未传 --table-dir/--long-dir 时使用）
TEST_A_LONG_DIR = os.path.join(TEST_A_DIR, "finix_huge_long_rest_A")
TEST_A_TABLE_DIR = os.path.join(TEST_A_DIR, "finix_huge_table_rest_A")

# 输出目录（自动创建，均在本工程内）
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

# ==================== API配置 ====================
API_URL = "https://finixdocapi.alipay.com/api/finix_doc/call_with_file"
API_KEY = "F935A5503983FB19F26FA3F00A94EBF9"

# 可用的userId白名单（5个，可用于并行调用）
USER_IDS = [
    "finixA1001",
    "finixB2002",
    "finixC3003",
    "finixD4004",
    "finixE5005",
]

# API调用参数
API_TIMEOUT = None         # 单次请求超时（秒）；None=无限等待，优先保证接口完整返回
API_MAX_RETRIES = 5        # 最大重试次数- 增加以应对不稳定
API_RETRY_DELAY = 10       # 重试初始间隔（秒），实际用指数退避
INTER_CALL_DELAY = 0.5     # 正常调用间隔（秒），避免触发限流
MAX_API_CONCURRENT = 16    # 官方建议并发控制上限

# ==================== 图像切块配置 ====================
# API建议的切片尺寸上限（长文档用）
MAX_CHUNK_WIDTH = 1024
MAX_CHUNK_HEIGHT = 3000     # 平衡API调用次数和识别质量，利用缓存加速

# 表格文档：API实际能处理更大图片，尽量不切或少切
# 小于此尺寸的表格图直接提交不切块
TABLE_DIRECT_SUBMIT_WIDTH = 8000
TABLE_DIRECT_SUBMIT_HEIGHT = 8000
# 超过直提尺寸时，用更大的切块（而非1024x1600）
TABLE_CHUNK_WIDTH = 4000
TABLE_CHUNK_HEIGHT = 4000

# 表格文档按高度切条——解决超密集大表被 FinixDoc 输出上限截断
TABLE_STRIP_HEIGHT = 500       # 切条基准高度；自适应/CV策略会按行密度再细分
TABLE_STRIP_OVERLAP = 40       # 条间小重叠（像素），避免行被切断
ENABLE_TABLE_STRIP = True      # 是否启用表格切条（False 回退旧直提/大切块）
TABLE_MAX_ROWS_PER_STRIP = 8   # 自适应行边界切条上限，避免密集表一次返回过少行
TABLE_LINE_PADDING = 8         # CV行边界裁切时保留上下安全边距
TABLE_ROW_GAP_DARK_RATIO = 0.03   # 暗像素直方图：暗像素占比<此值认为是行间隔
TABLE_ROW_GAP_MIN_HEIGHT = 6      # 暗像素直方图：连续行间隔最少行数才认定为切点
TABLE_BLANK_CHUNK_THRESHOLD = 0.005  # 暗像素占比<此值认为是空白chunk，跳过不发API
ENABLE_TABLE_COLUMN_SPLIT = False  # 左右切分关闭（合并质量不够）
TABLE_COLUMN_SPLIT_MIN_WIDTH = 2400  # 保留参数但不生效
TABLE_COLUMN_SPLIT_MIN_LINES = 8
TABLE_MAX_COLS_PER_SEGMENT = 20     # 保留参数但不生效
TABLE_VLINE_KERNEL_RATIO = 0.3      # 竖线检测核高度占子图高度的比例

# 自适应行数控制（基于API输出token上限）
API_MAX_OUTPUT_CHARS = 2000         # API单次输出上限（字符数）
CHARS_PER_CELL = 20                 # 每个单元格平均输出字符数（内容+标记）

# 切块重叠比例（占块高度的百分比）
OVERLAP_RATIO = 0.15

# 投影直方图参数：判断一行/列为"空白"的阈值（像素均值大于此值认为有内容）
BLANK_THRESHOLD = 250  # 0-255，越大越严格

# ==================== 图像预处理配置 ====================
ENABLE_CONTENT_CROP = True       # 长文档/表格均启用左右白边裁剪
CROP_PROJECTION_THRESHOLD = 248  # 列均值低于该值认为存在有效内容
CROP_MIN_CONTENT_RATIO = 0.002   # 单列暗像素比例阈值，防止淡线被均值淹没
CROP_PADDING = 24                # 裁剪时保留的安全边距
ENABLE_LONG_RESIZE = False       # 长文档默认不缩放，避免标题/小字识别下降
LONG_RESIZE_SCALE = 0.5          # 实验开关：需要时长文档整体缩放比例

# ==================== 运行配置 ====================
# 并行API调用的最大线程数
MAX_WORKERS = 5

# 是否启用API结果缓存（按图片内容MD5缓存，重跑加速；首次复现为全新调用）
ENABLE_CACHE = True

# 确保目录存在
for d in [OUTPUT_DIR, LOG_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)
