# 基于 PyTorch 的 DeepFM 推荐系统

> 从零实现 LR / FM / MLP / DeepFM，在 MovieLens-1M 上做 CTR 式推荐与排序评估。
>
> From-scratch PyTorch CTR models (LR, FM, MLP, DeepFM) for MovieLens-1M,
> with leave-one-out evaluation, negative sampling and a FastAPI serving layer.

![CI](https://github.com/202012770/ctr-deepfm-pytorch/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-EE4C2C)
![FastAPI](https://img.shields.io/badge/serving-FastAPI-009688)

---

## 核心结果

<!-- RESULTS_START -->
**部署模型（DeepFM + 用户/物品统计特征，10 轮）**

| 指标 | 数值 |
|---|---:|
| 测试 AUC | **0.9097** |
| HR@10 | **0.6928** |
| NDCG@10 | 0.3968 |
| 3 个随机种子 | AUC **0.9090 ± 0.0005**，HR@10 **0.6898 ± 0.0024** |
| HR@10 的 95% Bootstrap 置信区间 | **[0.6756, 0.6979]** |
| 冷启动用户（10% 完全不可见） | AUC 0.8317，HR@10 0.4785 |

**线上工程指标**（CPU，单进程）：

| 指标 | 数值 |
|---|---:|
| 推荐 P95（冷缓存，全量 3883 部候选） | 268 ms |
| 推荐 P95（命中用户缓存） | **1.36 ms** |
| 缓存加速比（P95） | **197×** |
| 可观测性 | 请求 ID + JSON 日志 + Prometheus `/metrics` |
| 漂移监控 | 请求画像 + Top-1 分数 PSI（同分布 max PSI 0.074；注入漂移 0.91 告警） |

**模型骨架对比**（未加统计特征，统一 15 轮预算，每个用户 1 正 + 99 负）：

| 模型 | 参数量 | 训练时间 | 验证 AUC | 测试 AUC | HR@10 | NDCG@10 | HR@20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **DeepFM** | 241,584 | 12.8 min | **0.9187** | **0.9080** | **0.6949** | **0.4030** | **0.8510** |
| MLP | 241,584 | 7.9 min | 0.9167 | 0.9061 | 0.6925 | 0.3922 | 0.8510 |
| FM | 181,935 | 4.8 min | 0.8994 | 0.8869 | 0.6330 | 0.3554 | 0.8051 |
| LR | 181,935 | 2.6 min | 0.8438 | 0.8316 | 0.4862 | 0.2686 | 0.6779 |

![Model comparison](reports/figures/model_comparison.png)

三点结论：

1. **二阶交叉是最大的单步收益**：LR → FM 将 AUC 从 0.832 提到 0.887，HR@10 从 0.486 提到 0.633。
2. **非线性塔再加一层**：FM → MLP 的 AUC 再 +1.9 个点，HR@10 +6.0 个点。
3. **DeepFM 的增益很小**：相对 MLP 仅 +0.2 个点 AUC，NDCG@10 +1.1 个点。说明在这套特征上，FM 与 MLP 的表达有较大重叠——这是可以坦诚讨论的结论，而不是硬吹 DeepFM 全面领先。

> 注：模型骨架对比使用 15 轮预算；部署模型使用统计特征 + 10 轮，两者训练预算不同，不宜逐位比较。参数量中含共享 Embedding 表。
<!-- RESULTS_END -->


---

## 项目做了什么

不使用现成推荐库，四个模型全部用 PyTorch 手写：

| 模型 | 结构 | 文件 |
|---|---|---|
| **LR** | 每个特征一个权重，线性打分 | `models/deepfm.py` |
| **FM** | 线性项 + 二阶特征交叉 `0.5·((Σe)² − Σe²)` | `models/layers.py` |
| **MLP** | 线性项 + Embedding 拼接后过 MLP | `models/deepfm.py` |
| **DeepFM** | 线性项 + FM + MLP，共享 Embedding | `models/deepfm.py` |

评估同时覆盖**排序**与**CTR**：

- `AUC` / `LogLoss`：候选集上的点击概率判别能力
- `HR@10 / 20`、`NDCG@10 / 20`：每个用户 1 个正样本 + 99 个负样本的排序质量

---

## 数据与评估协议

MovieLens-1M：6040 个用户、约 100 万条评分、3883 部电影。

1. **正样本**：评分 ≥ 4 的交互
2. **留一法切分**：每个用户按时间排序，最后一条正样本作测试集，倒数第二条作验证集
3. **训练负采样**：每条正样本配 2 个用户未评分的电影（可配置）
4. **验证/测试**：每个用户 1 个正样本 + 99 个未评分电影，计算排序与 CTR 指标
5. **特征**（30 个类别字段）：用户 ID、性别、年龄、职业、邮编前缀，以及用户活跃度/均分/偏好类型；电影 ID、年代、物品热度/均分；18 个类型标记
6. 未见过的类别 ID 映射到保留索引 0，支持冷启动

这一协议避免了两个常见错误：随机切分导致未来信息泄漏，以及用已交互物品当负样本。

---

## 消融实验

固定 DeepFM、10 轮训练预算，分别改变 Embedding 维度与训练负采样比例（`reports/ablation.csv`）：

| Embedding 维度 | 测试 AUC | HR@10 | NDCG@10 | 训练时间 |
|---:|---:|---:|---:|---:|
| 8 | 0.9006 | 0.6656 | 0.3767 | 6.8 min |
| **16** | **0.9065** | **0.6843** | **0.3926** | 8.2 min |
| 32 | 0.9074 | 0.6835 | 0.3924 | 8.8 min |

| 负采样比例 | 测试 AUC | HR@10 | NDCG@10 | 训练时间 |
|---:|---:|---:|---:|---:|
| 1 : 1 | 0.8923 | 0.6356 | 0.3574 | 4.0 min |
| 1 : 2 | 0.9065 | 0.6843 | 0.3926 | 8.2 min |
| **1 : 4** | **0.9109** | **0.6966** | **0.4098** | 11.2 min |

![Ablation](reports/figures/ablation.png)

- **8 维明显欠拟合**；16 → 32 维只多 0.09 个 AUC 点，却多 7% 训练时间，因此 16 维是性价比最优。
- **负样本越多，判别力越强**：1:2 → 1:4 再 +0.44 个 AUC 点、HR@10 +1.2 点。但训练时间接近翻倍，且训练分布会进一步偏离线上真实分布，不能无限加。

---

## 改进实验：稳定性、冷启动与负采样

### 1. 多种子与置信区间

同一配置跑 3 个随机种子（种子同时控制参数初始化与每轮样本打乱，已验证两次运行结果完全一致）：

| 种子 | 测试 AUC | HR@10 | NDCG@10 | 最佳轮次 |
|---:|---:|---:|---:|---:|
| 42 | 0.9097 | 0.6928 | 0.3968 | 10 |
| 2024 | 0.9084 | 0.6895 | 0.4003 | 9 |
| 7 | 0.9088 | 0.6870 | 0.3970 | 10 |
| **均值 ± 标准差** | **0.9090 ± 0.0005** | **0.6898 ± 0.0024** | **0.3980 ± 0.0016** | — |

- HR@10 的 95% Bootstrap 置信区间：**[0.6756, 0.6979]**（按用户重采样 1000 次）
- NDCG@10 的 95% CI：[0.3885, 0.4052]
- **结论**：AUC 的种子间波动只有 0.0005，说明结果很稳；HR@10 的波动稍大（0.0024），做模型对比时应以区间而非单点为准

**统计特征的效果**：同样的 DeepFM、10 轮、1:2 采样，不加统计特征时 AUC 0.9065 / HR@10 0.6843；加入用户活跃度、用户均分、偏好类型、物品热度、物品均分后，3 种子均值为 0.9090 / 0.6898。

- 增益约 **+0.25 AUC 点、+0.55 HR@10 点**，幅度不大但方向一致；这也符合预期——ID Embedding 已经能表达大部分用户/物品信息，统计特征主要帮助样本少的用户与物品

### 2. 冷启动

留出 **10% 用户（604 人）**，训练集中完全不出现他们的任何交互；模型只能看到性别、年龄、职业、邮编画像，用户 ID 的 Embedding 保持随机初始化。评估仍为 1 正 + 99 负：

| 用户类型 | AUC | HR@10 | NDCG@10 | HR@20 |
|---|---:|---:|---:|---:|
| 热启动（训练中出现过） | 0.9071 | 0.6855 | 0.3972 | 0.8499 |
| **冷启动（完全不可见）** | **0.8317** | **0.4785** | **0.2539** | 0.6904 |
| 差值 | −0.0754 | **−0.207** | −0.1433 | −0.1595 |

- 冷启动的 AUC 还能到 0.83，说明画像特征有信息量；但 **HR@10 掉了 20.7 个点**，排序质量明显退化
- 这个差距是推荐系统的常态，也是改进方向：冷启动用户应走独立的召回策略，而不是直接复用热启动的排序模型

### 3. 负采样方式

训练负样本从均匀采样改为按电影热度幂律采样（α = 0.75），评估协议保持均匀负采样：

| 训练负采样 | 测试 AUC | HR@10 | NDCG@10 |
|---|---:|---:|---:|
| **均匀采样** | **0.9097** | **0.6928** | **0.3968** |
| 热度采样 | 0.9006 | 0.6764 | 0.3954 |

- **热度负采样反而更差**（−0.91 AUC 点、−1.64 HR 点）。一个合理解释是：热门物品本身更容易被所有用户交互，用它们做负样本会把"是否热门"和"是否喜欢"混在一起，模型反而更难区分
- 这说明"难负样本一定更好"是误区；难负样本需要与评估分布、损失函数和训练预算一起调，不能直接套用

---

## 项目结构

```text
ctr-deepfm-pytorch/
├── configs/default.yaml       # 数据、模型、训练配置
├── src/recsys/
│   ├── data/                  # 解析 ml-1m、特征编码、留一法切分与负采样
│   ├── models/                # Embedding/线性/FM/MLP 层与四个模型
│   ├── train/                 # 训练循环、早停、AUC/HR/NDCG
│   ├── serving/               # Recommender、缓存、指标、FastAPI
│   ├── monitoring/            # 预测日志、PSI 漂移
│   ├── experiments.py         # 模型对比与消融
│   └── cli.py                 # `ctr` 命令行
├── tests/                     # 27 个单元/API 测试
├── scripts/download_data.py   # 下载并解压 MovieLens
├── Dockerfile / Makefile / pyproject.toml
```

| 模块 | 职责 |
|---|---|
| `data/loader.py` | 解析 `ratings/users/movies.dat`，提取年代与类型 |
| `data/features.py` | 类别词表、冷启动保留位、训练/服务共用编码 |
| `data/split.py` | 留一法切分、负采样、用户历史 |
| `models/layers.py` | 字段偏移 Embedding、线性层、FM 交叉、MLP |
| `train/trainer.py` | 手写训练循环、早停、按批推理 |
| `serving/recommender.py` | Top-K 排序、已看过滤、冷启动画像 |

---

## 快速开始

### 1. 安装

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -e ".[dev]"
```

PyTorch 默认从 PyPI 安装；如需 CPU 专用包：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. 下载数据

```bash
python scripts/download_data.py
# 网络受限时：浏览器下载 ml-1m.zip 放到 data/raw/，再执行
python scripts/download_data.py --skip-download
```

### 3. 训练与对比

```bash
ctr prepare                 # 切分 + 编码（含统计特征），约 1 分钟
ctr train --model deepfm    # 训练 DeepFM 并保存服务 bundle
ctr compare                 # 四个模型完整对比
ctr ablation                # Embedding 维度与负采样比例消融
ctr robustness              # 3 个随机种子 + Bootstrap 置信区间
ctr negatives               # 均匀负采样 vs 热度负采样
ctr coldstart               # 留出 10% 用户评估冷启动
ctr reference               # 生成漂移参考分布
ctr drift                   # 基于预测日志计算 PSI 漂移
python scripts/benchmark.py # 冷/热缓存延迟基准
```

### 4. 启动推荐服务

```bash
ctr serve --port 8000
# Swagger: http://127.0.0.1:8000/docs
```

```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "top_k": 5}'
```

冷启动（用户不在训练集）：

```json
{
  "user_id": 99999,
  "top_k": 5,
  "profile": {"gender": "M", "age": "35", "occupation": "2", "zip_prefix": "456"}
}
```

---

## 服务接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 模型加载状态 |
| GET | `/model/info` | 模型名、用户/电影数、评估指标 |
| POST | `/recommend` | 给定用户返回 Top-K 电影（可选过滤已看） |
| POST | `/score` | 对指定候选电影打分 |

---

## 工程化：可观测性、监控与性能

### 请求链路

- 每个请求生成 `x-request-id`，同时写入响应头与结构化 JSON 日志
- 日志字段：`request_id`、`user_id`、`cold_start`、`cache`、`n_items`、`latency_ms`
- `/metrics` 暴露：请求数（按状态码）、请求延迟直方图、推荐条数、推荐分数分布、缓存命中/未命中、模型版本

```text
{"timestamp": "...", "logger": "recsys.access", "message": "recommend",
 "request_id": "a1b2c3d4", "user_id": 1, "cold_start": false,
 "cache": "miss", "n_items": 10, "latency_ms": 118.4}
```

### 预测日志与漂移

- 每次 `/recommend` 写入 SQLite：用户画像、Top-K 结果、Top-1 分数、延迟、模型版本
- `ctr reference` 通过**服务同一路径**抽样 300 个用户，保存 Top-1 分数与画像分布
- `ctr drift` 对画像字段与 Top-1 分数计算 PSI（阈值 0.1 / 0.25）

实测：

| 场景 | 结果 |
|---|---|
| 同分布回放 350 条 | `stable`，max PSI **0.0738** |
| 注入 30 条异常画像（age=99 / occupation=99 / zip=999） | `drift`，age PSI **0.9125** |

报告：`reports/drift_report_stable.html`、`reports/drift_report_shifted.html`。

两个真实踩坑，值得在面试里讲：

1. **漂移口径必须和线上一致**：最初参考分布用的是测试候选集（含大量负样本）的分数，而线上记录的是全量物品 Top-K 分数，首日就误报 PSI=12。改成 Top-1 分数后恢复正常
2. **高基数特征要降噪**：邮编有约 680 个取值，350 个请求下 PSI 被采样噪声推到 0.15；监控层折叠到首位数字后回到 0.06

### 缓存与延迟

- 用户级 LRU + TTL 缓存（默认 4096 用户 / 300 秒），key 包含模型版本和画像，模型更新后自动失效
- 冷缓存：P50 **121 ms**、P95 **268 ms**（一次请求对 3883 部电影全量打分）
- 热缓存：P50 **0.72 ms**、P95 **1.36 ms**，P95 加速 **197×**
- 基准脚本：`python scripts/benchmark.py` → `reports/benchmark.json`

### 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `CTR_BUNDLE_DIR` | `models/bundle` | 模型 bundle 路径 |
| `CTR_PREDICTION_LOG` | `data/predictions.db` | 预测日志路径 |
| `CTR_CACHE_SIZE` | `4096` | 用户缓存条目上限（0 关闭） |
| `CTR_CACHE_TTL` | `300` | 缓存 TTL（秒） |
| `CTR_NUM_THREADS` | CPU 核数一半 | PyTorch 线程数 |

---

## 工程与测试

- **27 个测试**，覆盖率 **67%**（含覆盖率门禁 60%）：数据切分、两种负采样、统计特征、FM 数学、Bootstrap 置信区间、预测日志、漂移、缓存、API
- `ruff` + `mypy` 零错误；`pre-commit` 钩子防止格式回退
- 随机种子同时控制参数初始化与样本打乱，已验证相同种子两次运行结果完全一致
- CI 两阶段：`quality`（lint → mypy → 覆盖率测试）与 `docker-smoke`（构建镜像、起容器、调 `/health`、`/recommend`、`/metrics`）
- Docker 多阶段构建（CPU 版 PyTorch），非 root + 健康检查

### Docker

```bash
docker build -t ctr-deepfm:latest .
docker run -p 8000:7860 ctr-deepfm:latest

# 国内网络：torch 走阿里云 wheel 镜像，pip 与 apt 走清华镜像
docker build -t ctr-deepfm:latest . \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  --build-arg TORCH_FIND_LINKS=https://mirrors.aliyun.com/pytorch-wheels/cpu/ \
  --build-arg APT_MIRROR=mirrors.tuna.tsinghua.edu.cn
```

---

## 设计取舍

- **不做分布式 / 特征存储**：单机 CPU、百万级样本，引入重型组件只会增加讲述成本
- **结果按多种子 + 置信区间报告**：单次运行不用于模型对比结论
- **负样本只排除已评分物品**：未曝光物品仍可能被当作负样本，存在假阴性
- **热度负采样实验保留负结果**：它在本协议下更差，说明"难负样本一定更好"是误区

---

## 局限与改进方向

1. 冷启动（完全不可见用户）的 HR@10 比热启动低 20.7 个点，仍需要独立的召回策略
2. 未引入序列模型（DIN/DIEN）或图模型，推荐链路仍偏 CTR 静态特征
3. 冷缓存单次请求需要全量打分（P95 268 ms），物品规模到十万级时需要两阶段召回 + ANN
4. 只做了留一法协议，没有做多折交叉验证；好在这份数据上种子方差很小（AUC ±0.0005）
5. 未做线上曝光数据验证，负样本与真实曝光分布仍有差距

---

## 数据来源

[MovieLens-1M](https://grouplens.org/datasets/movielens/1m/)，GroupLens Research，用于学术研究。
