# Lasso / Ridge 正则化线性模型代码解读（Baseline 对齐版）

## 总览

本组 notebook 在 OLS baseline 的基础上，将逐日截面回归模型替换为带正则化的线性回归模型，包括：

- Lasso 固定 alpha 版
- Ridge 固定 alpha 版
- Lasso 训练集内部调参版
- Ridge 训练集内部调参版

四个版本均采用与原 OLS baseline 一致的 beta 更新口径：

```text
230 因子
→ IC 初筛
→ 共线去重
→ 逐日 Lasso / Ridge 拟合 beta
→ train beta 与 test beta 拼接
→ EWM 平滑
→ shift(1)
→ Walk-Forward 预测
→ IC 评估与每日排序输出
```

本版本主要用于和原 OLS baseline 做横向对比。因此，Lasso / Ridge 沿用原 baseline 的 `beta_test + EWM + shift(1)` 设计，而不是最严格的样本外预测口径。

---

## 一、任务定位

本任务仍然属于**回归预测 + 排序选股**，不是分类任务。

模型预测目标为连续收益：

```text
FWD_RET_10D
```

模型输出为连续预测分数：

```text
y_pred = intercept + X @ beta
```

在排序输出中，将 `y_pred` 记为：

```text
reg_score
```

含义是：模型对未来 10 日收益的连续预测得分。`reg_score` 越高，该股票在当日横截面内的排序越靠前。

---

## 二、与 OLS Baseline 的核心区别

| 项目 | OLS Baseline | Lasso / Ridge |
|---|---|---|
| 模型类型 | 普通最小二乘回归 | 正则化线性回归 |
| 目标变量 | FWD_RET_10D | FWD_RET_10D |
| 输出 | 连续收益预测值 | 连续收益预测值 |
| 是否分类 | 否 | 否 |
| 正则项 | 无 | 有 |
| 主要参数 | 无 | alpha |
| 排序分数 | y_pred | y_pred / reg_score |

OLS 的目标函数是：

```text
min ||y - Xβ||²
```

Lasso 的目标函数是：

```text
min ||y - Xβ||² + alpha * ||β||₁
```

Ridge 的目标函数是：

```text
min ||y - Xβ||² + alpha * ||β||²
```

其中，Lasso 使用 L1 正则化，可能将部分系数压缩为 0，具有一定特征选择效果；Ridge 使用 L2 正则化，通常不会将系数压缩为 0，而是整体缩小系数，缓解共线性带来的不稳定。

---

## 三、四个 notebook 版本说明

| 文件 | 模型 | alpha 设置方式 | 说明 |
|---|---|---|---|
| `lasso_baseline_aligned_intercept.ipynb` | Lasso | 固定 alpha | 使用预设 alpha，直接训练与评估 |
| `ridge_baseline_aligned_intercept.ipynb` | Ridge | 固定 alpha | 使用预设 alpha，直接训练与评估 |
| `lasso_tune_baseline_aligned_intercept.ipynb` | Lasso | 训练集内部网格搜索 | 在训练集末尾切出验证集选择 alpha |
| `ridge_tune_baseline_aligned_intercept.ipynb` | Ridge | 训练集内部网格搜索 | 在训练集末尾切出验证集选择 alpha |

固定 alpha 版本适合快速和 baseline 对比；调参版本用于观察不同正则化强度对 IC 的影响。

---

## 四、Step 0：参数配置

核心参数包括：

```python
FOLD_ID          = 1
LABEL_COL        = "FWD_RET_10D"
WINSORIZE_MAD    = 3.0
IC_THRESHOLD     = 0.02
T_STAT_THRESHOLD = 2.0
CORR_THRESHOLD   = 0.7
EWM_HALFLIFE     = 20
MODEL_TYPE       = "lasso" 或 "ridge"
ALPHA            = 固定值，或由 ALPHA_GRID 搜索得到
```

固定 alpha 版本直接指定：

```python
ALPHA = 0.001   # Lasso 示例
ALPHA = 0.01    # Ridge 示例
```

调参版本使用：

```python
ALPHA_GRID = [0.0001, 0.001, 0.01, 0.1]
VAL_DAYS = 60
```

即从训练集最后 60 个交易日切出验证集，在训练集内部选择 alpha。

---

## 五、Step 1：函数定义

主要函数分为六类。

### 1. 数据读取

```python
load_fold()
identify_columns()
```

负责读取 train/test CSV，并识别 `ALPHA_*` 因子列和 `FWD_RET_10D` 标签列。

### 2. 截面预处理

```python
cross_section_winsorize()
cross_section_demean()
cross_section_zscore()
```

逐日按 `trade_date` 分组，对每个因子进行：

```text
MAD 去极值 → 截面去均值 → 截面标准化
```

这样每个交易日内部的因子具有相对可比性。

### 3. 因子筛选

```python
compute_ic_panel()
summarize_ic()
filter_by_ic()
```

只在训练集上计算因子与 `FWD_RET_10D` 的日度 Spearman IC，并按：

```text
|ic_mean| > IC_THRESHOLD
|t_stat|  > T_STAT_THRESHOLD
```

筛选初步有效因子。

### 4. 共线性去重

```python
spearman_union_find()
select_representatives()
```

对 IC 初筛后的因子计算 Spearman 相关矩阵，若两个因子的绝对相关超过 `CORR_THRESHOLD`，则视为共线。随后用并查集将共线因子合并为组，并在每组中保留 `|IC mean|` 最大的代表因子。

### 5. 正则化建模

```python
daily_regularized_beta()
```

逐日截面拟合 Lasso 或 Ridge，输出每日 beta 时间序列。

本版本已经统一保存截距项：

```text
__intercept__
```

因此预测公式为完整形式：

```text
y_pred = intercept + X @ beta
```

### 6. 预测与评估

```python
smooth_beta_ewm()
predict_walkforward()
evaluate_ic()
```

用于 beta 平滑、测试期预测、日度 IC 评估。

---

## 六、Step 2：数据加载

读取 fold 数据：

```text
fold_1_train.csv
fold_1_test.csv
```

训练集用于：

```text
预处理参数计算
IC 筛选
共线去重
训练期 beta 拟合
```

测试集用于：

```text
测试期 beta 拟合
预测
评估
排序输出
```

注意：本版本为了与原 baseline 保持一致，测试集也会逐日拟合 beta，并通过 `shift(1)` 避免使用当天 beta 预测当天。这是 baseline 对齐口径。

---

## 七、Step 3：截面预处理

预处理在 train 和 test 中分别进行，不跨数据集混合。

每个交易日内部，依次执行：

### 1. MAD Winsorize

```text
median ± 3 * MAD
```

用于削弱极端值影响。

### 2. Demean

每个因子减去当日截面均值，用于消除截面水平偏移。

### 3. Z-score

每个因子除以当日截面标准差，使不同因子尺度可比。

---

## 八、Step 4：IC 初筛

IC 初筛只使用训练集，防止测试集信息参与因子选择。

流程为：

```text
每个交易日：
    计算每个 ALPHA 因子与 FWD_RET_10D 的 Spearman IC

对所有训练日汇总：
    ic_mean
    ic_std
    t_stat
    pos_ratio

按阈值筛选：
    |ic_mean| > 0.02
    |t_stat| > 2
```

这一步与 OLS baseline 保持一致。

---

## 九、Step 5：共线性去重

对 IC 筛选后的因子计算 Spearman 相关矩阵。

若：

```text
|corr(i, j)| > 0.7
```

则认为两个因子高度共线。

通过并查集将共线因子分组后，每组只保留一个代表因子：

```text
保留 |IC mean| 最大的因子
```

这样可以降低因子冗余，减少模型不稳定。

---

## 十、Step 5.5：Alpha 调参版本

调参版本会在训练集内部切出验证集：

```text
search_df = 训练集前半部分
val_df    = 训练集最后 60 个交易日
```

对每个候选 alpha：

```text
1. 在 search_df 上逐日拟合 beta
2. 对 beta 做 EWM 平滑
3. 将 beta 延续到验证集日期
4. shift(1)
5. 在 val_df 上预测
6. 计算验证集 IC
```

选择验证集 IC 最高的 alpha 作为最终参数：

```python
ALPHA = best_alpha
```

这一过程只使用训练集内部数据，不使用测试集选择 alpha。

---

## 十一、Step 6：逐日 Lasso / Ridge 拟合 beta

固定 alpha 版本直接使用指定 `ALPHA`。

调参版本先完成 Step 5.5，得到最优 `ALPHA`。

随后在训练集和测试集上分别逐日拟合 beta：

```python
beta_train = daily_regularized_beta(train_df, x_cols, label_col, MODEL_TYPE, ALPHA)

beta_test  = daily_regularized_beta(test_df,  x_cols, label_col, MODEL_TYPE, ALPHA)
```

输出的 beta 包括：

```text
__intercept__
ALPHA_xxx
ALPHA_xxx
...
```

其中 `__intercept__` 是每日截距项。

---

## 十二、Step 7：EWM 平滑 + Shift(1)

为了与原 OLS baseline 保持一致，本版本沿用 baseline beta 更新口径：

```python
beta_all = pd.concat([beta_train, beta_test]).sort_index()

beta_ewm = smooth_beta_ewm(beta_all, EWM_HALFLIFE)

beta_for_pred = beta_ewm.shift(1)
```

含义是：

```text
t 日预测使用 t-1 日及更早的 beta 信息。
```

需要注意的是：由于 `beta_test` 是使用测试集真实标签拟合得到的，因此该口径适合与原 baseline 做横向比较，但不是最严格的样本外预测口径。

如果要做严格样本外实验，应改为只使用 `beta_train`，并且不使用测试集真实标签更新 beta。

---

## 十三、Step 8：Walk-Forward 预测

预测公式为：

```text
y_pred = intercept + X @ beta
```

其中：

```text
intercept = beta_for_pred["__intercept__"]
beta      = beta_for_pred[x_cols]
X         = 当日股票因子矩阵
```

预测结果保存为：

```text
trade_date
ts_code
y_pred
y_true
```

其中：

```text
y_true = FWD_RET_10D
```

`y_pred` 是连续收益预测分数。

---

## 十四、Step 9：IC 评估

每日计算：

```text
Spearman(y_pred, y_true)
```

得到日度 IC 序列。

汇总指标包括：

```text
ic_mean
ic_std
ic_ir
pos_ratio
n_days
n_obs
```

其中：

```text
ic_ir = ic_mean / ic_std * sqrt(252 / 10)
```

该指标与原 OLS baseline 可比。

---

## 十五、Step 10：持久化输出

每个 notebook 会输出以下文件：

| 文件 | 内容 |
|---|---|
| `fold_1_y_pred.parquet` | 测试集预测结果，含 y_pred 和 y_true |
| `fold_1_ic_series.csv` | 日度 IC 序列 |
| `fold_1_summary.csv` | 模型参数与评估汇总 |
| `fold_1_kept_factors.txt` | 最终保留因子 |
| `fold_1_beta_history.parquet` | 平滑前 beta 时间序列 |
| `fold_1_rank_output.csv` | 每日股票排序结果 |

其中 `rank_output.csv` 的格式为：

```text
trade_date
rank
stockid
reg_score
```

对 Lasso/Ridge 来说：

```text
reg_score = y_pred
```

即连续收益预测分数。

排序规则为：

```text
每个 trade_date 内，按 reg_score 从高到低排序。
```

---

## 十六、关于 baseline 对齐口径与严格样本外口径

本版本采用的是 baseline 对齐口径：

```text
train/test 都逐日拟合 beta
→ 拼接 beta_train 与 beta_test
→ EWM 平滑
→ shift(1)
→ 用 t-1 beta 预测 t
```

优点是：

```text
与原 OLS baseline 完全一致，便于横向比较模型本身的差异。
```

需要说明的是：

```text
该口径中 beta_test 的拟合使用了测试集真实 FWD_RET_10D。
虽然 shift(1) 避免了使用当天 beta 预测当天，但测试期 beta 本身仍依赖测试集标签。
因此它不是最严格意义上的样本外预测。
```

如果研究目标是严格样本外预测，应采用：

```text
只用训练集 beta
测试期不再使用真实标签更新 beta
必要时考虑 FWD_RET_10D 的标签可得性滞后
```

因此，在汇报中建议明确区分：

```text
baseline 对齐实验：用于和原 OLS baseline 比较。
严格样本外实验：用于评估真实预测能力。
```

---

## 十七、为什么 Lasso / Ridge 可能不一定超过 OLS

正则化模型不保证 IC 一定高于 OLS。

可能原因包括：

### 1. Alpha 偏大导致欠拟合

如果 alpha 过大，系数被压缩过强，预测分数区分度下降，IC 可能变差。

### 2. 前置筛选已经较强

若 OLS baseline 已经通过 IC 筛选和共线去重保留了较稳定的因子，正则化带来的增益可能有限。

### 3. Lasso 会压掉部分因子

Lasso 的 L1 正则可能将部分仍有边际贡献的因子系数压为 0，从而损失部分信号。

### 4. Ridge 更偏向稳定而非稀疏

Ridge 主要降低系数波动，未必提高单期 IC。

因此，Lasso/Ridge 的价值不只体现在 IC 是否更高，也体现在：

```text
系数是否更稳定
预测是否更平滑
不同 alpha 下结果是否稳健
是否减少过拟合风险
```

---

## 十八、最终汇报建议

可以这样表述本组实验设计：

```text
本实验在 OLS baseline 的基础上，引入 Lasso 和 Ridge 两类正则化线性模型，用于检验正则化约束是否能够改善多因子收益预测与横截面排序效果。为保证与原 baseline 可比，主实验沿用 baseline 的 beta 更新口径，即训练集和测试集均逐日拟合 beta，拼接后进行 EWM 平滑，并使用 shift(1) 预测下一交易日。模型输出为连续收益预测分数 y_pred，在每日横截面内按该分数降序排列，得到 rank / stockid / reg_score 格式的排序结果。调参版本仅在训练集内部切分验证集选择 alpha，不使用测试集调参。需要注意，该 baseline 对齐口径主要服务于模型横向比较，并非最严格样本外口径。
```
