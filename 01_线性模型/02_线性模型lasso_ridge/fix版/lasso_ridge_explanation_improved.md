# 线性模型 Lasso / Ridge 代码解读

## 总览

本组 notebook 在 OLS baseline 的基础上，将逐日截面 OLS 替换为带正则化的线性回归，包括 Lasso 与 Ridge 两类模型。任务本质仍然是**回归预测 + 排序选股**，不是分类任务。

模型输入为每日股票截面的因子值，标签为连续型未来收益 `FWD_RET_10D`。模型输出 `y_pred` 表示预测收益得分；为了符合任务清单中的输出形式，最终会进一步生成：

```text
rank: 当日排序名次
stockid: 股票代码
reg_score: 模型预测得分
```

整体流程为：

```text
230 因子
→ 截面预处理
→ IC 初筛
→ 共线性去重
→ 逐日 Lasso / Ridge 回归
→ EWM 平滑
→ shift(1)
→ Walk-Forward 预测
→ IC 评估
→ rank / stockid / reg_score 排序输出
```

---

## 一、任务定位：回归模型，不是分类模型

Lasso 和 Ridge 都属于线性回归模型的正则化版本。它们直接拟合连续收益标签：

```python
LABEL_COL = "FWD_RET_10D"
```

因此这里的任务不是“涨 / 跌分类”，而是：

```text
用因子预测未来 10 日收益得分
再按预测得分从高到低排序股票
```

与逻辑回归不同，Lasso / Ridge 不需要把收益离散化为 0/1 标签，也不需要设计涨跌阈值。因此本 notebook 不使用 Accuracy、AUC、混淆矩阵，而主要使用 IC、IC_IR、正 IC 占比等排序型评估指标。

---

## 二、四个 notebook 的分工

本组文件分为“不调参版”和“调参版”。

| 文件 | 模型 | 是否调参 | 参数来源 | 作用 |
|---|---|---|---|---|
| `lasso_ridge.ipynb` | Lasso | 不调参 | 手动固定 `ALPHA` | 快速得到 Lasso 基准结果 |
| `lasso_ridge-ridge.ipynb` | Ridge | 不调参 | 手动固定 `ALPHA` | 快速得到 Ridge 基准结果 |
| `lasso_ridge调参.ipynb` | Lasso | 调参 | 训练集内部验证集选择 `ALPHA` | 比较不同 L1 正则强度 |
| `lasso_ridge调参-ridge.ipynb` | Ridge | 调参 | 训练集内部验证集选择 `ALPHA` | 比较不同 L2 正则强度 |

### 2.1 不调参版

不调参版直接给定一个固定正则化强度：

```python
MODEL_TYPE = "lasso"   # 或 "ridge"
ALPHA = 0.001
```

这种版本的优点是逻辑简单、运行快、方便与 OLS baseline 做初步对比。缺点是 `alpha` 不一定最优。

### 2.2 调参版

调参版使用训练集末尾若干天作为验证集：

```python
ALPHA_GRID = [0.0001, 0.001, 0.01, 0.1]
VAL_DAYS = 60
```

流程为：

```text
训练集前半段 search_df → 拟合不同 alpha
训练集末尾 val_df → 评估验证 IC
选择验证 IC 最高的 alpha
再用完整训练集拟合最终模型
最后只在测试集上评估
```

注意：调参只发生在训练集内部，测试集不参与 `alpha` 选择。

---

## 三、Lasso 与 Ridge 的模型差异

OLS 的目标函数为：

```text
min ||y - Xβ||²
```

Lasso 在 OLS 基础上加入 L1 惩罚：

```text
min ||y - Xβ||² + alpha * ||β||₁
```

Lasso 的特点是会把一部分系数压缩为 0，因此具有自动稀疏化和特征选择效果。适合在因子较多、希望模型更稀疏时使用。

Ridge 在 OLS 基础上加入 L2 惩罚：

```text
min ||y - Xβ||² + alpha * ||β||²
```

Ridge 的特点是通常不会把系数压成严格的 0，而是整体缩小系数幅度。它对共线性更稳定，适合在多个因子相关性较强但仍希望保留整体信息时使用。

---

## 四、参数配置

核心参数如下：

```python
FOLD_ID          = 1
LABEL_COL        = "FWD_RET_10D"
WINSORIZE_MAD    = 3.0
IC_THRESHOLD     = 0.02
T_STAT_THRESHOLD = 2.0
CORR_THRESHOLD   = 0.7
EWM_HALFLIFE     = 20
```

Lasso / Ridge 新增参数如下。

不调参版：

```python
MODEL_TYPE = "lasso"   # 或 "ridge"
ALPHA = 0.001
```

调参版：

```python
MODEL_TYPE = "lasso"   # 或 "ridge"
ALPHA_GRID = [0.0001, 0.001, 0.01, 0.1]
VAL_DAYS = 60
```

`alpha` 越大，正则化越强，系数越保守。`alpha` 越小，模型越接近 OLS。

---

## 五、主要代码步骤

### Step 1：函数定义

主要函数包括：

| 函数 | 作用 |
|---|---|
| `load_fold` | 读取训练集和测试集 |
| `identify_columns` | 识别 `ALPHA_*` 因子列与收益标签列 |
| `cross_section_winsorize` | 逐日截面 MAD 去极值 |
| `cross_section_demean` | 逐日截面去均值 |
| `cross_section_zscore` | 逐日截面标准化 |
| `compute_ic_panel` | 计算逐日因子 IC |
| `summarize_ic` | 汇总 IC 均值、标准差、t 值 |
| `filter_by_ic` | 根据 IC 和 t 值筛选因子 |
| `spearman_union_find` | 根据 Spearman 相关进行共线性分组 |
| `select_representatives` | 每个共线组选择代表因子 |
| `daily_regularized_beta` | 逐日拟合 Lasso / Ridge，返回截距和系数 |
| `smooth_beta_ewm` | 对 beta 时间序列进行 EWM 平滑 |
| `predict_walkforward` | 使用滞后 beta 预测测试集 |
| `make_rank_output` | 生成每日股票排序结果 |
| `evaluate_ic` | 计算日度 IC 和汇总指标 |

其中，AI 生成函数均在 docstring 中标明了输入和输出，符合任务清单中的代码说明要求。

---

## 六、预处理与特征筛选

预处理与 baseline 保持一致：

```text
MAD Winsorize → Demean → Z-score
```

训练集和测试集分别独立预处理，不跨集合混合。

IC 筛选只使用训练集：

```text
|ic_mean| > 0.02 且 |t_stat| > 2.0
```

共线性去重也只使用训练集。方法是计算保留因子之间的 Spearman 相关矩阵，对 `|ρ| > 0.7` 的因子连边，再用并查集找共线因子组，每组选取 `|IC|` 最大的因子作为代表。

---

## 七、逐日 Lasso / Ridge 拟合

每日单独取一个横截面：

```text
X_t = 当日所有股票的因子矩阵
y_t = 当日所有股票的未来 10 日收益
```

然后拟合：

```python
model = Lasso(alpha=ALPHA, fit_intercept=True)
# 或
model = Ridge(alpha=ALPHA, fit_intercept=True)
```

输出包括：

```text
__intercept__ + 所有因子系数
```

保留截距的原因是：虽然截距对同一日的横截面排序影响不大，但它是完整预测值的一部分，保存后模型结构更完整，也更接近 baseline 中带常数项的 OLS 设定。

---

## 八、防泄漏设计

修正版 notebook 采用更严格的防泄漏设定。

| 环节 | 防泄漏处理 |
|---|---|
| 预处理 | 训练集和测试集分别处理 |
| IC 筛选 | 只用训练集计算 |
| 共线性去重 | 只用训练集计算 |
| alpha 调参 | 只在训练集内部切分 search / validation |
| 模型拟合 | 只用训练集真实收益拟合 beta |
| 测试期预测 | 不使用测试集真实收益拟合 `beta_test` |
| EWM + shift | 测试日 t 只使用 t-1 及以前的 beta |

原始 notebook 中的一个关键风险是：在测试集上也拟合了 `beta_test`，再拼接进 beta 序列。即使后续做了 `shift(1)`，测试期后续日期仍可能间接使用前面测试日的真实收益标签。因此修正版去掉了测试集拟合，只使用训练期 beta 经 EWM 平滑后向测试期延展。

---

## 九、预测与排序输出

预测函数输出：

```text
trade_date
ts_code
y_pred
y_true
```

其中：

```text
y_pred = intercept + X @ beta
```

进一步生成排序文件：

```text
trade_date
rank
stockid
reg_score
```

其中：

```text
reg_score = y_pred
```

每个交易日内，`reg_score` 越高，`rank` 越靠前。

---

## 十、评估指标

主要评估指标为 IC：

```text
IC = SpearmanCorr(y_pred, y_true)
```

汇总指标包括：

| 指标 | 含义 |
|---|---|
| `ic_mean` | 日度 IC 均值 |
| `ic_std` | 日度 IC 标准差 |
| `ic_ir` | 年化 IC 信息比率 |
| `pos_ratio` | 正 IC 天数比例 |
| `n_days` | 有效评估天数 |

因为本任务是回归排序任务，所以不使用分类准确率、AUC 或混淆矩阵。

---

## 十一、输出文件

每个 notebook 会输出：

| 文件 | 内容 |
|---|---|
| `fold_1_y_pred.parquet` | 原始预测结果：`trade_date / ts_code / y_pred / y_true` |
| `fold_1_rank_output.csv` | 排序输出：`trade_date / rank / stockid / reg_score` |
| `fold_1_ic_series.csv` | 每个测试日的 IC |
| `fold_1_summary.csv` | 模型类型、alpha、IC 汇总指标、保留因子数、调参模式 |
| `fold_1_kept_factors.txt` | 最终保留的因子列表 |
| `fold_1_beta_history.parquet` | 训练期逐日 beta 历史 |

---

## 十二、版本使用建议

建议先运行不调参版，得到 Lasso 与 Ridge 的基础结果：

```text
lasso_ridge.ipynb
lasso_ridge-ridge.ipynb
```

然后再运行调参版：

```text
lasso_ridge调参.ipynb
lasso_ridge调参-ridge.ipynb
```

最终比较四组结果：

```text
Lasso 固定 alpha
Ridge 固定 alpha
Lasso 训练集内部调参
Ridge 训练集内部调参
```

比较时重点看：

```text
ic_mean
ic_ir
pos_ratio
n_features_kept
alpha
tuning_mode
```

如果调参版提升不明显，固定 alpha 版本更适合作为简洁可解释的主结果；如果调参版明显提升，则可以把调参版作为主结果，并在说明中强调测试集未参与调参。
