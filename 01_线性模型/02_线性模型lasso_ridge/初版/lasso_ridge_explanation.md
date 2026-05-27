# 线性模型 Lasso / Ridge 代码解读

## 总览

本 notebook 在 baseline（逐日 OLS）的基础上，将建模核心替换为**带正则化的线性回归**，其余流程（预处理 → IC 筛选 → 共线去重 → EWM 平滑 → Walk-Forward 预测 → IC 评估）与 baseline 完全一致。

```
230 因子 → [IC 筛选 76] → [共线去重 38] → 逐日 Lasso/Ridge β → EWM 平滑 → shift(1) 预测 → IC 评估
```

### 与 baseline 的核心区别

| 项目 | Baseline (OLS) | 本 notebook (Lasso/Ridge) |
|------|---------------|--------------------------|
| 建模方法 | `np.linalg.lstsq`，无惩罚项 | `sklearn.Lasso` / `sklearn.Ridge`，带 L1/L2 惩罚 |
| 参数 | 无超参数 | `alpha`（正则化强度） |
| 稀疏性 | 无，所有系数非零 | Lasso 可将系数压缩至 0（自动特征选择） |
| 共线性处理 | 依赖前置去重 | Ridge 本身对共线性更鲁棒，Lasso 倾向于只保留一个 |
| 新增参数 | — | `MODEL_TYPE`（lasso/ridge）、`ALPHA` |

---

## Step 0: 参数配置

```python
FOLD_ID         = 1           # fold_1: train 2016-2019, test 2020 H1
LABEL_COL       = "FWD_RET_10D"
WINSORIZE_MAD   = 3.0
IC_THRESHOLD    = 0.02
T_STAT_THRESHOLD= 2.0
CORR_THRESHOLD  = 0.7
EWM_HALFLIFE    = 20

# ↓ 新增：Lasso/Ridge 专属参数
MODEL_TYPE      = "lasso"     # "lasso" 或 "ridge"
ALPHA           = 0.001       # 正则化强度，越大惩罚越重，系数越小
```

### alpha 怎么选

- **alpha 越小** → 越接近 OLS，系数可以很大，容易过拟合
- **alpha 越大** → 系数被压缩得越狠，模型越保守，但可能欠拟合
- 起点推荐：Lasso 用 `0.001`，Ridge 用 `0.01`
- 注意：leader 说了不要过分调参（因为标签可能还会变），先跑一个合理的值出结果即可

---

## Step 1: 函数定义（15 个）

在 baseline 14 个函数基础上新增 1 个，其余完全复用：

| 组 | 函数 | 作用 | 变化 |
|----|------|------|------|
| 数据 | `load_fold` | 读 CSV，按 trade_date+ts_code 排序 | 不变 |
| 数据 | `identify_columns` | 自动识别 `ALPHA_*` / `FWD_RET_*` | 不变 |
| 预处理 | `cross_section_winsorize` | 逐日 MAD 截尾 | 不变 |
| 预处理 | `cross_section_demean` | 逐日去均值 | 不变 |
| 预处理 | `cross_section_zscore` | 逐日标准化 | 不变 |
| 筛选 | `compute_ic_panel` | 逐日 Spearman IC 面板 | 不变 |
| 筛选 | `summarize_ic` | IC mean/std/t/pos_ratio | 不变 |
| 筛选 | `filter_by_ic` | 阈值过滤 | 不变 |
| 筛选 | `spearman_union_find` | 共线性并查集分组 | 不变 |
| 筛选 | `select_representatives` | 每组选 \|IC\| 最大 | 不变 |
| **建模** | **`daily_regularized_beta`** | **逐日 Lasso/Ridge 拟合** | **新增，替换 `daily_ols_beta`** |
| 建模 | `smooth_beta_ewm` | `ewm(halflife=20).mean()` | 不变 |
| 预测 | `predict_walkforward` | `X[t] @ β_ewm[t-1]` | 不变 |
| 评估 | `evaluate_ic` | 日度 Spearman + 汇总 | 不变 |

---

## Step 2: 数据加载（与 baseline 完全相同）

```
fold_1_train.csv → train_df: 282,719 行, 976 日
fold_1_test.csv  → test_df:   32,331 行, 108 日
```

---

## Step 3: 截面预处理（与 baseline 完全相同）

train 和 test 各自独立处理，三步流水线：

1. **MAD Winsorize**：去极值，3 倍 MAD 截断
2. **Demean**：逐日截面减均值
3. **Z-score**：逐日标准化

缺失值处理方式同 baseline：**不填充，NaN 自然传播**，在拟合时用 `np.isfinite` 过滤掉含 NaN 的行。

---

## Step 4: IC 初筛（与 baseline 完全相同）

**只用训练集**计算。`|ic_mean| > 0.02` 且 `|t_stat| > 2.0`。

预计结果：230 → 76 个因子（与 baseline 一致，因为这一步和模型无关）。

---

## Step 5: 共线性去重（与 baseline 完全相同）

Spearman 相关矩阵 + 并查集，阈值 0.7。

预计结果：76 → 38 个因子（与 baseline 一致）。

> **注意**：Ridge 本身对共线性有一定免疫力（正则化让系数都变小而不是某个为 0），保留去重步骤是为了和 baseline 保持可比性。Lasso 本身也有特征选择能力，但前置去重能加速收敛并减少不稳定性。

---

## Step 6: 逐日正则化回归（核心改动）

### 与 OLS 的区别

OLS 最小化：`||y - Xβ||²`

Lasso 最小化：`||y - Xβ||² + alpha * ||β||₁`（L1 惩罚，部分系数变为精确的 0）

Ridge 最小化：`||y - Xβ||² + alpha * ||β||²`（L2 惩罚，所有系数变小但不为 0）

### 实现

```python
from sklearn.linear_model import Lasso, Ridge

def daily_regularized_beta(df, x_cols, y_col, model_type="lasso", alpha=0.001):
    # 每日切截面 → sklearn 拟合 → 提取系数
    # 输出格式与 daily_ols_beta 完全一致：index=trade_date, columns=x_cols
```

关键点：
- sklearn 的 `Lasso`/`Ridge` 通过 `fit_intercept=True` 单独处理截距，输出时只取 `coef_`（不含截距）保持与 baseline β 格式一致
- 每日样本数不足时（`< x_cols数量 + 5`）跳过，同 baseline
- 输出的 β 是系数向量，后续 EWM 平滑和 walk-forward 预测逻辑**完全不变**

---

## Step 7: EWM 平滑 + Shift（与 baseline 完全相同）

```python
beta_all      = pd.concat([beta_train, beta_test]).sort_index()
beta_ewm      = beta_all.ewm(halflife=20, adjust=False).mean()
beta_for_pred = beta_ewm.shift(1)   # ← 防泄漏核心，不能改
```

---

## Step 8: Walk-Forward 预测（与 baseline 完全相同）

```python
# 对每个测试日 d:
#   y_pred[d] = X[d] @ β_for_pred[d]
# β_for_pred[d] = β_ewm[d-1]
```

---

## Step 9: IC 评估（与 baseline 完全相同）

```python
ic_series, summary = evaluate_ic(pred_df)
```

| 指标 | Baseline (OLS) 参考值 | Lasso/Ridge 预期 |
|------|----------------------|-----------------|
| ic_mean | 0.1357 | 待跑出 |
| ic_ir | 5.66 | 待跑出 |
| pos_ratio | 87.96% | 待跑出 |

---

## Step 10: 持久化输出

输出目录：`D:/quant/ML_teamwork/01_线性模型/02_线性模型lasso_ridge/output/{date}/`，文件名和格式与 baseline 完全一致：

| 文件 | 内容 |
|------|------|
| `fold_1_y_pred.parquet` | trade_date, ts_code, y_pred, y_true |
| `fold_1_ic_series.csv` | 日度 IC |
| `fold_1_summary.csv` | 6 项汇总指标 + model_type + alpha |
| `fold_1_kept_factors.txt` | 保留因子名 |
| `fold_1_beta_history.parquet` | β 时间序列（平滑前） |

---

## 防泄漏设计总结（与 baseline 完全相同）

| 环节 | 防泄漏措施 |
|------|-----------|
| 预处理 | train/test 各自独立，不跨集 |
| IC 筛选 | 只用 train 计算 IC |
| 共线去重 | 只用 train 计算相关矩阵 |
| 正则化拟合 | train/test 分别逐日拟合，只用 `β_ewm[t-1]` 预测 |
| EWM + shift | `shift(1)` 确保 t 日看不到 t 日信息 |

---

## 如何对比 baseline

跑完后，在 summary.csv 中对比以下指标：

```
模型       ic_mean    ic_ir    pos_ratio    n_features_kept
OLS        0.1357     5.66     87.96%       38
Lasso      ?          ?        ?            ?（可能 < 38，因为部分系数被压为 0）
Ridge      ?          ?        ?            38（不稀疏）
```

Lasso 的实际使用因子数可通过统计非零系数的天数来近似反映。
