# dataxx: 使用 `dataX.mat` 训练并评估回归模型（Python）

本项目提供一套可复现的 Python 方案，满足以下约束：

- 只使用 `inputtrain` 和 `outputtrain` 训练模型；
- 不使用测试集参与训练；
- 在测试集 `inputtest` 上预测，并与 `outputtest` 对比；
- 目标：测试集 RMSE < `1e-2`。

---

## 1. 数据说明

`dataX.mat` 需要包含以下变量：

- `inputtrain`：训练输入
- `outputtrain`：训练输出
- `inputtest`：测试输入
- `outputtest`：测试输出（仅用于最终评估）

脚本会自动处理常见的列向量/行向量维度情况。

---

## 2. 模型说明

本实现采用：

- `StandardScaler` 对输入做标准化；
- `TransformedTargetRegressor + StandardScaler` 对输出做标准化；
- 核心回归器：`GaussianProcessRegressor`（高斯过程回归，RBF + 常数核 + 白噪声核）；
- 使用训练集交叉验证估计训练侧泛化误差（仅训练数据），折数最多为 5，并会根据训练样本数自动调整。

选择理由：

- 对平滑非线性映射拟合能力强；
- 对中小规模数据常能达到较低 RMSE；
- 结合输入/输出标准化后，数值更稳定、复现性更好。

---

## 3. 环境与安装

建议 Python 3.10+。

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

当前项目依赖已在 `requirements.txt` 中列出，包括：

- `numpy`
- `scipy`
- `scikit-learn`

---

## 4. 运行方式

在仓库根目录执行：

```bash
python train.py --data dataX.mat --report results.json --save-pred predictions_test.npy
```

运行后将：

- 在终端打印评估结果（JSON）；
- 生成 `results.json`（包含 RMSE 与是否达标）；
- 生成 `predictions_test.npy`（测试集预测值）。

如果测试集 RMSE 未达到 `1e-2`，脚本将以非 0 退出码退出，并给出提示。

---

## 5. 可复现性说明

- 固定随机种子 `random_state=42`；
- 交叉验证使用 `KFold(shuffle=True, random_state=42)`；
- 交叉验证折数最多为 5，并会根据训练样本数自动调整；
- 所有训练与调参均仅基于训练集。

---

## 6. 输出示例（字段说明）

`results.json` 关键字段：

- `cv_rmse_train_only`：仅训练集上的交叉验证 RMSE；
- `test_rmse`：测试集 RMSE（最终指标）；
- `threshold_met`：是否满足 `test_rmse < 1e-2`。

---

## 7. 若需进一步优化

如未满足阈值，可在 `train.py` 中尝试：

- 增大 `n_restarts_optimizer`；
- 调整 kernel 组合（例如增加 `Matern` / `RationalQuadratic`）；
- 在不使用测试集训练的前提下，仅基于训练集做更系统的超参数搜索。
