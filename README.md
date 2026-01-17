# 毒性预测：概率 + 把握（不确定性）+ 证据检索 + 改造建议（Web Demo）

这个小项目把你提出的想法做成了一个可运行原型：

1) **毒性概率**：输出 `p(toxic)`  
2) **把握/不确定性**：基于 evidential learning（Dirichlet）同时输出 `uncertainty`/`total_evidence`  
3) **证据**：从训练集中检索最相似序列（这些序列带 label，可视为“有实验/标注结果的证据样本”）  
4) **改造建议**：生成单点突变候选并按“更低毒性概率 + 更高把握”排序

当前实现使用你提供的 **ToxGIN 训练数据（序列+label）**：默认读取
`/root/private_data/dd_model/ToxGIN/train_sequence.csv` 与 `/root/private_data/dd_model/ToxGIN/test_sequence.csv`。

---

## 快速开始

### 1) 训练并生成 artifacts

```bash
python scripts/train_evi_tox.py --out-dir artifacts
```

如果你的数据不在默认路径：

```bash
python scripts/train_evi_tox.py --train-csv /path/to/train_sequence.csv --test-csv /path/to/test_sequence.csv --out-dir artifacts
```

会生成：

- `artifacts/evi_tox.pt`：evidential 模型权重
- `artifacts/retrieval.joblib`：证据检索索引（TF-IDF char n-gram）
- `artifacts/metrics.json`：简单评估指标

### 2) 启动 Web

```bash
python scripts/serve.py
```

浏览器打开：`http://127.0.0.1:8000`

---

## 输出解释（把握怎么定义）

模型输出 Dirichlet 参数 `alpha`（2 类）：

- `p(toxic)`：`alpha / sum(alpha)` 的期望概率（取第 2 维）
- `uncertainty`：`K / sum(alpha)`（K=2），越大表示证据越少、越不确定
- `confidence`：`1 - uncertainty`
- `total_evidence`：`sum(alpha) - K`

---

## 怎么迁移到你的 ToxGIN（改造点）

这个原型把 **“evidential head + evidential loss + uncertainty 输出”** 独立成了 `toxapp/evidential.py` 与 `toxapp/model.py` 的形式。

如果你要在 ToxGIN（GIN 图模型）里做同样改造，核心就是：

1) 把原来的 `Sigmoid + BCELoss` 改为 **2 类 Dirichlet 输出 + `dirichlet_evidence_loss`**
2) 推理阶段同时输出 `p(toxic)` 与 `uncertainty/total_evidence`
3) 增加一个“证据检索”模块，把最相似的训练样本（带 label）作为可解释证据展示到 Web

---

## 下一步建议（让它更像“新方式做老课题”）

- 把 evidential head 接到你现有的 ToxGIN 图表示上（结构+序列），而不是当前的序列-only baseline
- 把证据从“训练集相似序列”扩展到外部数据库/文献（例如对接特定毒肽数据库，或做序列检索/BLAST 再自动生成引用与证据摘要）
- 给“改造建议”加约束：保持活性/结构、限制疏水性/净电荷、避免关键半胱氨酸框架被破坏等

