# FinVisor: 企业经营预测与投资分析平台

**FinVisor** 是一个基于大数据和机器学习的企业经营预测与投资分析平台。该项目旨在帮助投资者和决策者通过实时的数据分析、精准的市场预测和科学的决策支持工具，更好地应对复杂的金融市场环境。本平台通过爬取上市公司股票数据与财务报表，结合数学建模和机器学习方法，提供全面的投资分析和经营预测。

## 项目特点与创新

- **数据获取与处理**：利用 `Tushare` 接口爬取A股市场股票数据与财务报表，确保数据的时效性与全面性。
- **实时预测**：运用机器学习算法（如 LSTM）对企业未来收益与市场趋势进行预测。
- **数据可视化**：开发交互式可视化平台，帮助用户直观理解分析结果与预测信息。
- **综合分析**：结合多种数学建模与机器学习方法，提高预测精度与数据分析的准确性。

## 项目目标

本项目旨在实现以下目标：

1. 为决策者提供科学、准确的经营预测数据，支持他们做出更为明智的经营策略。
2. 为投资者提供精准的投资分析与市场预测，帮助他们规避风险并提高投资回报率。
3. 提高金融数据分析的效率，通过机器学习提升预测准确性。

## 技术栈

- **Python**: 主要编程语言。
- **Tushare**: 获取中国股市数据。
- **Matplotlib / Seaborn**: 数据可视化。
- **PyTorch**: 深度学习与模型训练。
- **Scikit-learn**: 机器学习模型与数据预处理。

## 安装与运行

### 环境要求

- Python 3.x
- 安装以下依赖库：

```
pip install tushare pandas numpy matplotlib seaborn scikit-learn torch
```

### 使用 Tushare API 获取数据

1. **注册 Tushare**: 访问 [Tushare官网](https://tushare.pro) 并注册账号，获取 API Token。
2. **设置 API Token**：在代码中配置你的 Token：

```
import tushare as ts
ts.set_token('你的API Token')
pro = ts.pro_api()
```

## 项目结构

```
FinVisor/
│
├── data/                  # 存储下载的股票数据和财务报表
│   ├── 000031.csv         # 示例：某股票数据
│
├── utils/                 # 主要功能代码文件夹
│   ├── data_update.py     # 数据更新
│   ├── LSTM.py            # 模型训练
│   ├── utils.py           # 工具函数（如数据处理、特征工程等）
│
├── requirements.txt       # 依赖库
├── README.md              # 项目说明文档
└── main.py                # 主程序入口
```

## 使用说明

1. 下载并安装所有依赖。
2. 获取 Tushare 数据并将其存储到 `data/` 文件夹中。
3. 调用 `train_model.py` 来训练模型，模型会保存至指定路径。
4. 使用 `predict.py` 对新数据进行预测，生成实时预测结果。

## 贡献

欢迎贡献！如果你有任何改进建议、问题报告或新功能，欢迎提一个 [Issue](https://github.com/Pchan-nju/FinVisor/issues) 或提交 Pull Request。

## 致谢

感谢 Tushare 提供的金融数据接口，感谢开源社区的所有贡献者。