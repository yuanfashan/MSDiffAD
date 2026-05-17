# MSDiffAD

MSDiffAD 是一个用于质谱数据的双阶段表征学习框架。它协同生成式扩散增强策略与对比学习机制，实现了分子结构特征与环境噪声的有效解耦，从而获取鲁棒的化学低维嵌入向量（Embeddings）。

---





## 环境配置

请按照以下步骤，使用项目自带的 `environment.yml` 文件创建并激活名为 `MSDiffAD` 的 Conda 虚拟环境。

### 1. 创建虚拟环境
打开Anaconda Prompt，运行以下命令开始安装：
```bash
conda env create -f environment.yml
```



### 2. 激活虚拟环境
环境安装完成后，使用以下命令激活它：
```bash
conda activate MSDiffAD
```

### 3. 验证安装
您可以运行以下命令，检查 PyTorch 2.2.1 是否已成功安装并正常运行：
```bash
python -c "import torch; print(torch.__version__)"
```
