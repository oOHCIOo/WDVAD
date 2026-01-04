# 基于联邦学习的视频异常检测

本项目目录专为**UCF 数据集**实验设计（其他数据集实验请切换相应文件夹）。基于 Flower 与 PyTorch 框架，实现分布式异常检测及标签自修正管道。**所有实验操作请先 cd 到本目录（`./WDVAD-ucf-Crime`）进行！**

## 代码结构

* `server.py`：联邦学习**服务器端**入口，负责聚合各客户端的模型更新。
* `client.py`：联邦学习**客户端**入口。每个客户端模拟一个数据孤岛，在其本地数据上进行训练。
* `options.py`：集中式**配置**与**命令行参数**定义文件（服务器与客户端共用）。
* `model.py`：用于异常评分的**神经网络结构**定义。
* `train.py`：每个客户端执行的**本地训练**逻辑。
* `test.py`：每个客户端执行的**本地评估**逻辑。
* `video_dataset_anomaly_balance_uni_sample.py`：**数据加载与预处理**模块。
* `losses.py`：自定义**损失函数**定义。
* `generate_new_labels.py`：基于 **MMD** 的**动态标签更新**核心逻辑及 TS/F1 评估。
* `ablation_study.py`：**消融实验脚本**，用于测试不同 pvalue 阈值下的标签修正性能（**F1 和 TS 计算方式与论文中结果的计算方式一致**）。
* `standalone_eval.py`：**独立评估脚本**，基于已训练的模型检查点（ckpt）进行独立评估（**F1 和 TS 计算方式与 `ablation_study.py` 不同**）。
* `eval.py`：包含 `calculate_f1` 和 `calculate_ts_align` 等评估函数。
* `SALA.py`：实现 **SALA（Self-Adaptive Local Aggregation，自适应本地聚合）** 客户端策略。
* `environment.yml`：**Conda 环境**依赖清单。

## 环境准备与安装

1. **克隆仓库：**

   ```bash
   git clone <your-repo-url>
   cd ./ucf_crime
   ```

2. **创建 Conda 环境：**
   请先确保已安装 Anaconda 或 Miniconda，然后根据给定文件创建环境：

   ```bash
   conda env create -f environment.yml
   ```

3. **激活环境：**

   ```bash
   conda activate FEDKD
   ```

4. **准备数据与预训练模型：**

   * 将你的数据集（**视频特征**与初始 **`.pickle` 标签文件**）放置到 `ucf_dataset` 子目录。
   * 确保为每个客户端准备好**预训练模型权重**（在 `options.py` 的 `pretrained_ckpt_scene_*` 参数中指定路径）。**训练首轮会加载这些模型。**

## 运行方式

该系统采用**客户端—服务器**模式。请先启动**服务器端**，再启动一个或多个**客户端**。

1. **启动服务器：**
   在终端中激活 Conda 环境并运行 `server.py`。可以指定通信轮数与服务器地址。

   ```bash
   # 示例：在 8080 端口启动服务器，进行 5 轮通信
   python server.py --rounds 5 --server_address 0.0.0.0:8080
   ```

   使用 `0.0.0.0` 可使服务器对局域网内其他机器可见。

2. **启动客户端：**
   对每个参与的客户端，在新的终端（或不同机器）中激活环境并运行 `client.py`。**务必为每个客户端提供唯一的 `--cid`（Client ID）**。

   ```bash
   # 机器 1（或终端 1）启动客户端 1
   python client.py --cid 1 --server_address <server_ip>:8080
   
   # 机器 2（或终端 2）启动客户端 2
   python client.py --cid 2 --server_address <server_ip>:8080
   
   # ... 以此类推
   ```

   将 `<server_ip>` 替换为服务器所在机器的实际 IP。如果在同一台机器上运行，可使用 `127.0.0.1`。

3. **监控与结果收集：**

   * 服务器与客户端会在联邦训练过程中**打印日志**。
   * 每个客户端的结果（例如各轮的 **AUC** 指标）会保存到 `result_{dataset_name}` 目录下的 `result_cid_{cid}.txt` 文件中。


## 配置说明

所有运行时参数均可通过命令行指定，定义见 `options.py`。常用参数包括：

* `--cid`（客户端）：客户端唯一 ID。
* `--server_address`（客户端与服务器端）：服务器的 IP 与端口。
* `--rounds`（服务器端）：联邦学习的总通信轮数。
* `--dataset_type`：一般固定为 `ucf_`。
* `--lr`：本地训练的学习率。
* `--t_id`：**教师客户端（teacher clients）ID 列表**。这些客户端**不进行本地训练更新**，只参与知识蒸馏/聚合等（具体行为由实现决定）。

---

如需进一步定制（例如概念漂移场景下的**动态标签刷新**策略、**联邦遗忘**工作流、或 **SALA** 的参数设置），请参考 `generate_new_labels.py` 与 `SALA.py` 中的实现细节，并在 `options.py` 中调整相应超参数。
