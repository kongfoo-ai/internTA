# internTA: 基于InternLM2大模型的《合成生物学》助教

<div align="center"><img src="./demo.gif" width="350"></div>

## 摘要
代码仓库：[[GitHub]](https://github.com/kongfoo-ai/internTA)

模型仓库：[[OpenXLab]](https://openxlab.org.cn/models/detail/Kongfoo_EC/internTA)

演示视频：[[Google]](https://www.bilibili.com/video/BV1RK421s7dm/)

在线体验Demo：[[GPUShare]](http://i-2.gpushare.com:50259/)

## 背景

从人造肉、人造蛋白，到基因编辑技术 CRISPR-Cas9，近年来， 合成生物学在各个领域落地开花，引领着一场被称为“第三次生物技术革命”的科学浪潮。而合成生物学知识的普及，面临着以下的挑战：

一方面，合成生物学是一门融合了生物学、化学、工程学、计算机科学等领域知识的交叉学科，许多尖端技术都对合成生物学的进步起到了不可或缺的推动作用。

另一方面，我国在生物科技领域依旧和先进发达国家有着一定差距，处于追赶状态。特别是具有跨领域知识储备以及丰富实践经验的教师人才十分匮乏。

为弥补上述不足，我们开发了基于InternLM2大语言模型的《合成生物学》助教InternTA，旨在帮助学生更好地学习《合成生物学》这门课程。通过提供关键词和思路，并指出教材中相关的章节，我们希望开发的InternTA可以引导学生自行思考，从而完成对合成生物学知识的高质量学习，达到“授人以渔”的目的。


## 介绍

InternTA使用情景模拟器生成的情景数据作为微调数据集，使用[Xtuner](https://github.com/InternLM/xtuner)工具对[InternLM2-Chat-1.8B-SFT](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-1_8b-sft/summary)基础模型进行微调，使用streamlit作为框架开发网页端DEMO应用。

InternTA的实现原理如下图所示：

<div align="center"><img src="./internTA.png" width="350"></div>

其中微调数据准备是最为关键的环节之一。我们准备的微调训练数据包含两类：直接问答数据和引导式问答数据。微调数据准备的步骤如下图所示：

<div align="center"><img src="./data.png" width="350"></div>


> - 首先，我们整理出问题库，主要包括三类：课后思考题、附录关键名词和基础概念知识。我们根据这些问题在《合成生物学》教材中检索相应的答案。
> - 紧接着，我们整理检索到的答案，形成助教可以使用的回答数据库。对于关键名词和基础概念知识，我们会直接提供答案与用户交流。
> - 对于《合成生物学》教材的课后思考题，我们使用更大参数规模的大语言模型（如GPT4o）将正确答案改写成引导式回答形式，避免直接告知用户标准答案。


## 快速体验

**在线体验地址**：[[GPUShare]](http://i-2.gpushare.com:50259/)

**本地体验方法**(8G显存以上NVIDIA GPU)：

```sh
# 克隆仓库
git clone https://github.com/BestAnHongjun/InternDog.git

# 进入项目目录
cd InternDog

# 安装依赖
pip install -r requirements.txt

# 运行网页版demo
python app.py

# 运行终端demo
python app_cli.py
```

## 使用教程

### 1.训练数据生成

使用本项目开发的场景模拟器生成训练数据。

```sh
cd data
python gen_all_data.py
```

### 2.模型微调

安装依赖项。

```sh
pip install -r requirements_all.txt
```

下载[InternLM2-Chat-1.8B-SFT](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-1_8b-sft/summary)大模型。

```sh
python python fine-tune/download_pretrain_model.py 
```

基于xtuner微调模型。

```sh
xtuner train ./fine-tune/internlm2_1_8b_qlora_lift_e3.py --deepspeed deepspeed_zero2
```

生成Adapter。

```sh
# 注意修改.sh文件第六行模型文件路径
./tools/1.convert_model.sh
```

合并Adapter。

```sh
# 注意修改模型路径
./tools/2.merge_model.sh
```

### 3.模型量化

W4A16量化模型。

```sh
# 注意修改模型路径
./tools/3.quantize_model.sh
```

转化为TurboMind模型。

```sh
# 注意修改模型路径
./tools/4.turbomind_model.sh
```

## 特别鸣谢

- [InternLM2-Chat-1.8B-SFT](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-1_8b-sft/summary)
- [ZhangsanPufa](https://github.com/AllYoung/InternLM4Law)
- [Xtuner](https://github.com/InternLM/xtuner)
