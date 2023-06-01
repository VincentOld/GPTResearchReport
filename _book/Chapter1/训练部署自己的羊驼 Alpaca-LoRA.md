# 训练部署自己的羊驼 Alpaca-LoRA

参考链接：[Gpt进阶（二）:训练部署自己的ChatGPT模型(羊驼 Alpaca-LoRA）](https://mp.weixin.qq.com/s/8gdrmdZfQO_Ji_frXpjc7g)

```
Alpaca-LoRA(羊驼模型):
	github.com/tloen/alpaca-lora
Chinese-alpaca-lora（开源的中文数据集）
	github.com/LC1332/Chinese-alpaca-lora/blob/main/data/trans_chinese_alpaca_data.json
lora:大型语言模型的低秩适配器;简单来说就是微调模型的另一种方式，来调试模型在具体场景下的准确度；假设模型适应过程中的权重变化也	  具有较低的“内在秩”，从而提出了低秩适应(low - rank adaptation, LoRA)方法。
	论文地址（github）:https://github.com/microsoft/LoRA
```

---

1. ##### python环境

```shell
#网址：https://conda.io/en/latest/miniconda.html
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh #下载脚本
sh Miniconda3-py39_4.12.0-Linux-x86_64.sh 	# 执行
~/miniconda3/bin/conda init 				#初始化Shell，以便直接运行conda
conda create --name alpaca python=3.9 		#关启shell，创建虚拟环境
conda activate alpaca 						#激活 
```

##### 2 下载羊驼代码

```shell
git clone https://github.com/tloen/alpaca-lora.git #下载源代码
cd alpaca-lora
pip install -r requirements.txt 				   #安装依赖
												   #测试pytorch
import torch
torch.cuda.is_available()
```

##### 3 准备数据集

构造指令数据集结构，类似于instruct的方法，可参考使用开源的中文数据集：Chinese-alpaca-lora，链接开头已经给出，下载后放到项目根目录下。

<img src="pics/训练部署自己的羊驼 Alpaca-LoRA/image-20230506104702692.png" alt="image-20230506104702692" style="zoom:43%;" /> 

##### 4 下载LLaMA基础模型：下载完成后放到根目录下/llama-7b-hf

```
huggingface.co/decapoda-research/llama-7b-hf/tree/main
```

#### 5 训练模型

```shell
python finetune.py \
    --base_model 'llama-7b-hf' \
    --data_path './trans_chinese_alpaca_data.json' \
    --output_dir './lora-alpaca-zh'
```

```
其他具体参数可以git链接
模型训练后，lora-alpaca-zh 下就有模型生成了
```

#### 6 模型推理

```shell
Inference (generate.py)
python generate.py \
    --load_8bit \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights 'tloen/alpaca-lora-7b'
```

#### 7 云端部署

```
使用kaggle部署模型，访问web交互
地址：www.kaggle.com/models
```

<img src="pics/训练部署自己的羊驼 Alpaca-LoRA/image-20230506104856586.png" alt="image-20230506104856586" style="zoom:33%;" /> 
