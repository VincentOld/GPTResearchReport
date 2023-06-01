# 训练部署自己的ChatGLM-6B

参考链接：[【实战讲解】ChatGLM-6B模型训练完整流程详解](https://mp.weixin.qq.com/s/8gdrmdZfQO_Ji_frXpjc7g)

```
本文实现了基于 P-Tuning v2 的高效参数微调方法，通过实际动手操作，提升对大模型的理解和应用能力。
ChatGLM-6B 是一个开源的、支持中英双语的对话语言模型，基于 General Language Model (GLM) 架构，具有 62 亿参数。 ChatGLM-6B 使用了和 ChatGPT 相似的技术，针对中文问答和对话进行了优化。经过约 1T 标识符的中英双语训练，辅以监督微调、反馈自助、人类反馈强化学习等技术的加持，62 亿参数的 ChatGLM-6B 已经能生成相当符合人类偏好的回答。 
```

---

1. ##### conda环境

```shell
下载miniconda，选择匹配的操作系统的版本，
从官网ttps://docs.conda.io/en/latest/miniconda.html下载。
$ sh Miniconda3-latest-***.sh

构建虚拟环境
$ conda create -n chatglm   --clone base      # 创建新环境
$ source activate chatglm        			  # 激活环境
$ conda env list  							  #显示环境列表
```

##### 2 下载代码&安装依赖

```shell
$ git clone https://github.com/THUDM/ChatGLM-6B.git
$ cd ChatGLM-6B

安装依赖
$ pip install -r requirements.txt
$ pip install rouge_chinese nltk jieba datasets
```

##### 3 模型和数据准备

对于ChatGLM-6B模型的训练，需要准备相应的数据集。使用ADGEN数据集，其任务为根据输入（content）生成一段广告词（summary）。下载ADGEN数据集，从 Google Drive 或者 Tsinghua Cloud 下载处理好的 ADGEN 数据集，将解压后的 AdvertiseGen 目录放到本目录下。

```shell
模型下载, 从 Hugging Face Hub 下载模型需要先安装Git LFS，然后运行
$ yum install git-lfs
$ cd /data/pre_model/chatglm
$ git clone https://huggingface.co/THUDM/chatglm-6b

训练数据下载存放路径：
$ cd /data/train_data/ChatGLM-6B/ptuning 
$ wget https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1

训练数据示例：
$ head train.json
{"content": "类型#裤*版型#宽松*风格#性感*图案#线条*裤型#阔腿裤", "summary": "宽松的阔腿裤这两年真的吸粉不少，明星时尚达人的心头爱。毕竟好穿时尚，谁都能穿出腿长2米的效果宽松的裤腿，当然是遮肉小能手啊。上身随性自然不拘束，面料亲肤舒适贴身体验感棒棒哒。系带部分增加设计看点，还让单品的设计感更强。腿部线条若隐若现的，性感撩人。颜色敲温柔的，与裤子本身所呈现的风格有点反差萌。"}

评估数据示例：
$ head dev.json
{"content": "类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞", "summary": "简约而不简单的牛仔外套，白色的衣身十分百搭。衣身多处有做旧破洞设计，打破单调乏味，增加一丝造型看点。衣身后背处有趣味刺绣装饰，丰富层次感，彰显别样时尚。"}
```

#### 5 训练模型

```shell
模型训练可以使用一些机器学习框架，PyTorch。使用预训练模型来初始化ChatGLM-6B，然后通过P-Tuning v2 的高效参数微调进行训练。需要考虑到训练时间和硬件资源的因素
train.sh 中的 PRE_SEQ_LEN 和 LR 分别是 soft prompt 长度和训练的学习率，可以进行调节以取得最佳的效果。
P-Tuning-v2 方法会冻结全部的模型参数，可通过调整 quantization_bit 来被原始模型的量化等级，不加此选项则默认为 FP16 精度加载。
如果你想要从本地加载模型，可以将 train.sh 中的 THUDM/chatglm-6b 改为你本地的模型路径。
```

```shell
1、训练脚本train.sh如下：
$ cat train.sh
PRE_SEQ_LEN=8
LR=2e-2
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --do_train \
    --train_file AdvertiseGen/train.json \
    --validation_file AdvertiseGen/dev.json \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path ../../../pre_model/chatglm/chatglm-6b \
    --output_dir ./output/adgen-chatglm-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4
    
2、开始训练,后台执行：
$ nohup  sh train.sh &

3、训练结束，查看结果，max_steps 3000，训练40分钟完成 
$ tail nohup.out
{'loss': 7.1214, 'learning_rate': 6.666666666666667e-05, 'epoch': 0.42}
{'loss': 7.088, 'learning_rate': 0.0, 'epoch': 0.42}
{'train_runtime': 2402.1329, 'train_samples_per_second': 19.982, 'train_steps_per_second': 1.249, 'train_loss': 8.597134847005208, 'epoch': 0.42}

***** train metrics *****
  epoch                    =       0.42
  train_loss               =     8.5971
  train_runtime            = 0:40:02.13
  train_samples            =     114599
  train_samples_per_second =     19.982
  train_steps_per_second   =      1.249

4、查看训练生成的模型文件及模型结果
$ ls  /data/train_data/ChatGLM-6B/ptuning/output/adgen-chatglm-6b-pt-8-2e-2
all_results.json  checkpoint-1000  checkpoint-2000  checkpoint-3000  trainer_state.json  train_results.json

$ cat  all_results.json
{
    "epoch": 0.42,
    "train_loss": 8.597134847005208,
    "train_runtime": 2402.1329,
    "train_samples": 114599,
    "train_samples_per_second": 19.982,
    "train_steps_per_second": 1.249
}

$ cat train_results.json
{
    "epoch": 0.42,
    "train_loss": 8.597134847005208,
    "train_runtime": 2402.1329,
    "train_samples": 114599,
    "train_samples_per_second": 19.982,
    "train_steps_per_second": 1.249
}

$ ls  /data/train_data/ChatGLM-6B/ptuning/output/adgen-chatglm-6b-pt-8-2e-2/checkpoint-3000
config.json generation_config.json  modeling_chatglm.py  pytorch_model.bin  rng_state.pth  special_tokens_map.json  tokenizer_config.json  training_args.bin
configuration_chatglm.py  ice_text.model optimizer.pt quantization.py 
1、训练脚本train.sh如下：
$ cat train.sh
PRE_SEQ_LEN=8
LR=2e-2

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --do_train \
    --train_file AdvertiseGen/train.json \
    --validation_file AdvertiseGen/dev.json \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path ../../../pre_model/chatglm/chatglm-6b \
    --output_dir ./output/adgen-chatglm-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4
    
2、开始训练,后台执行：
$ nohup  sh train.sh &

3、训练结束，查看结果，max_steps 3000，训练40分钟完成 
$ tail nohup.out
{'loss': 7.1214, 'learning_rate': 6.666666666666667e-05, 'epoch': 0.42}
{'loss': 7.088, 'learning_rate': 0.0, 'epoch': 0.42}
{'train_runtime': 2402.1329, 'train_samples_per_second': 19.982, 'train_steps_per_second': 1.249, 'train_loss': 8.597134847005208, 'epoch': 0.42}
***** train metrics *****
  epoch                    =       0.42
  train_loss               =     8.5971
  train_runtime            = 0:40:02.13
  train_samples            =     114599
  train_samples_per_second =     19.982
  train_steps_per_second   =      1.249

4、查看训练生成的模型文件及模型结果
$ ls  /data/train_data/ChatGLM-6B/ptuning/output/adgen-chatglm-6b-pt-8-2e-2
all_results.json  checkpoint-1000  checkpoint-2000  checkpoint-3000  trainer_state.json  train_results.json

$ cat  all_results.json
{
    "epoch": 0.42,
    "train_loss": 8.597134847005208,
    "train_runtime": 2402.1329,
    "train_samples": 114599,
    "train_samples_per_second": 19.982,
    "train_steps_per_second": 1.249
}

$ cat train_results.json
{
    "epoch": 0.42,
    "train_loss": 8.597134847005208,
    "train_runtime": 2402.1329,
    "train_samples": 114599,
    "train_samples_per_second": 19.982,
    "train_steps_per_second": 1.249
}

$ ls  /data/train_data/ChatGLM-6B/ptuning/output/adgen-chatglm-6b-pt-8-2e-2/checkpoint-3000
config.json generation_config.json  modeling_chatglm.py  pytorch_model.bin  rng_state.pth  special_tokens_map.json  tokenizer_config.json  training_args.bin
configuration_chatglm.py  ice_text.model optimizer.pt  quantization.py scheduler.pt   tokenization_chatglm.py  trainer_state.json
```

#### 6 模型评估

```shell
训练完成后，需要进行模型评估和调整。可以使用一些指标来评估模型的性能。
将 evaluate.sh 中的 CHECKPOINT 更改为训练时保存的 checkpoint 名称，
运行以下指令进行模型推理和评测：bash evaluate.sh。
生成的结果保存在 ./output/adgen-chatglm-6b-pt-8-1e-2/generated_predictions.txt。
```

```shell
1、模型文件
/data/train_data/ChatGLM-6B/ptuning/output/adgen-chatglm-6b-pt-8-2e-2/checkpoint-3000

2、执行评测脚本， sh evaluate.sh
评测样本示例：
Quantized to 4 bit
input_ids [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 65421, 61, 75898, 32, 68554, 61, 77257, 64555, 32, 65107, 61, 66268, 32, 65347, 61, 71689, 32, 69768, 61, 85428, 32, 65173, 73942, 61, 70984, 32, 65173, 70936, 61, 64703, 65509, 130001, 130004]

inputs 类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞
label_ids [5, 71689, 66561, 67061, 77257, 70984, 6, 72194, 65173, 64290, 64622, 81549, 63823, 65173, 64290, 83343, 63832, 63912, 65209, 64703, 65509, 64051, 6, 69418, 78598, 87019, 6, 64257, 71319, 66069, 74197, 63823, 65173, 72265, 64880, 64131, 63832, 73416, 85428, 66261, 6, 65594, 87834, 6, 73412, 105145, 65388, 63823, 130001, 130004]

labels 简约而不简单的牛仔外套,白色的衣身十分百搭。衣身多处有做旧破洞设计,打破单调乏味,增加一丝造型看点。衣身后背处有趣味刺绣装饰,丰富层次感,彰显别样时尚。

3、查看评测脚本
$ cat evaluate.sh
PRE_SEQ_LEN=8
CHECKPOINT=adgen-chatglm-6b-pt-8-2e-2
STEP=3000
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --do_predict \
    --validation_file AdvertiseGen/dev.json \
    --test_file AdvertiseGen/dev.json \
    --overwrite_cache \
    --prompt_column content \
    --response_column summary \
    --model_name_or_path ../../../pre_model/chatglm/chatglm-6b \
    --ptuning_checkpoint ./output/$CHECKPOINT/checkpoint-$STEP \
    --output_dir ./output/$CHECKPOINT \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4

4、查看评测结果：
$ cd /data/train_data/ChatGLM-6B/ptuning/output/adgen-chatglm-6b-pt-8-2e-2
$ cat generated_predictions.txt
{"labels": "简约而不简单的牛仔外套,白色的衣身十分百搭。衣身多处有做旧破洞设计,打破单调乏味,增加一丝造型看点。衣身后背处有趣味刺绣装饰,丰富层次感,彰显别样时尚。", "predict": "修身修身的UNK>,这款UNK>感感,加上加上性感的面料,展现气质,整体整体整体性感,展现彰显修饰修饰的UNK>。加上加上气质,让简约设计,穿着穿着设计,搭配搭配。"}
```

#### 7 模型部署

```shell
当模型训练和评估完成后，可以将它部署到适当的平台上。
在部署时，可以考虑到模型的可用性、可扩展性和性能等因素。
本次只是演示，使用部署脚本加载本地模型,并加载新的Checkpoint。
注意需要将 pre_seq_len 改成你训练时的实际值，具体部署验证演示代码如下：

1、执行部署脚本
$ python deploy.py

2、查看部署脚本
$ cat deploy.py

import os
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
model_name = "../../../pre_model/chatglm/chatglm-6b" 					# 模型名 或 模型路径
checkpoint_path = "./output/adgen-chatglm-6b-pt-8-2e-2/checkpoint-3000" # 模型checkpoint路径
pre_seq_len = 8 														# 模型前缀长度 跟你训练的PRE_SEQ_LEN一致
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
config.pre_seq_len = pre_seq_len
model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
prefix_state_dict = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"))
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
model.half().cuda()
print(model.chat(tokenizer, "你是谁"))
```

 
