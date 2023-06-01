# 类ChatGPT的部署与微调(上)

相关链接：[(77条消息) 类ChatGPT的部署与微调(上)：从LLaMA、Alpaca/Vicuna/BELLE、中文版](https://blog.csdn.net/v_JULY_v/article/details/129709105)

---

##### 一、**LLaMA的代码解读：**RMSNorm/SwiGLU/RoPE/Transformer

1. 概述：LLaMA只使用公开的数据(总计1.4T即**1,400GB的token**，其中CommonCrawl的数据占比67%，C4数据占比15%，Github Wikipedia Books这三项数据均各自占比4.5%，ArXiv占比2.5%，StackExchange占比2%)

2. 项目环境依赖：torch、fairscale、fire、sentencepiece

   ```java
   torch
   fairscale			fairscale是用来做GPU分布的，一般是当使用DDP仍然遇到超显存的问题时使用fairscale
   fire				fire是一个命令行工具，用或者不用他都可以
   sentencepiece		sentencepiece是用于tokenizer的工具包
   ```

3. ##### Tokenizer：调用SentencePieceProcessor进行tokenize

```PYTHON
class Tokenizer:
    def __init__(self, model_path: str):
        # 加载tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        #assert 是一个断言语句，用于检查特定条件是否为真。如果条件为假，则会引发 AssertionError 异常。
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()
 
    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t
 
    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)
```

4. ##### RMSNorm：对每个Transformer子层的输入进行归一化

（1）概述：对每个transformer子层的输入进行归一化，而不是对输出进行归一化

（2）RMSNorm：与layerNorm相比，RMS Norm的主要区别在于去掉了减去均值的部分(re-centering)，只保留方差部分(re-scaling)

```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        // eps防止取倒数之后分母为0
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
 
    // x是输入
    def _norm(self, x):
        // torch.rsqrt是开平方并取倒数
        // x.pow(2)是平方
        / mean(-1)是在最后一个维度(即hidden特征维度)上取平均
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
 
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        // weight是末尾乘的可训练参数，即gi
        return output * self.weight
```

（3）SwiGLU替代ReLU：用SwiGLU替代ReLU，在维度上使用维度是2/3*4d，而不是PaLM中的4d

