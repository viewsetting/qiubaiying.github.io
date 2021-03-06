---
layout:     post
title:      pytorch-transformers本地加载
subtitle:   备忘录
date:       2019-10-17
author:     viewsetting
header-img: img/Pytorch-banner.png
catalog: true
tags:
    - NLP
    - Pytorch_Transformers
    - Notes
    - PyTorch
---

本文将以Pytorch-transformers库中的Bert为例。

## 从本地文件中加载模型

### 下载至本地

BERT的预训练模型下载地址如下:

<https://github.com/huggingface/transformers/blob/master/transformers/modeling_bert.py> line 37

```Python
BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    'bert-base-german-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
    'bert-base-german-dbmdz-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-pytorch_model.bin",
    'bert-base-german-dbmdz-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-pytorch_model.bin",
}
```

选择想要下载的模型对应的地址复制到浏览器，打开地址即可开始下载。

![1](/img/2019-10-17/1.png)

然后选择一个文件夹放入下载完的bin模型备用。

同理，将上面地址中的`pytorch-model.bin`字样均替换为`config.json`，我们就得到了这些模型的config脚本，打开后是这样。

`bert-base-uncased-config.json`

```json
{
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "max_position_embeddings": 512,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "type_vocab_size": 2,
  "vocab_size": 30522
}
```

当然这个文件你也可以自己写一个。

### 重命名

将上述两个文件保存至文件夹后，将.bin后缀的模型参数文件改为`pytorch_model.bin`，而另一个json文件则重命名为config.json。

![3](/img/2019-10-17/3.png)

原因见源码：<https://github.com/huggingface/transformers/blob/master/transformers/file_utils.py> line: 70

![2](/img/2019-10-17/2.png)

由此也可见tensorflow模型的参数文件的默认命名，这样我们就可以通过传递文件夹路径的方式直接调用BertModel。

BertModel类是继承自PreTrainedBert类，由此也有了从与训练调入这一函数，我们可以直接传入文件夹路径：

```python
from pytorch_transformers import BertModel
model_path = '/add/your/path/here'
bert_uncased = BertModel.from_pretrained(model_path)
```

到这里，就可以快速调入本地的模型文件了，有效解决了服务器断网问题！逃）

## 从本地加载Tokenizer

我们讲第一部分中的地址搬过来，将最后的文件名换成`vocab.txt`，我们就又得到了官方模型对应的词汇表文件。如：<https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt>这个就是ber-base-uncased的vocab文件，打开后长这样：

![4](/img/2019-10-17/4.png)

然后我们复制到一个文件夹中，并重命名为`vocab.txt`，当然你也可以不重命名，区别在于：重命名后可以直接制定文件夹作为参数初始化`BertTokenizer`，这样就会自动读取文件夹中的`vocab.txt`作为词汇表，而且还有一些控制的json文件也可以一起读入。命名规则如下：<https://github.com/huggingface/transformers/blob/master/transformers/tokenization_utils.py> line: 35

```python
SPECIAL_TOKENS_MAP_FILE = 'special_tokens_map.json'
ADDED_TOKENS_FILE = 'added_tokens.json'
TOKENIZER_CONFIG_FILE = 'tokenizer_config.json'
```

最后我们可以这样调用测试：

```python
from pytorch_transformers import BertTokenizer
vocab_dir = '/your/vocab/dir/here'
vocab_txt_name = 'vocab.txt'
tokenizer = BertTokenizer.from_pretrained(vocab_dir + vocab_txt_name)
# or： tokenizer = BertTokenizer.from_pretrained(vocab_dir)
```



第一种调入就是直接载入词汇表，而第二种则是需要在文件夹中将词汇表重命名为`vocab.txt`，同时可以放入`special_tokens_map.json`等json文件提前配置。
