---
layout:     post
title:      Shift from CPU to GPU
subtitle:   PyTorch Notes
date:       2019-05-25
author:     viewsetting
header-img: img/post-adjani.jpg
catalog: true
tags:
    - Python
    - PyTorch
    - Notes
    - Deep Learning
---

# Check CUDA status

```python
if torch.cuda.is_available():
    print("CUDA is READY")
```

![1558779896716](/img/2019-05-25/1558779896716.png)

# Model

```python
model = Seq2Seq()
model.cuda(0)
```



# Variable

```python
hidden = Variable(torch.zeros(1, 1, n_hidden)) # Previous
hidden = Variable(torch.zeros(1, 1, n_hidden)).cuda()  # Now
```

# Input & Output Tensor

```python
input_batch, output_batch, target_batch = make_batch(seq_data)
input_batch=input_batch.cuda()
output_batch=output_batch.cuda()
target_batch=target_batch.cuda()
```

