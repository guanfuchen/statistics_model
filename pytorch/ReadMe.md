# PyTorch

PyTorch使用教程，记录常用模块使用示例，其中UnSort文件夹存放细小模块的脚本。


---
## 功能集合

---
## Transforms
Transforms用来对输入数据转换为最终的数据，类似于包装了大量预先的预处理方法，下面是图像中常用的Transforms变换。

```python
transform = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
])
```

---
## 网络层
网络层包含Module的模块。

---
### Embedding

Embedding是一个保存了固定字典和大小的简单查找表。这个模块常用来保存词嵌入和用下标检索它们。模块的输入是一个下标的列表，输出是对应的词嵌入。

---
## 参考资料

[torch.nn中文文档](https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-nn/)

[Source code for torch.nn.modules.sparse中文文档](http://pytorch.apachecn.org/cn/0.3.0/_modules/torch/nn/modules/sparse.html)

[torch.nn英文文档](https://pytorch.org/docs/master/nn.html)

[torch vision](https://github.com/pytorch/vision#models) 其中实现了大量的视觉模型。
