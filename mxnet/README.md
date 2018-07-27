# MXNet

MXNet适合于分布式GPU训练，具体优点如下所示：
- 设备放置：使用MXNet，可以轻松指定每个数据结构应存在的位置。 
- 多GPU培训：MXNet可以通过可用GPU的数量轻松扩展计算。 
- 自动区分：MXNet自动执行曾经陷入神经网络研究的衍生计算。 
- 优化的预定义图层：虽然您可以在MXNet中编写自己的图层，但预定义的图层会针对速度进行优化，优于竞争库。

下述示例计算和相应图示：

```python
hidden_linear = mx.sym.dot(X, W)
hidden_activation = mx.sym.tanh(hidden_linear)
```

![](https://raw.githubusercontent.com/kevinthesun/web-data/master/mxnet/get-started/architecture.png)

```python
# 上述图示的示例代码
import mxnet.ndarray as nd

X  = nd.zeros((10000, 40000), mx.cpu(0))           #Allocate an array to store 1000 datapoints (of 40k dimensions) that lives on the CPU
W1 = nd.zeros(shape=(40000, 1024), mx.gpu(0))      #Allocate a 40k x 1024 weight matrix on GPU for the 1st layer of the net
W2 = nd.zeros(shape=(1024, 10), mx.gpu(0))         #Allocate a 1024 x 1024 weight matrix on GPU for the 2nd layer of the net

# 定义具体的计算设备GPU或者CPU
with mx.Context(mx.gpu()):          # Absent this statement, by default, MXNet will execute on CPU
    h = nd.tanh(nd.dot(X, W1))
    y = nd.sigmoid(nd.dot(h, W2))
```

MXNet支持两种编程风格：命令式编程（由NDArray API支持）和符号编程（由Symbol API支持）。简而言之，命令式编程是您可能最熟悉的风格。这里，如果A和B是表示矩阵的变量，则C = A + B是一段代码，在执行时对A和B引用的值求和并将它们的和C存储在新变量中。另一方面，符号编程允许通过计算图抽象地定义函数。在符号风格中，我们首先用占位符值表达复杂函数。然后，我们可以通过将它们绑定到实际值来执行这些功能。

## 符号编程

除了通过NDArray提供快速数学运算外，MXNet还提供了一个通过计算图抽象地定义运算的接口。使用mxnet.symbol，我们根据占位符抽象地定义操作。例如，在下面的代码中，a和b表示将在运行时提供的实际值。当我们调用c = a + b时，不执行数值计算。此操作只是构建一个图形，用于定义a，b和c之间的关系。为了执行实际计算，我们需要将c绑定到实际值。

```python
a = mx.sym.Variable('a')
b = mx.sym.Variable('b')
c = a + b
executor = c.bind(mx.cpu(), {'a': X, 'b': Y}) # 类似于Tensorflow中的feed_dict
result = executor.forward()
```

## 使用MXNet图层构建模型

```python
data = mx.symbol.Variable('data')
fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=num_classes)
mlp  = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')
```

## Gluon和Module的区别

Gluon软件包是MXNet的高级接口，易于使用，同时保持低级API的大部分灵活性。 Gluon支持命令式和符号式编程，可以很容易地在Python中强制训练复杂模型，然后使用C ++和Scala中的符号图进行部署。 基于Gluon API规范，Apache MXNet中的Gluon API为深度学习提供了清晰，简洁和简单的API。它可以轻松地在不牺牲训练速度的情况下进行原型，构建和训练深度学习模型，具有如下优点，具体查看[Gluon Package](https://mxnet.incubator.apache.org/api/python/gluon/gluon.html)：
- 简单易懂的代码：Gluon提供了一整套即插即用的神经网络构建模块，包括预定义的层，优化器和初始化器。 
- 灵活，势在必行的结构：Gluon不需要严格定义神经网络模型，而是将训练算法和模型更紧密地结合在一起，以便在开发过程中提供灵活性。 
- 动态图：Gluon使开发人员能够定义动态的神经网络模型，这意味着它们可以动态构建，使用任何结构，并使用Python的任何本机控制流。 
- 高性能：Gluon提供上述所有优势，而不会影响底层引擎提供的培训速度。

---
## 数据操作
在深度学习中，我们通常会频繁地对数据进行操作。作为动手学深度学习的基础，本节将介绍如何对内存中的数据进行操作。
在MXNet 中，NDArray 是存储和变换数据的主要⼯具。如果你之前用过NumPy，你会发现NDArray和NumPy 的多维数组非常类似。然而，NDArray 提供诸如GPU 计算和自动求导在内的更多功能。这些都使得NDArray 更加适合深度学习。

### numpy和mxnet数据转换

```python
# Create a numpy array from an mxnet NDArray
A_np = np.array([[0,1,2,3,4],[5,6,7,8,9]])
A_nd = nd.array(A)  

# Convert back to a numpy array
A2_np = A_nd.asnumpy()
```


---
## 参考资料
- [Deep Learning - The Straight Dope](https://gluon.mxnet.io/)
- [Apache MXNet (Incubating)](https://mxnet.apache.org/) mxnet apache孵化项目
- [动手学深度学习](https://zh.gluon.ai/) 用 Apache MXNet (incubating) 的最新 Gluon 接口来演示如何从零开始实现深度学习的各个算法。
- [《动手学深度学习》第一季课程汇总](https://discuss.gluon.ai/t/topic/753) MXNet视频课程学习，中文教程，非常值得参考，对应的pdf教程[动手学深度学习](https://zh.gluon.ai/gluon_tutorials_zh.pdf)。
- [Why MXNet?](https://mxnet.incubator.apache.org/faq/why_mxnet.html) 官方非常好的科普文章。
- [MXNet - Python API](https://mxnet.incubator.apache.org/api/python/index.html) 其中介绍了MXNet Gluon API和Module API这两种API接口的区别以及相关的Python API。
- [MXNet Tutorials](https://mxnet.incubator.apache.org/tutorials/index.html) MXNet示例教程。