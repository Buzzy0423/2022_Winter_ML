# CNN

相较于多层感知机的优势：**参数少**

适合计算机视觉的网络结构应该有的特性：

1.  *平移不变性*（translation invariance）：不管检测对象出现在图像中的哪个位置，神经网络的前面几层应该对相同的图像区域具有相似的反应，即为“平移不变性”。
2.  *局部性*（locality）：神经网络的前面几层应该只探索输入图像中的局部区域，而不过度在意图像中相隔较远区域的关系，这就是“局部性”原则。最终，可以聚合这些局部特征，以在整个图像级别进行预测。

对应的为卷积神经网络中的卷积核和步长

数学上的卷积：$(f*g)(x)=\int f(z)g(x-z)$

其离散形式：$(f*g)(i)=\sum_a f(a)g(i-z)$

**通道**：图像不仅仅具有长和宽，还有RGB等属性，所以我们通常会采用三维张量，即一系列具有二维向量的通道

因为权重是由学习得到的，所以互相关运算和卷积运算差别不大

### 卷积的反向传播

MaxPooling：<img src="/Users/lee/Library/Application Support/typora-user-images/截屏2022-01-25 上午11.22.43.png" alt="截屏2022-01-25 上午11.22.43" style="zoom:25%;" />

将大小和max值位置还原：<img src="/Users/lee/Library/Application Support/typora-user-images/截屏2022-01-25 上午11.23.12.png" alt="截屏2022-01-25 上午11.23.12" style="zoom:25%;" />

因为除最大值之外的值对下一层的贡献为0，所以可以认为他们通过f(x)=0传递，而最大值以f(x)=x传递，所以最大值的偏导数为1，其余为0

AveragePooling：<img src="/Users/lee/Library/Application Support/typora-user-images/截屏2022-01-25 上午11.28.55.png" alt="截屏2022-01-25 上午11.28.55" style="zoom:25%;" /> 平分回去

卷积层：$\delta^l$的推导，先上公式

<img src="/Users/lee/Library/Application Support/typora-user-images/截屏2022-01-25 下午3.11.36.png" alt="截屏2022-01-25 下午3.11.36" style="zoom:40%;" />

卷积 <img src="/Users/lee/Library/Application Support/typora-user-images/截屏2022-01-25 下午3.12.14.png" alt="截屏2022-01-25 下午3.12.14" style="zoom:40%;" />

展开得<img src="/Users/lee/Library/Application Support/typora-user-images/截屏2022-01-25 下午3.13.03.png" alt="截屏2022-01-25 下午3.13.03" style="zoom:40%;" />

求$a^l$的梯度(链式法则)$\nabla a^l=\frac{\partial J(W,b)}{\partial a^{l-1}}=\frac{\partial J(W,b)}{\partial z}\frac{\partial z^l}{\partial a^{l-1}}=\delta^l\frac{\partial z^l}{\partial a^{l-1}}$

得到<img src="/Users/lee/Library/Application Support/typora-user-images/截屏2022-01-25 下午3.27.30.png" alt="截屏2022-01-25 下午3.27.30" style="zoom:40%;" />

用矩阵形式表示就是<img src="/Users/lee/Library/Application Support/typora-user-images/截屏2022-01-25 下午3.29.36.png" alt="截屏2022-01-25 下午3.29.36" style="zoom:40%;" />

对w反向传播<img src="/Users/lee/Library/Application Support/typora-user-images/截屏2022-01-25 下午4.14.01.png" alt="截屏2022-01-25 下午4.14.01" style="zoom:40%;" />

对b反向传播<img src="/Users/lee/Library/Application Support/typora-user-images/截屏2022-01-25 下午4.14.36.png" alt="截屏2022-01-25 下午4.14.36" style="zoom:40%;" />