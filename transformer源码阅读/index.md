# transformer源码阅读


本文介绍 Tranformer 的代码。

## 模型结构

Encoder 将输入序列$(x_{1},\cdots,x_{n})$ 映射成一个连续的序列$z = (z_{1},\cdots,z_{n})$。而 Decoder 根据$z$来解码得到输出序列$(y_{1},\cdots,y_{m})$。Decoder 是自回归的(auto-regressive)--它会把前一个时刻的输出作为当前时刻的输入。Encoder-Decoder 结构模型的代码如下：

```python
class EncoderDecoder(nn.Module):
	"""
	标准的Encoder-Decoder架构。
	"""
	def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
		super(EncoderDecoder,self).__init__()
		self.encoder = encoder
		self.decoder = decoder
		# 源语言和目标语言的embedding
		self.src_embed = src_embed
		self.tgt_embed = tgt_embed
		# generator主要是根据Decoder的隐状态输出当前时刻的词(单个词)
		# 基本的实现就是隐状态输入一个全连接层，然后接一个softmax变成概率
		self.generator = generator

	def forward(self, src, tgt, src_mask, tgt_mask):
		# 首先调用encode方法对输入进行编码，然后调用decode方法进行解码
		return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

	def encode(self, src, src_mask):
		# 调用self.encoder函数
		return self.encoder(self.src_embed(src), src_mask)

	def decode(self, memory, src_mask, tgt, tgt_mask):
		# 调用self.decoder函数 注意⚠️：这里定义的memery是encoder的输出结果。
		return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

```

EncoderDecoder 定义了一种通用的 Encoder-Decoder 架构，具体的 Encoder、Decoder、src_embed、target_embed 和 generator 都是构造函数传入的参数。这样我们做实验更换不同的组件就会更加方便。

```python
class Generator(nn.Module):
	def __init__(self, d_model, vocab):
		super(Generator, self).__init__()
		# d_model是Decoder输出的大小，vocab是词典的大小
		self.proj = nn.Linear(d_model, vocab)

	def forward(self, x):
		return F.log_softmax(self.proj(x), dim=-1)
```

注意 ⚠️：Generator 返回的是 softmax 的 log 值。在 pytorch 中为了计算交叉熵损失，有两种方法。第一种方法就是 nn.CrossEntropyLoss(),一种是使用 NLLLoss()。第一种方法更加容易懂，但是在很多开源代码里第二种更常见。

> CrossEntropyLoss:

```python
criterion = nn.CrossEntropyLoss()

x = torch.randn(1,5) # 服从0-1的正太分布。
y = torch.empty(1, dtype = torch.long).random_(5)

loss = criterion(x,y)
```

比如上面的代码，假设是 5 分类问题，`x`表示模型的输出 logits(batch=1)，而 y 是真实分类的下标(0-4)。实际的计算过程为：$loss = -\sum^{5}_{i=1}y_{i}log(softmax(x_{i}))$。

> NLLLoss(Negative Log Likelihood Loss)是计算负 log 似然损失。它输入的 x 是 log_softmax 之后的结果（长度为 5 的数组），y 是真实分类（0-4），输出就是 x[y]。因此代码为：

```python
m = F.log_softmax(x, dim=1)
criterion = nn.NLLLoss()
x = torch.randn(1, 5)
y = torch.empty(1, dtype = torch.long).random_(5)
loss = criterion(m(x), y)
```

Transformer 模型也是遵循上面的架构，只不过它的 Encoder 是 N(6)个 EncoderLayer 组成，每个 EncoderLayer 包含一个 Self-Attention SubLayer 层和一个全连接 SubLayer 层。而它的 Decoder 也是 N(6)个 DecoderLayer 组成，每个 DecoderLayer 包含一个 Self-Attention SubLayer 层、Attention SubLayer 层和全连接 SubLayer 层。如下图所示。

![transformer的结构图](http://fancyerii.github.io/img/transformer_codes/the-annotated-transformer_14_0.png)

## Encoder 和 Decoder Stack

前面说了 Encoder 和 Decoder 都是由 N 个相同结构的 Layer 堆积(stack)而成。因此我们首先定义 clones 函数，用于克隆相同的 SubLayer。

```python
def clones(module, N):
	# 克隆N个完全相同的SubLayer，使用了copy.deepcopy
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
```

这里使用了 nn.ModuleList, ModuleList 就像一个普通的 Python 的 List，我们可以使用下标来访问它，它的好处是传入的 ModuleList 的所有 Module 都会注册的 PyTorch 里，这样 Optimizer 就能找到这里面的参数，从而能够用梯度下降更新这些参数。但是 nn.ModuleList 并不是 Module（的子类），因此它没有 forward 等方法，我们通常把它放到某个 Module 里。接下来定义 Encoder:

```python
class Encoder(nn.Module):
	# Encoder是N个EncoderLayer的stack
	def __init__(self, layer, N):
		super(Encoder, self).__init__()
		# layer是一个SubLayer，我们clone N个
		self.layers = clones(layer, N)
		# 再加一个LayerNorm层
		self.norm = LayerNorm(layer.size)

	def forward(self, x, mask):
		# 逐层进行处理
		for layer in self.layers:
			x = layer(x, mask)
		# 最后进行LayerNorm
		return self.norm(x)
```

Encoder 就是 N 个 SubLayer 的 stack，最后加上一个 LayerNorm。我们来看 LayerNorm:

```python
class LayerNorm(nn.Module):
	def __init__(self, features, eps=1e-6):
		super(LayerNorm, self).__init__()
		self.a_2 = nn.Parameter(torch.ones(features))
		self.b_2. = nn.Parameter(torch.zeros(feagures))
		self.eps = eps
	def forward(self, x):
		mean = x.mean(-1, keepdim = True)
		std = x.std(-1, keepdim = True)
		return self.a_2 * (x - mean)/(std+self.eps) + self.b_2
```

LayerNorm：假设数据为[batch_size, unit, 1, features]，这里是对整个样本进行 normalization。这里的 Layer Normalization 不是 Batch Normalization。

```
x -> attention(x) -> x+self-attention(x)[残差] -> layernorm(x+self-attention(x)) => y

y -> dense(y) -> y+dense(y) -> layernorm(y+dense(y)) => z(输入下一层)

```

这里稍微做了一点修改， 在 self-attention 和 dense 之后加了一个 dropout 层。另一个不同支持就是把 layernorm 层放到前面了。这里的模型为：

```
x -> layernorm(x) -> attention(layernorm(x)) -> a + attention(layernorm(x)) => y
y -> layernorm(y) -> dense(layernorm(y)) -> y+dense(layernorm(y))
```

原始论文的 layernorm 放在最后；而这里把它放在最前面并且在 Encoder 的最后在加了一个 layernorm。这里的实现和论文的实现基本是一致的，只是给最底层的输入 x 多做了一个 LayerNorm，而原始论文是没有的。下面是 Encoder 中的 forward 方法，这样比读者可能会比较清楚为什么 N 个 EncoderLayer 处理完成后还需要一个 LayerNorm。

```python
def forward(self, x, mask):
	for layer in self.layers:
		x = layer(x, mask)
	return self.norm(x)
```

不管是 Self-Attention 还是全连接层，都首先是 LayerNorm，然后是 Self-Attention/Dense，然后是 Dropout，最好是残差连接。

```python

class SublayerConnection(nn.Module):
	"""
	LayerNorm+sublayer(Self-Attention/Dense) + dropout + 残差连接
	为了简单，把LayerNorm放到了前面，这和原始论文稍有不同，原始论文LayerNorm在最后。
	"""
	def __init__(self, size, dropout):
		supper(SublayerConnection, self).__init__()
		self.norm = LayerNorm(size)
		self.dropout = nn.Droupout(dropout)
	def forward(self, x, sublayer):
		# sublayer是传入的参数,之后进行残差连接
		return x+self.dropout(sublayer(self.norm(x)))
```

Self-Attention 或者 Dense 并不在这里进行构造，而是放在了 EncoderLayer 里，在 forward 的时候由 EncoderLayer 传入。这样的好处是更加通用，比如 Decoder 也是类似的需要在 Self-Attention、Attention 或者 Dense 前面加上 LayerNorm 和 Dropout 以及残差连接，我们就可以复用代码。但是这里要求传入的 sublayer 可以使用一个参数来调用的函数。

```python
class EncoderLayer(nn.Module):
	# EncoderLayer由self-attn和feed_forward组成
	def __init__(self, size, self_attn, feed_forward, dropout):
		super(EncoderLayer, self).__init__()
		self.size = size
		self.self_attn = self_attn
		self.feed_forward = feed_forward
		self.sublayer = clones(SublayerConnection(size, dropout), 2)
	def forward(self, x, mask):
		x = self.sublayer[0](x, lambda x:self.self_attn(x,x,x,mask))
		return self.sublayer[1](x, self.feed_forward)
```

为了复用，这里的 self_attn 层和 feed_forward 层也是传入的参数，这里只构造两个 SublayerConnection。forward 调用 sublayer[0](这是SubLayerConnection对象)的**call**方法，最终会调到它的 forward 方法，而这个方法需要两个参数，一个是输入 Tensor， 一个是一个 callable, 并且这个 callable 可以用一个参数来调用。而 self_attn 函数需要 4 个参数（Query 的输入，key 的输入，Value 的输入和 Mask），因此这里我们使用 lambda 的技巧把它变成一个参数 x 的函数(mask 可以堪称已知的数)。因为 lambda 的形参也叫 x.

> Decoder

```python
class Decoder(nn.Module):
	def __init__(self, layer, N):
		super(Decoder, self).__init__()
		self.layers = clones(layer, N)
		self.norm = LayerNorm(layer.size)

	def forward(self, x, memory, src_mask, tgt_mask):
		for layer in self.layers:
			x = layer(x, memory, src_mask, tgt_mask)
		return self.norm(x)
```

Decoder 也是 N 个 DecoderLayer 的 stack，参数 layer 是 DecoderLayer，它也是一个 callable，最终**call**会调用 DecoderLayer.forward 方法，这个方法需要 4 个参数，输入 x, Encoder 层的输出 memory, 输入 Encoder 的 Mask(src_mask)和输入 Decoder 的 Mask(tgt_mask)。所有这里的 Decoder 的 forward 也需要 4 个参数。

```python
class DecoderLayer(nn.Module):
	# Decoder包括self-attn, src-attn, feed_forward
	def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
		super(DecoderLayer, self).__init__()
		self.size = size
		self.self_attn = self_attn
		self.src_attn = src_attn
		self.feed_forward = feed_forward
		self.sublayer = clones(SublayerConnection(size, dropout), 3)
	def forward(self, x, memory, src_mask, tgt_mask):
		m = memory
		x = self.sublayer[0](x, lambda x: self.self_attn(x,x,x,tgt_mask))
		x = self.sublayer[1](x, lambda x: self.src_attn(x,m,m,src_mask))
		return self.sublayer[2](x, self.feed_forward)
```

DecoderLayer 比 EncoderLayer 多了一个 src-attn 层，这是 Decoder 时 attend to Encoder 的输出(memory)。src_attn 和 self_attn 的实现是一样的，只不过使用的 Query, Key 和 Value 的输入不同。普通的 Attention(src_attn)的 Query 是从下层输入进行来的。 Key 和 Value 是 Encoder 最后一层的输出 memory;而 Self-Attention 的 Query, Key 和 Value 都是来自下层输入进来的。

Decoder 和 Encoder 有一个关键的不同：Decoder 在解码第 t 个时刻的时候只能用$1 \cdots t$时刻的输入，而不能使用$t+1$时刻及其之后的输入。因此我们需要一个函数来产生一个 Mask 矩阵,代码如下：

```python
def subsequent_mask(size):
	# mask out subsequent positoins
	attn_shape = (1, size, size)
	subsequent_mask = np.triu(np.ones(attn_shape),k=1).astype('uint8')
	return torch.from_numpy(subsequent_mask) == 0
```

我们阅读代码之前先看它的输出：

```python
print(subsequent_mask(5))
# 输出
1 0 0 0 0
1 1 0 0 0
1 1 1 0 0
1 1 1 1 0
1 1 1 1 1
```

我们发现它输出的是一个方阵，对角线和下面都是 1。第一行只有第一列是 1，它的意思是时刻 1 只能 attend to 输入 1， 第三行说明时刻 3 可以 attend to $\{1, 2, 3 \}$而不能 attend to $\{4,5\}$的输入，因为在真正 Decoder 的时候这是属于 Future 的信息。

## MultiHeadedAttention 多头注意力机制

Attention(包括 Self-Attention 和普通的 Attention)可以堪称一个函数，它的输入是 Query，Key，Value 和 Mask，输出是一个 Tensor。其中输出是 Value 的加权平均，而权重来自 Query 和 Key 的计算。具体的计算如下图所示，计算公式为：$$Attention(Q,K,V) = softmax(\frac{QK^{T}}{\sqrt{d_{k}}})V$$

![Attention计算图](http://fancyerii.github.io/img/transformer_codes/the-annotated-transformer_33_0.png)

代码为：

```python
def attention(query, key, value, mask=None,dropout=None):
	d_k = query.size(-1)
	scores = torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)
	if mask is not None:
		scores = scores.mask_fill(mask==0,-1e9)
	p_attn = F.softmax(scores, dim=-1)

	if dropout is not None:
		p_attn = dropout(p_attn)
	return torch.matmul(p_attn,value),p_attn
```

这里主要的疑问是在`score.mask_fill`，主要用于把 mask 是 0 的变成一个很小的数，这样后面经过 softmax 之后的概率就很接近零（但是理论上还是用了很少一点点未来的信息）。

之前介绍过，对于每一个 Head，都是用三个矩阵$W^{Q}$，$W^{K}$，$W^{V}$把输入转换成 Q，K 和 V。然后分别用每一个 Head 进行 Self- Attention 的计算，最后把 N 个 Head 的输出拼接起来，最后用一个矩阵$W^{O}$把输出压缩一下。具体计算框架为：

![Multi-Head Self-Attention](http://fancyerii.github.io/img/transformer_codes/the-annotated-transformer_38_0.png)

代码如下：

```python
class MultiHeadAttention(nn.Module):
	def __init__(self, h, d_model, dropout=0.1):
		super(MultiHeadAttention, self).__init__()
		assert d_model % h == 0
		self.d_k = d_model // h # 这里是整除
		self.h = h
		self.linears = clones(nn.Linear(d_model,d_k),4)
		self.attn = None
		self.dropout = nn.Dropout(p=dropout)

	def foward(self, query, key, value, mask=None):
		if mask is not None:
			# 所有h个head的mask都是相同的
			mask = mask.unsqueeze(1)
		nbatches = query.size(0)
		# 1) 首先使用线性变换，然后把d_model分配给h个Head，每个head为d_k=d_model/h
		query, key, value = [l(x).view(nbatches,-1,self.h,self.d_k).transpose(1,2) for l,x in zip(self.linears,(query, key, value))]
		# 2) 使用attention函数计算
		x, self.attn = attention(query, key, value, mask=mask,dropout = self.dropout)
		# 3）
		x = x.transpose(1,2).contiguous().view(nbatches, -1,self.h*self.d_k)
		return self.linears[-1](x)
```

我们首先来看构造函数， 这里 d_model(512)是 Multi-Head 的输出大小，因为有 h(8)个 head， 因此每个 head 的 d_k=512/8=64。接着我们构造 4 个(d_model $*$ d_model)的矩阵，后面我们会看到它的用处。最后是构造一个 Dropout 层。

然后我们来看 forward 方法。输入的 mask 是（batch,1,time）的，因为每个 head 的 mask 都是一样的，所以先用 unsqueeze(1)变成(batch,1,1,time)，mask 我们前面已经分析过了。

接下来就是根据输入的 query, key, value 计算变换后的 Multi-Head 的 query, key 和 value。`zip(self.linears, (query,key,value))`是把`(self.linear[0], self.linear[1], self.linear[2])`和`(query, key, value)`放在一起遍历。我们可以只看一个`self.linear[0] (query)`。根据构造函数的定义，`self.linear[0]`是一个`[512,512]`的矩阵，而`query`是`(batch, time, 512)`，相乘之后得到的新 query 还是 512 维的向量，然后用 view 把它变成`(batch, time, 8, 64)`。然后 transpose 成(batch, 8, time, 64)，这是 attention 函数要求的 shape。分别对 8 个 Head，每个 Head 的 Query 都是 64 维。最后使用`self.linear[-1]`对`x`进行线性变换，`self.linear[-1]`是`[512, 512]`的，因此最终的输出还是`(batch, time, 512)`。

