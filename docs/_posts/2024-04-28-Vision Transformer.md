---
layout: post
title:  "Vision Transformer"
date:   2024-04-28
categories: LEARNING
tags: AI Transformer Vision
---
Refer 1: [[Transformer_learning]] for general Transformer tech. 
Refer 2: [[transformer_implementation]] for general  Transformer tech. 

# Swin Transformer

refer to https://arxiv.org/abs/2103.14030 

refer to [图解Swin Transformer - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/367111046)

refer to ""/workspace/deeplearning/NLP/Vision Transformers from Scratch (PyTorch).ipynb" and "/workspace/deeplearning/NLP/vision-transformer-vit-tutorial-baseline.ipynb"

- pre-condition: Vision Transformer (ViT) from the paper “An Image is Worth 16x16 Words” has showed promising results for some high-level vision tasks such as image classification

- issue: the global attention mechanism in ViT scales with quadratic computational complexity with respect to the resolution of an image which is not sustainable for tasks that work with images in the high resolution space

- answer: Microsoft proposed the Swin-Transformer which features a local attention mechanism based on shifting windows whose computational complexity scales linearly and could serve as an all-purpose backbone for general vision tasks.



 The model (Swin-Transformer) starts by **splitting an image into p x p non-overlapping patches with a linear embedding exactly like ViT**. Our image transforms from **(h,w,c) to (h/p,w/p,c*p^2) from patch partitioning, and then to **(h/p * w/p, C)**** after the linear projection. We treat the h*w patches as the tokens of the transformer sequence and C as our embedding dimension.

## Swin Transformer的整体架构

整个模型采取层次化的设计，一共包含4个Stage，每个stage都会缩小输入特征图的分辨率，像CNN一样逐层扩大感受野。

- 在输入开始的时候，做了一个`Patch Embedding`，将图片切成一个个图块，并嵌入到`Embedding`。
- 在每个Stage里，由`Patch Merging`和多个Block组成。
- 其中`Patch Merging`模块主要在每个Stage一开始降低图片分辨率。
- 而Block具体结构如右图所示，主要是`LayerNorm`，`MLP`，`Window Attention` 和 `Shifted Window Attention`组成 (为了方便讲解，我会省略掉一些参数)

![img](/assets/Vision%20Transformer.assets/17fyQm7H34kXGwENxyK1GYw.png)

We then send the image into a series of transformer blocks and patch merging blocks.

- The transformer blocks feature attention computed locally within windows with an alternating shifted window configuration to allow the model to gain global input information. The attention computation also features relative embeddings to directly encode positional information into the model.

- In between these transformer blocks we use **patch merging layers** to decrease the height and width of our images while increasing the channel size/embedding dimension of the model. This is similar to how CNN’s transform the input as you go deeper into the model. The goal is to have the Swin-Transformer work as a computer vision backbone for various vision tasks just like how the CNN has formally been used in computer vision.

  其中有几个地方处理方法与ViT不同：

  - ViT在输入会给embedding进行位置编码。而Swin-T这里则是作为一个**可选项**（`self.ape`），Swin-T是在计算Attention的时候做了一个`相对位置编码`
  - ViT会单独加上一个可学习参数，作为分类的token。而Swin-T则是**直接做平均**，输出分类，有点类似CNN最后的全局平均池化层

![image-20240116091743493](/assets/Vision%20Transformer.assets/image-20240116091743493.png)

![image-20240116092157934](/assets/Vision%20Transformer.assets/image-20240116092157934.png)

![image-20240116140930491](/assets/Vision%20Transformer.assets/image-20240116140930491.png)

![image-20240116193809061](/assets/Vision%20Transformer.assets/image-20240116193809061.png)

## **Patch Embedding**

在输入进Block前，我们需要将图片切成一个个patch，然后嵌入向量。

具体做法是对原始图片裁成一个个 `patch_size * patch_size`的窗口大小，然后进行嵌入。

这里可以通过二维卷积层，**将stride，kernelsize设置为patch_size大小**。设定输出通道来确定嵌入向量的大小。最后将H,W维度展开，并移动到第一维度

==Patch Partition + Linear Embedding==

> “It first splits an input RGB image into non-overlapping patches by a patch splitting module, like ViT. Each patch is treated as a “token” and its feature is set as a concatenation of the raw pixel RGB values. In our implementation, we use a patch size of 4 × 4 and thus the feature dimension of each patch is 4 × 4 × 3 = 48. A linear embedding layer is applied on this raw-valued feature to project it to an arbitrary dimension (denoted as C).”

```
class SwinEmbedding(nn.Module):

    '''
    input shape -> (b,c,h,w)
    output shape -> (b, (h/4 * w/4), C)
    '''

    def __init__(self, patch_size=4, C=96):
        super().__init__()
        self.linear_embedding = nn.Conv2d(3, C, kernel_size=patch_size, stride=patch_size)
        self.layer_norm = nn.LayerNorm(C)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.linear_embedding(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.relu(self.layer_norm(x))
        return x
```

- **image_resolution** = `224 x 224`
- **patch_size** = `4 x 4` in terms of pixels per height/width
- **number of pixels(features) in one patch** = `4 x 4 x 3` = `48`
- **total number of patches** in the whole image = `224/4 x 224/4` = `3136`
- **window_size** = `7 x 7` in terms of patches per height/width
- **number of patches** in one window = `7 x 7` = `49`
- **total number of windows** in the whole image = `224/4/7 x 224/4/7` = `8 x 8` = `64`

## Linear Embedding Layer

A **linear embedding layer** is applied on this feature concatenation to project it to an arbitrary dimension (denoted as **C**). In the official implementation, **patch partition** and **embedding** are simultaneously done with `nn.Conv2d` with `kernel_size=patch_size`, `stride=patch_size`. Although the architecture in the paper first performs patch partition and the linear embedding, the official implementation **first performs embedding**(both partition & embedding) and uses **window partition** module after.

```python
class PatchEmbed(nn.Module):
    """ Convert image to patch embedding

    Args:
        img_size (int): Image size (Default: 224)
        patch_size (int): Patch token size (Default: 4)
        in_channels (int): Number of input image channels (Default: 3)
        embed_dim (int): Number of linear projection output channels (Default: 96)
        norm_layer (nn.Module, optional): Normalization layer (Default: None)
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size) # (img_size, img_size) to_2tuple simply convert t to (t,t)
        patch_size = to_2tuple(patch_size) # (patch_size, patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]] # (num_patches, num_patches)

        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # proj layer: (B, 3, 224, 224) -> (B, 96, 56, 56)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """
        x: (B, C, H, W) Default: (B, 3, 224, 224)
        returns: (B, H//patch_size * W//patch_size, embed_dim) (B, 56*56, 96)
        """
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}]) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # (B, 3, 224, 224) -> (B, 96, 56, 56)
        x = self.proj(x)

        # (B, 96, 56, 56) -> (B, 96, 56*56)
        x = x.flatten(2)

        # (B, 96, 56*56) -> (B, 56*56, 96): 56 refers to the number of patches
        x = x.transpose(1, 2)

        if self.norm is not None:
            x = self.norm(x)

        return x
```

## Patch Merging

该模块的作用是在每个Stage开始前做降采样，用于缩小分辨率，调整通道数 进而形成层次化的设计，同时也能节省一定运算量。

> 在CNN中，则是在每个Stage开始前用`stride=2`的卷积/池化层来降低分辨率。

每次降采样是两倍，因此**在行方向和列方向上，间隔2选取元素**。

然后拼接在一起作为一整个张量，最后展开。<font color=red>**此时通道维度会变成原先的4倍**（因为H,W各缩小2倍），此时再通过一个**全连接层再调整通道维度为原来的两倍**</font> ==此处维度缩减==

*To produce a hierarchical representation, the number of tokens is reduced by patch merging layers as the network gets deeper. The first patch merging layer concatenates the features of each group of 2 × 2 neighboring patches, and applies a linear layer on the 4C-dimensional concatenated features. **This reduces the number of tokens by a multiple of 2×2 = 4 (2× downsampling of resolution), and the output dimension is set to 2C.***

```python
#The patch merging layer is straightforward. We initialize a linear layer with 4C input channels to 2C output channels and initialize a layer norm with the output embedding size. In our forward function we use einops rearrange to reshape our tokens from 2x2xC to 1x1x4C. We finish by passing our inputs through the linear projection and layer norm.

class PatchMerging(nn.Module):

    '''
    input shape -> (b, (h*w), C)
    output shape -> (b, (h/2 * w/2), C*2)
    '''

    def __init__(self, C):
        super().__init__()
        self.linear = nn.Linear(4*C, 2*C)
        self.layer_norm = nn.LayerNorm(2*C)

    def forward(self, x):
        height = width = int(math.sqrt(x.shape[1])/2)
        x = rearrange(x, 'b (h s1 w s2) c -> b (h w) (s2 s1 c)', s1=2, s2=2, h=height, w=width)
        return self.layer_norm(self.linear(x))
```

![img](/assets/Vision%20Transformer.assets/v2-a1a0ea5d9455083caed65006433c4efe_720w.webp)



## Shifted Window based Self-Attention**

标准的Transformer结构或其变体都采用的是Global Self Attention，其会计算一个token和其他所有token的关系，其计算复杂度太高，不适合与密集预测等需要大量token的任务

为了降低计算复杂度，SwinTransformer在局部Windows内部计算Self-Attention。每个image 都会被平均划分为若干个windows，并且这些Windows之间是没有重蛋的。假设image的大小为 $h * w$ ，**每个Window包含 $M * M$ 个patches**，则标准MSA和基于window的局部SelfAttention 的计算量分别为:
$$
\begin{aligned}
& \Omega(\mathrm{MSA})=4 h w C^2+2(h w)^2 C \\
& \Omega(\mathrm{W}-\mathrm{MSA})=4 h w C^2+2 M^2 h w C
\end{aligned}
$$

两个公式的推导可参见下图:

对于MSA来说:
tokens: hxwxC heads: $d$
(1) 首光计算每个head的计算量
(a) 计算 $q, k . v$ :

$\Rightarrow$ 对于 $q$ ，input: $h w \times \frac{c}{d} $, output: $h w \times \frac{c}{d}$需要计算量 $h w \times \frac{c}{d} \times c$

![image-20240111134545199](/assets/Vision%20Transformer.assets/image-20240111134545199.png)

参考右侧矩阵乘法: 输出短阵大小为$hw \times \frac{c}{d}$, 其中的每个元素需要进行C次运算,  对于 $q$ ，需要计算量 $h w \times \frac{c}{d} \times c$

同理，得到 $k,v: h \omega \times \frac{c^2}{d}$

(b) $Q K^{\top}$
(c) $Q K^{\top} V($ 忽略 Softmax)
QK $K^{\top}$
(d) 一个head 整体计算量:
$3 h w \times \frac{c^2}{d}+2(h w)^2 (c/d) $

总体计算量： 
$$
quad d\left[3 h w \times \frac{c^2}{d}+2(h w)^2 \frac{c}{d}\right]+h w \times c^2

=4 h \omega \times c^2+2(h \omega)^2 c
$$
(2) 融合多个 head:
$$
\left.h w\left[{ }^c\right]{ }^c[] \rightarrow i^c\right] \quad h w \times c \times c
$$
对于W-MSA来说
tokens: hxwxC ,  heads: $d$ ,  window size: $M \times M$
每个window 内部计算 Self-A tetention, 其计算量可根据MSA 来推
$4 m^2 x C^2+2 M^4 C$
一共有 $\frac{h}{M} \times \frac{W}{M}$ 个window:
总体计算量为:
$$
\left(\frac{h}{m} \times \frac{w}{m}\right) \times\left(4 m^2 C^2+2 M^4 c\right)
=4 h \omega c^2+2 M^2 h \omega c
$$
由于Window的大小是固定的（论文中设定为7），W-MSA的计算量将远远小于MSA

### Window Attention Mechanism

这是这篇文章的关键。传统的Transformer都是**基于全局来计算注意力的**，因此计算复杂度十分高。而Swin Transformer则将**注意力的计算限制在每个窗口内**，进而减少了计算量。

主要区别是在原始计算Attention的公式中的Q,K时**加入了相对位置编码**。后续实验有证明相对位置编码的加入提升了模型性能。

首先QK计算出来的Attention张量形状为`(numWindows*B, num_heads, window_size*window_size, window_size*window_size)`。

**window partition 函数是用于对张量划分窗口，指定窗口大小。将原本的张量从 N H W C, 划分成 num_windows*B, window_size, window_size (ZTD: how many patches in one window, ==patch should be already be calculated by in previouse 'patch embedding'== ), C ，其中 num_windows = H\*W / (window_size\*window_size)，即窗口的个数。而window reverse函数则是对应的逆过程。这两个函数会在后面的Window Attention用到。**

![img](/assets/Vision%20Transformer.assets/1Kgi0npIhx7pdSBddP5m28A.png)

```python
def window_partition(x, window_size=7):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size (Default: 7)

    Returns:
        windows: (num_windows * B, window_size, window_size, C)
                 (8*8*B, 7, 7, C)
    """

    B, H, W, C = x.shape

    # Convert to (B, 8, 7, 8, 7, C) 
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)

    # Convert to (B, 8, 8, 7, 7, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()

    # Efficient Batch Computation - Convert to (B*8*8, 7, 7, C)
    windows = windows.view(-1, window_size, window_size, C)


    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
```

Window Attention Example

In the Swin Transformer, attention is computed with the familiar attention formula shown in the image above **but in parallel across non-overlapping windows**. We will start by first coding the standard window based self attention mechanism and we will deal with the alternating shifted windows later.

```python
'''
We start by initializing our parameters embed_dim, num_heads, and and window_size and defining two linear projections. 
  - The first is our projection from inputs to Queries, Keys, and Values which we do in one parallel projection so the output size is set to 3*C. 
  - The second projection is a linear projection applied after the attention computation. This projection is for communication between the concatenated parallel multi-headed attention units.

'''

class ShiftedWindowMSA(nn.Module):

'''
 input shape -> (b,(h*w), C)
 output shape -> (b, (h*w), C)
'''

    def __init__(self, embed_dim, num_heads, window_size=7):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.proj1 = nn.Linear(embed_dim, 3*embed_dim)
        self.proj2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        h_dim = self.embed_dim / self.num_heads
        height = width = int(math.sqrt(x.shape[1]))
        x = self.proj1(x)

        x = rearrange(x, ‘b (h w) (c K) -> b h w c K’, K=3, h=height, w=width)
        x = rearrange(x, ‘b (h m1) (w m2) (H E) K -> b H h w (m1 m2) E K’, H=self.num_heads, m1=self.window_size, m2=self.window_size)
        
      '''
        H = # of Attention Heads
        h,w = # of windows vertically and horizontally
        (m1 m2) = total size of each window
        E = head dimension
        K = 3 = a constant to break our matrix into 3 Q,K,V matricies 
      '''

        Q, K, V = x.chunk(3, dim=6)
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)
        att_scores = (Q @ K.transpose(4,5)) / math.sqrt(h_dim)
        att = F.softmax(att_scores, dim=-1) @ V

        x = rearrange(att, ‘b H h w (m1 m2) E -> b (h m1) (w m2) (H E)’, m1=self.window_size, m2=self.window_size)
        x = rearrange(x, ‘b h w c -> b (h w) c’)

        return self.proj2(x)
```

### **Shifted Window Attention**

前面的Window Attention是在每个窗口下计算注意力的，为了更好的和其他window进行信息交互，Swin Transformer还引入了shifted window操作。

![image-20240112151212065](/assets/Vision%20Transformer.assets/image-20240112151212065.png)

Shifted Window方法是在**连续的两个Transformer Block之间实现的**。
- 第一个模块使用一个标准的window partition策略，**从feature map的左上角出发**，例如一个 $8 * 8$ 的feature map会被平分为 $2 * 2$ 个window，每个window的大小为 $M=4$ 。
- 紧接着的第二个模块则使用了移动窗口的策略，window会从feature map的 $\left(\left\lfloor\frac{M}{2}\right\rfloor,\left\lfloor\frac{M}{2}\right\rfloor\right)$ 位置处开始，然后再进行window partition操作。

这样一来，不同window之间在两个连续的模块之间便有机会进行交互。基于移动窗口策略，两个连续的SwinTransformer Block的计算过程如下:
$$
\begin{aligned}
& \hat{\mathbf{z}}^l=\mathrm{W}-\operatorname{MSA}\left(\operatorname{LN}\left(\mathbf{z}^{l-1}\right)\right)+\mathbf{z}^{l-1} \\
& \mathbf{z}^l=\operatorname{MLP}\left(\operatorname{LN}\left(\hat{\mathbf{z}}^l\right)\right)+\hat{\mathbf{z}}^l, \\
& \hat{\mathbf{z}}^{l+1}=\operatorname{SW}-\operatorname{MSA}\left(\operatorname{LN}\left(\mathbf{z}^l\right)\right)+\mathbf{z}^l \\
& \mathbf{z}^{l+1}=\operatorname{MLP}\left(\operatorname{LN}\left(\hat{\mathbf{z}}^{l+1}\right)\right)+\hat{\mathbf{z}}^{l+1}
\end{aligned}
$$
Shifted Window Partition存在一个问题，由于没有与边界对齐，其会产生更多的Windows，从 $\left\lceil\frac{h}{M}\right\rceil \times\left\lceil\frac{w}{M}\right\rceil$ 个Windows上升至 $\left\lceil\frac{h}{M}+1\right\rceil \times\left\lceil\frac{w}{M}+1\right\rceil$ ，并且其中很多windows的大小也不足 $M * M$ ，具体可以参见原论文中的Figure 2。

**Naive Solution**

比较Naive的一种解决方法如下图所示：

![img](/assets/Vision%20Transformer.assets/v2-c7d037d20ef14a5c0098572e38cc2bd2_720w.webp)

可以看出这种解决方法的缺点在于额外计算了很多padding的部分，浪费了大量计算。

**Batch Computation Approach**

为此，SwinTransformer采用了一个更为高效的 Batch Computation Approach。

![img](/assets/Vision%20Transformer.assets/1InwcdUt4II6dSl5srk0dew.png)

这一部分在论文中并没有详细说明，仅仅通过上图进行了展示，其实整体思想就是：通过设定特殊的mask，**在Attention时，仅对一个window内的有效部分进行Attention，其余部分被mask掉**，即可实现在原来计算Attention方法不变的情况下，对非规则的Window计算Attention。

在实际代码里，我们是**通过对特征图移位，并给Attention设置mask来间接实现的**。能在**保持原有的window个数下**，最后的计算结果等价。 

> **特征图移位操作**
>
> The idea of shifted window attention is that on alternating attention computation layers we shift our windows so that our shifted windows overlap over the previous layers windows to allow cross window communication for the model. We can achieve this efficiently by a cyclic shift as depicted in the image above. PyTorch has a function torch.roll we can use that will perform our cyclic shift of size window_size/2 on our input.
>
> 代码里对特征图移位是通过`torch.roll`来实现的，下面是示意图
>
> ![img](/assets/Vision%20Transformer.assets/v2-8d8274d62026e0732c8a7827de1070fc_720w.webp)



```python
# calculate attention mask for SW-MSA
Hp = int(np.ceil(H / self.window_size)) * self.window_size
Wp = int(np.ceil(W / self.window_size)) * self.window_size
img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
h_slices = (slice(0, -self.window_size),  #1
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None))
w_slices = (slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None))
cnt = 0  # 2
for h in h_slices:
    for w in w_slices:
        img_mask[:, h, w, :] = cnt
        cnt += 1

mask_windows = window_partition(img_mask, self.window_size)  # (nW, window_size, window_size, 1) #3
mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) # 4
attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
```

1. Slice the indices into `[0: -window_size]`, `[-window_size: -shift_size]`, and `[-shift_size:]` for both horizontally and vertically. Then we get these 9 partitions as shown in the Figure 11.
2. Assign numbers `0 ~ 8` to **each patch** of the divided regions (numbers are drawn region-wise for simplicity).
3. Perform **window partitioning** with equal size of window (`M x M`). After the partition, we have a total of 4 windows. The first window has region `0`, the second has `1, 2`, the third has `3, 6`, and the fourth has `4, 5, 7, 8`.
4. For each of the window, **perform self-attention individually for each region number by applying attention masks**. ==The implementation of this separate self-attention is the beauty I think. Remember that we apply this separate self-attention when we're dealing with self-attention, meaning $q k^{\top}$ which has $M^2 \times M^2$ shape. (More details below).==
5. For attention mask, set to `0` for regions we want to apply self-attention, other wise a small number like `-100` for regions we don't want to apply self-attention.
6. Later when we get $q k^{\top}$ add this attention-mask.
7. Perform reverse cyclic shift.

> squeeze and unsqueeze 主要用于对数据的维度进行压缩或者解压
>
> 再看torch.unsqueeze()这个函数主要是对数据维度进行扩充。给指定位置加上维数为一的维度，比如原本有个三行的数据（3)，在0的位置加了一维就变成一行三列 $(1,3)$ 。a.squeeze $(N)$ 就是在a中指定位置 $N$ 加上一个维数为 1 的维度。还有一种形式就是 $b=$ torch.unsqueeze $(a ， N)$ a 就是在a中指定位置 $N$ 加上一个维数为 1 的维度

#### Attention masks 1

**cyclic shift is great!** However one issue will come from using the cyclic shift to perform shifted window attention. **Because we are now computing attention on the new shifted image, the model will be confused on where sections A,B,C** in the above example actually belong in the image. 

- Remember that the core concept of Swin Transformer is that **self-attention is performed only within each window locally**. 

  However, As you can see in the Fig. 9, one window(newly created window) contains parts of different windows(from the perspective of before cyclic shift).

  ![space-1.jpg](/assets/Vision%20Transformer.assets/swint9.png)

   This is not good.. we need to make sure that self-attention is performed only within each window from the perspective of **original 9 windows** before the cyclic shift. How can we do that? That's why we need **attention mask** as shown in Fig. 4.


![space-1.jpg](/assets/Vision%20Transformer.assets/swint10.png)

dive deep into the codes.

  ![space-1.jpg](/assets/Vision%20Transformer.assets/swint11.png)

Let's take an example of **bottom-left** window from Figure As stated earlier, we want to apply self-attention individually for each region(separately for `3` and `6`) in this window as shown 

![space-1.jpg](/assets/Vision%20Transformer.assets/swint12.png)

==The official implementation uses a nice yet super simple trick to achieve this. Follow each step along with the figure.==

1. First, create two duplicates of the same window with shape `M x M` for each window (`nW` means total number of windows)

2. Flatten `M x M` to `M^2` for each window.

3. `unsqueeze` one dimension vertically for one duplicate and horizontally for the other one.

4. Subtract the duplicates to obtain the `M^2 x M^2` shape attention-mask matrix.

5. Make all the **non-zero** patches to `-100`.

   |![space-1.jpg](/assets/Vision%20Transformer.assets/swint13-1705151546044-5.png)

#### Attention masks 2

以上几行即为Mask的计算代码，其中 $H ， W$ 即为输入feature map的高和宽。window_size即为 window的大小，也就是论文中的 $M$ ， shift_size为窗口移动的大小，shift_size $=\left\lfloor\frac{\bar{M}}{2}\right\rfloor$ ， self 是对象，可以忽略。详细说明见下图:

![img](/assets/Vision%20Transformer.assets/v2-dc2fe96c5c67510aeabbcff3489c9757_720w.webp)

![img](/assets/Vision%20Transformer.assets/v2-61469886ee2ef8995996a5a0acd69ab8_720w.webp)

![img](/assets/Vision%20Transformer.assets/v2-9cb8b56e82d02370c8b243a54a5efc00_720w.webp)

其他的window对应的Attention Mask可以采用上述类似的逻辑推导出其具体值。 下图依次为window (1)，window (2)，window (3)，window (4)对应的attn mask的示意图：

![img](/assets/Vision%20Transformer.assets/v2-47d473a4f6ac4d81deadbf5689e3579f_720w.webp)

![img](/assets/Vision%20Transformer.assets/v2-a55556d6193de2c4c352fef51b6302c1_720w.webp)

![img](/assets/Vision%20Transformer.assets/v2-1eb1afb0db41afa9414b9ad8da2cc00f_720w.webp)

![img](/assets/Vision%20Transformer.assets/v2-120e87528b7b80db050dd26c861c48ba_720w.webp)

![img](/assets/Vision%20Transformer.assets/v2-d4bd615abc8547385415512c4d4e0470_720w.webp)

![img](/assets/Vision%20Transformer.assets/v2-b483bbdff8181f3bef9c5445d86b2d36_720w.webp)

#### Attention mask 3

我认为这是Swin Transformer的精华，通过设置合理的mask，让`Shifted Window Attention`在与`Window Attention`相同的窗口个数下，达到等价的计算结果。首先我们对Shift Window后的每个窗口都给上index，并且做一个`roll`操作（window_size=2, shift_size=-1）

![img](/assets/Vision%20Transformer.assets/v2-52b0bec2b0e2341e1eab1fd6342bc9e6_720w.webp)

我们希望在计算Attention的时候，**让具有相同index QK进行计算，而忽略不同index QK计算结果**。

最后正确的结果如下图所示 (PS: 这个图的Query Key画反了。。。应该是4x1 和 1x4 做矩阵乘，读者们自行交换下位置，抱歉）

![img](/assets/Vision%20Transformer.assets/v2-af19485ae400a2f52ede6306fcfb078e_720w.webp)

而要想在原始四个窗口下得到正确的结果，我们就必须给Attention的结果加入一个mask（如上图最右边所示）

### Relative Position Bias

Another key component of Swin Transformer is relative position bias $B$. Unlike absolute position embedding, relative position bias literally tells us relative position information of the patches in a window. The relative position bias matrix $B$ is added after $\frac{Q K^{\top}}{\sqrt{d}}$ as shown below.
$$
\operatorname{Attention}(Q, K, V)=\operatorname{Softmax}\left(\frac{Q K^{\top}}{\sqrt{d}}+B\right) V
$$
where $Q, K, V \in \mathbb{R}^{M^2 \times \boldsymbol{d}}$ are the query, key, and value matrixes and $d$ is the query/key dimension and $M^2$ is the number of patches in a window.

Since a window size is $M \times M$, the relative position along each axis lies in the range $[-M+1, M-1]$ as shown in the following Fig 

![space-1.jpg](/assets/Vision%20Transformer.assets/swint14.png)

# Scalable Diffusion Models with Transformers (TBD)

[论文阅读：Scalable Diffusion Models with Transformers_adaptive layernorm-CSDN博客](https://blog.csdn.net/huzimu_/article/details/136509658)

[Scalable Diffusion Models with Transformers（DiTs）论文阅读 -- 文生视频Sora模型基础结构DiT - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/597695487)

[2212.09748.pdf (arxiv.org)](https://arxiv.org/pdf/2212.09748.pdf)

[facebookresearch/DiT: Official PyTorch Implementation of "Scalable Diffusion Models with Transformers" (github.com)](https://github.com/facebookresearch/DiT)

[run_DiT.ipynb - Colab (google.com)](https://colab.research.google.com/github/facebookresearch/DiT/blob/main/run_DiT.ipynb#scrollTo=5JTNyzNZKb9E)

[chuanyangjin/fast-DiT: Fast Diffusion Models with Transformers (github.com)](https://github.com/chuanyangjin/fast-DiT?tab=readme-ov-file)

[Scaleble Diffusion Models with Transformers | by Liuzhihui | Feb, 2024 | Medium](https://medium.com/@liuzhihui2046/scaleble-diffusion-models-with-transformers-527fbfa8eab7)

文章提出使用Transformers替换扩散模型中U-Net主干网络，分析发现，这种Diffusion Transformers（DiTs）不仅速度更快（更高的Gflops），而且在ImageNet 512×512和256×256的类别条件图片生成任务上，取得了更好的效果，256×256上实现了SOTA的FID指标（2.27）。

Transformers已经广泛应用于包括NLP、CV在内的机器学习的各个领域。然而，很多图片level的生成模型还坚持使用卷积神经网络，比如扩散模型采用的就是U-Net的主干网络架构。经过演化，扩散模型中的U-Net网络增加了稀疏的自注意力模块，此外 Dhariwal and Nichol 也尝试过在U-Net模型上的一些改变，比如通过增加适配的正则化层来注入条件信息和Channel数量。尽管如此，U-Net的顶层设计还是与原始U-Net相差无几。

文章的目标就是要揭开扩散模型架构选择的神秘面纱，提供一个强有力的baseline。文章发现U-Net并非不可替代，并且很容易使用诸如Transformers的结构替代U-Net，使用Transformers可以很好地保持原有的优秀特性，比如可伸缩性、鲁棒性、高效性等，并且使用新的标准化架构可能在跨领域研究上展现出更多的可能。文章从网络复杂度和采样质量两个方面对DiTs方法进行评估。

### DDPMs

- 扩散模型是借鉴了物理学上的扩散过程，在生成模型上，分为正向和逆向的过程。正向过程是向信号中逐渐每步加少量噪声，当步数足够大时可以认为信号符合一个高斯分布。所以逆向过程就是从随机噪声出发逐渐的去噪，最终还原成原有的信号。

- 去噪过程一般采用UNet或者ViT，使用t步的结果和条件输入预测t-1步增加的噪声，然后使用DDPM可以得到t-1步的分布，经过多步迭代就可以从随机噪声还原到有实际意义的信号。如果使用原始DDPM速度会慢很多，所以很多工作如DDIM、FastDPM等工作实现了解码加速。


### 扩散模型基础

- 前向过程是一个T步逐渐加噪的马尔科夫链，公式如下

$$
\begin{gathered}
q\left(x_T \mid x_0\right)=\prod_{t=1}^T q\left(x_t \mid x_{t-1}\right)=\prod_{t=1}^T \mathcal{N}\left(\sqrt{1-\beta_t} x_{t-1} ; \beta_t \mathbf{I}\right) \\
=\mathcal{N}\left(\sqrt{\bar{\alpha}_t} x_0 ;\left(1-\bar{\alpha}_t \mathbf{I}\right)\right) ; \\
\bar{\alpha}_t=\prod_{t=1}^T \alpha_t ; \quad \alpha_t=1-\beta_t
\end{gathered}
$$
给定前向扩散过程作为先验，扩散模型训练反转的过程，可以通过去除所加噪声从XT恢复成X0，并且每步的扩散过程都采样自特定的高斯分布，其期望和方差如下:
$$
p_\theta\left(x_{t-1} \mid x_t\right)=\mathcal{N}\left(\mu_\theta\left(x_t, t\right), \Sigma_\theta\left(x_t, t\right)\right)
$$

优化目标是负的X0概率似然，其上界如下所示:
$$
L=\mathbb{E}\left[-\log p_\theta\left(x_0\right)\right] \leq \mathbb{E}\left[-\log \frac{p_\theta\left(x_{0: T}\right)}{q\left(x_{1: T} \mid x_0\right)}\right]
$$

并且其目标可以简化为预测和ground truth之间的12 loss。
$$
\mathcal{L}=\mathbb{E}_{x, \epsilon \sim \mathcal{N}(0, I), t}\left[\left\|\epsilon-\epsilon_\theta\left(x_t, t\right)\right\|_2^2\right] .
$$

- Classifier-free guidance
	条件扩散模型是将条件信息作为额外的输入，比如一个分类标签c。这种情况下反向过程变为了
	$$
	p_\theta\left(x_{t-1} \mid x_t, c\right) \text {. }
	$$
	
	根据贝叶斯规则
	$$
	\log p(c \mid x) \propto \log p(x \mid c)-\log p(x)
	$$
	
	因此
	$$
	\nabla_x \log p(c \mid x) \propto \nabla_x \log p(x \mid c)-\nabla_x \log p(x)
	$$
	
	所以在想要条件的概率较大，就可以将条件的梯度增加到优化目标里，最终可以表示成如下形式:
	$$
	\epsilon_\theta\left(x_t, \emptyset\right)+s \cdot\left(\epsilon_\theta\left(x_t, c\right)-\epsilon_\theta\left(x_t, \emptyset\right)\right) .
	$$
	
	模型在训练时，使用一个网络架构优化两个模型 (uncond，cond)。
- Latent diffusion models：扩散模型在像素空间上训练和推理的计算开销过大，Latent Diffusion Model (LDM) 将像素空间替换为VAE编码得到的潜在空间 $z=E(x)$​ ，可以提高计算效率。本文提出的DiT沿用了LDM中的潛在空间，但是在预测潜在空间特征的模型上，将LDM中的U-Net替换为了纯Transformer骨架。



![img](/assets/Vision%20Transformer.assets/v2-00d94bdf03ccd7a8ef3b61f0726254b7_720w.webp)



In a typical diffusion model, a U-Net convolutional neural network (CNN) learns to estimate the noise to be removed from an image. ==DiTs replace this U-Net with a transformer==. This replacement shows that U-Net’s inductive bias is not necessary for the performance of diffusion models.

![Architecture of diffusion transformer](/assets/Vision%20Transformer.assets/ZfgaqMmUzjad_UTL_image1.jpeg)

**Patch化**：DiT的输入是通过VAE后的一个稀疏的表示z（256×256×3的图片，z为32×32×4），类似其他ViTs的方式，首先要将输入转成patch，文章采用超参p=2，4，8进行对比实验。

**DiT模块设计**：

- In-context条件：in-context条件是将t和c作为额外的token拼接到DiT的token输入中；
- Cross-attention模块：DiT结构与Condition交互的方式，与原来U-Net结构类似；
- Adaptive layer norm（adaLN）模块：使用adaLN替换原生LayerNorm（NeurIPS2019的文章，LN 模块中的某些参数不起作用，甚至会增加过拟合的风险。所以提出一种没有可学习参数的归一化技术）；
- adaLN-zero模块：之前的工作发现ResNets中每一个残差模块使用相同的初始化函数是有益的。文章提出对DiT中的残差模块的参数γ、β、α进行衰减，以达到类似的目的。

**模型大小**：与ViT大小相似，分别使用DiT-S、DiT-B、DiT-L和DiT-XL，Gflops从0.3dao118.6。

**Transformer Decoder**：在Transformer最上层需要预测噪音，因为Transformer可以保证大小与输入一致，所以在最上层使用一层线性进行decoder。

![image-20240411102956181](/assets/Vision%20Transformer.assets/image-20240411102956181.png)