---
layout: post
title: BERT GPT Diffusion Research
date: 2023-12-01
categories: LEARNING
tags: AI BERT GPT Diffusion
---

# 1.BERT (Bidirectional Encoder Representation Transformer#1)

## 1, The BERT Paper & Resources

- [Mastering BERT Model: A Complete Guide to Build it from Scratch | by CheeKean | Medium](https://kean-chan.medium.com/complete-guide-to-building-bert-model-from-sratch-3e6562228891)

- [BERT Explained: A Complete Guide with Theory and Tutorial | by Samia Khalid | Medium](https://medium.com/@samia.khalid/bert-explained-a-complete-guide-with-theory-and-tutorial-3ac9ebc8fa7c)

- [google-research/bert: TensorFlow code and pre-trained models for BERT (github.com)
- [Colab Notebook: Predicting Movie Review Sentiment with BERT on TF Hub](https://colab.research.google.com/github/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb#scrollTo=xiYrZKaHwV81)

- [Using BERT for Binary Text Classification in PyTorch](https://medium.com/swlh/a-simple-guide-on-using-bert-for-text-classification-bbf041ac8d04)

### BERT Paper

- **Paper:** [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)

- **Submitted:** Oct 11, 2018
- **First Author:** Jacob Devlin, Google AI Language
- **GitHub Repo:** [github.com/google-research/bert

### The Annotated Transformer (Blog Post)

- [](https://github.com/google-research/bert)
- **Initial Commit:** Oct 31, 2018
- "The Annotated Transformer” blog post : https://nlp.seas.harvard.edu/2018/04/03/attention.html

### BERT Announcement Post

- **Google Blog Post:** [Open Sourcing BERT: State-of-the-Art Pre-training for Natural Language Processing](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)
- **Published:** Nov 2, 2018
- **Authors:** Jacob Devlin, Ming-Wei Chang

### Attention is all you need (Paper)

- **Paper:** [Attention is all you need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)
- **Submitted:** Jun 12, 2017
- **First Author:** Ashish Vaswani, Google Brain

### Jay Alammar’s Posts

1. BERT——[The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)](http://jalammar.github.io/illustrated-bert/)
   - Published: Dec 3, 2018
2. Transformer——[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) 
   - Published: Jun 27, 2018
3. Attention—— [Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
   - Published: May 9, 2018

## 2, BER Architecture 

BERT which stands for [***Bidirectional Encoder Representation Transformer\***](https://arxiv.org/abs/1810.04805), a transformer based language model published by Google Research Team at 2018, is still gaining attention and being widely applied in Data Science Project today. This is due to the incredible model performance on multiple NLP tasks including question-answering, text tagging and sentence classification.

<img src="/assets/BERTGPTDiffusion%20Research.assets/1tf_5g-MwQG0cijTR1o15MQ-1684152127269-7.png" alt="img" style="zoom: 80%;" />

BERT relies on a Transformer mechanism, it contains the attention module that learns contextual relationships between words in a text. 

There are four types of pre-trained versions of BERT depending on the scale of the model architecture:

**BERT-Base**: 12-layer, 768-hidden-nodes, 12-attention-heads, 110M parameters
**BERT-Large**: 24-layer, 1024-hidden-nodes, 16-attention-heads, 340M parameters

***Fun fact\***: BERT-Base was trained on 4 cloud TPUs for 4 days and BERT-Large was trained on 16 TPUs for 4 days!

For details **on the hyperparameter and more on the architecture and results breakdown**, need go through the original paper.

### **WordPiece** Tokenization

The initial stage of creating a fresh BERT model involves training a new tokenizer. Tokenization is the process of breaking down a text into smaller units called “***tokens\***,” which are then converted into a numerical representation. An example of this would be splitting the sentence

```
“I like surfboarding!” → [‘[CLS]’, ‘i’, ‘like’, ‘surf’, ‘##board’, ‘##ing’, ‘!’, ‘[SEP]’] → [1, 48, 250, 4033, 3588, 154, 5, 2]
```

A tokenized BERT input always starts with a special **[CLS]** token and ends with a special **[SEP]** token, which are used for specific purposes that will be explained later. BERT employs a **WordPiece** tokenizer, which can split a single word into multiple tokens. 

![image-20230515201443054](/assets/BERTGPTDiffusion%20Research.assets/image-20230515201443054.png)

all sub words start with "##" except the beginning words

 **By referring to the explanation from [HuggingFace](https://huggingface.co/course/chapter6/6?fw=pt),, WordPiece computes a score for each pair, using the following:**

***score = (freq_of_pair) / (freq_of_first_element × freq_of_second_element)\***

By dividing the frequency of the pair by the product of the frequencies of each of its parts, **the algorithm prioritizes the merging of pairs where the individual parts are less frequent in the vocabular**y. For instance, it won’t necessarily merge `("un", "##able")` even if that pair occurs very frequently in the vocabulary, because the two pairs `"un"` and `"##able"` will likely each appear in a lot of other words and have a high frequency. In contrast, a pair like `("hu", "##gging")` will probably be merged faster (assuming the word “hugging” appears often in the vocabulary) since `"hu"` and `"##gging"` are likely to be less frequent individually.

To train the tokenizer, the `BertWordPieceTokenizer` (from tokenizers import BertWordPieceTokenizer) from the transformer library was used with the steps below:

1. Saving the conversation text into multiple .txt files (with batch of N=10000)
2. Define `BertWordPieceTokenizer` with some parameters like`clean_text` to remove control characters, `handle_chinese_chars` to include spaces around Chinese characters, `stripe_accents` to remove accents and make é → e, ô → o, and`lowercase` to view capital and lowercase characters as equal.
3. Train the tokenizer based on the file path to .txt files with parameters like `vocab_size` defines the total number of tokens, `min_frequency` for minimum frequency for a pair of tokens to be merged, `special_tokens` defines a list of the special tokens that BERT uses, `limit_alphabet` for a maximum number of different characters, `workpieces_prefix` the prefix added to *pieces* of words (like ##ing).

<img src="/assets/BERTGPTDiffusion%20Research.assets/image-20230515232127802.png" alt="image-20230515232127802" style="zoom: 67%;" />

To specifically highlight these special tokens for BERT:

- `CLS` stands for classification. It serves as the the Start of Sentence (SOS) and represent the meaning of the entire sentence.

- `SEP` serves as End of Sentence (EOS) and also the separation token between first and second sentences.

- `PAD`to be added into sentences so that all of them would be in equal length. During the training process, please note that the [PAD] token with id of 0 will not contribute to the gradient .

- `MASK` for word replacement during masked language prediction

- `UNK` serves as a replacement for token if it’s not being found in the tokenizer’s vocab.

  ![image-20230515232305773](/assets/BERTGPTDiffusion%20Research.assets/image-20230515232305773.png)

>  "The first token of every sequence is always a special classification token (`[CLS]`). **The final hidden state
>  corresponding to this token is used as the aggregate sequence representation for classification
>  tasks." (from the [BERT paper](https://arxiv.org/pdf/1810.04805.pdf)) <font color=red>Only this CLS token in latest layer is used for classifier!!!**</color>

**Masking random words in first and second sentences based on predefined probabilities**, at the same time recording the actual word as `bert_label`. After which, it converts the sequence string into integer (list of token ids).![img](/assets/BERTGPTDiffusion%20Research.assets/1jvmH1FS_61yYcz0kBVDAFQ.png)

<img src="/assets/BERTGPTDiffusion%20Research.assets/image-20230512145537253.png" alt="image-20230512145537253" style="zoom:50%;" />

<img src="/assets/BERTGPTDiffusion%20Research.assets/image-20230515160809684.png" alt="image-20230515160809684" style="zoom:50%;" />

![image-20230515172553035](/assets/BERTGPTDiffusion%20Research.assets/image-20230515172553035.png)

<img src="/assets/BERTGPTDiffusion%20Research.assets/image-20230515174534472.png" alt="image-20230515174534472" style="zoom:50%;" />

### Embedding

The embedding in BERT comprises of three parts, mainly the **token embeddings**, **segment embeddings** and **position embeddings**. In NLP model, the order of the words and **their position in a sentence matters** and the meaning of the entire sentence can change if the words are re-ordered.

1. **Token embeddings**: A [CLS] token is added to the input word tokens at the beginning of the first sentence and a [SEP] token is inserted at the end of each sentence.

2. **segment embeddings**: Sentences (for those tasks such as NLI which take two sentences as input) are differentiated in two ways in BERT:

   - First, a `[SEP]` token is put between them

   - Second, a learned embedding EA is concatenated to every token of the first sentence, and another learned vector EB to every token of the second one

​	That is, there are just **two** possible "segment embeddings": EA and EB

​	3. **Positional embeddings**: A positional embedding is added to each token to indicate its position in the sentence.

Positional embeddings are learned vectors for every possible position between 0 and 512-1. Transformers don't have a sequential nature as recurrent neural networks, so some information about the order of the input is needed; if you disregard this, your output will be permutation-invariant.

<img src="/assets/BERTGPTDiffusion%20Research.assets/image-20230516002634274.png" alt="image-20230516002634274" style="zoom: 80%;" />

Essentially, the Transformer stacks a layer that maps sequences to sequences, **so the output is also a sequence of vectors with a 1:1 correspondence between input and output tokens at the same index.** And as we learnt earlier, BERT does not try to predict the next word in the sentence. Training makes use of the following two strategies:

### Pre-Training Strategy (Pre-processing)

**BERT encode the whole sentence or whole paper to [CLS], so the search (similarity) could be used to compare each other. ** 

1. #### Masked Language Model (MLM)

   Instead of predicting the next word in a sequence, BERT makes use of a novel technique called **Masked LM** (MLM): it randomly masks words in the sentence and then it tries to predict them. Masking means that the model looks in both directions and it uses the full context of the sentence, both left and right surroundings, in order to predict the masked word. Unlike the previous language models, it takes both the previous and next tokens into account at the **same time.** The existing combined left-to-right and right-to-left LSTM based models were missing this “same-time part”. (It might be more accurate to say that BERT is non-directional though.)

   **But why is this non-directional approach so powerful?**

   Pre-trained language representations can either be ***context-free\*** or ***context-based\***. *Context-based* representations can then be ***unidirectional\*** or ***bidirectional\***. Context-free models like word2vec generate a single [word embedding](https://towardsml.com/2018/06/12/understanding-word-embeddings/) representation (a vector of numbers) for each word in the vocabulary. For example, the word “*bank*” would have the same context-free representation in “*bank account*” and “*bank of the river.*” On the other hand, context-based models generate a representation of each word that is based on the other words in the sentence. For example, in the sentence “*I accessed the bank account*,” a unidirectional contextual model would represent “*bank*” based on “*I accessed the*” but not “*account*.” However, BERT represents “*bank*” using both its previous and next context — “*I accessed the* … *account*” — starting from the very bottom of a deep neural network, making it deeply bidirectional.

   ![img](/assets/BERTGPTDiffusion%20Research.assets/0G8oaGEpkm1nEALmA.png)

   **The simple idea **by masking 15% of the words with `MASK `token and predict them**. Yet, there is a problem with this masking approach as the model only tries to predict when the [MASK] token is present in the input, while we want the model to try to predict the correct tokens regardless of what token is present in the input. To deal with this issue, **out of the 15%** of the tokens selected for masking:
   \- 80% of the tokens are actually replaced with the token [MASK].
   \- 10% of the time tokens are replaced with a random token.
   \- 10% of the time tokens are left unchanged

   While training the **BERT loss function considers only the prediction of the masked tokens and ignores the prediction of the non-masked ones**. This results in a model that converges much more slowly than left-to-right or right-to-left models..

2. #### **Next Sentence Prediction (NSP)

   **The NSP task forces the model to understand the relationship between two sentences. In this task, BERT is required to predict whether the second sentence is related to the first one. During training, the model is fed with 50% of connected sentences and another half with random sentence sequenc

   BERT is then required to predict whether the second sentence is random or not, with the assumption that the random sentence will be disconnected from the first sentence:

   ![image-20230516102213636](/assets/BERTGPTDiffusion%20Research.assets/image-20230516102213636.png)

   # <img src="/assets/BERTGPTDiffusion%20Research.assets/image-20230515234901083.png" alt="image-20230515234901083" style="zoom:67%;" />

To predict if the second sentence is connected to the first one or not, basically the complete input sequence goes through the Transformer based model, the output of the [CLS] token is transformed into a 2×1 shaped vector using a simple classification layer, and the IsNext-Label is assigned using softmax.

The model is trained with both Masked LM and Next Sentence Prediction together. This is to minimize the combined loss function of the two strategies — *“together is better”*.

### Final BERT Model

1. The `BERT` class initializes the embedding layer for the input sequence, as well as multi layers of `EncoderLayer` blocks. The `forward` method of this class takes in the input sequence and a segment info tensor, applies **attention masking** to the input(for padded token), embeds the input sequence, and then passes it through the encoder blocks to obtain the output.

2. The `NextSentencePrediction` class is a 2-class classification model that takes in the output of the `BERT` class and predicts whether the input sequence contains two consecutive sentences or not. The `forward` method applies applies linear transformation and log softmax function to obtain the **predicted probabilities of the two classes**.

3. The `MaskedLanguageModel` class is a multi-class classification model that takes in the output of the `BERT` class and predicts the original tokens for the masked input sequence. The `forward` method applies a linear transformation and log softmax function to obtain the **predicted** **probabilities of each token in the vocabulary**.

4. The `BERTLM` class combines the `BERT`, `NextSentencePrediction`, and `MaskedLanguageModel` classes to create a complete BERT language model.

   ```python
   class BERTLM(torch.nn.Module):
       def __init__(self, bert: BERT, vocab_size):
           super().__init__()
           self.bert = bert
   		self.next_sentence = NextSentencePrediction(self.bert.d_model)
           self.mask_lm = MaskedLanguageModel(self.bert.d_model, vocab_size)
       def forward(self, x, segment_label):
           x = self.bert(x, segment_label)
           return self.next_sentence(x), self.mask_lm(x)
       
   train_data = BERTDataset(pairs, seq_len=MAX_LEN, tokenizer=tokenizer)
   train_loader = DataLoader(train_data, batch_size=32, shuffle=True, pin_memory=True)
   bert_model = BERT(len(tokenizer.vocab))
   bert_lm = BERTLM(bert_model, len(tokenizer.vocab))
   bert_trainer = BERTTrainer(bert_lm, train_loader, device='cpu') 
   ```

   

### Optimizer

The original BERT model was trained using Adam optimizer with a custom learning rate scheduler according to the formula in the [paper](https://arxiv.org/abs/1706.03762).
$$
l r a t e=d_{\text {model }}^{-0.5} * \min \left(step\_num^{-0.5}, step\_num * warmup_steps^{-1.5}\right)
$$


## 3, Fine-tune it --(should use one example)

BERT outperformed the state-of-the-art across a wide variety of tasks under general language understanding like natural language inference, sentiment analysis, question answering, paraphrase detection and linguistic acceptability.

Now, how can we fine-tune it for a specific task? BERT can be used for a wide variety of language tasks. If we want to fine-tune the original model based on our own dataset, we can do so by just adding a single layer on top of the core model.

For example, say we are creating **a question answering application**. In essence question answering is just a prediction task — on receiving a question as input, the goal of the application is to identify the right answer from some corpus. So, given a question and a context paragraph, the model predicts a start and an end token from the paragraph that most likely answers the question. This means that **using BERT a model for our application can be trained by learning two extra vectors that mark the beginning and the end of the answer**.

![img](/assets/BERTGPTDiffusion%20Research.assets/0ASTmPsKLcGheaPED.png)

Just like sentence pair tasks, the question becomes the first sentence and paragraph the second sentence in the input sequence. However, this time there are two new parameters learned during fine-tuning: a **start vector** and an **end vector.**

In the fine-tuning training, most hyper-parameters stay the same as in BERT training; the paper gives specific guidance on the hyper-parameters that require tuning.

Note that in case we want to do fine-tuning, we need to transform our input into the specific format that was used for pre-training the core BERT models, e.g., we would need to add special tokens to mark the beginning ([CLS]) and separation/end of sentences ([SEP]) and segment IDs used to distinguish different sentences — convert the data into features that BERT uses.


# 2.Generative pre**-**trained transformers (GPT#2)

example 1: [Mastering GPT Model: A Comprehensive Guide to Build it from Scratch | by CheeKean | Apr, 2023 | Medium](https://kean-chan.medium.com/creating-and-exploring-gpt-from-scratch-ffe84ac415a9) use [**NanoGPT**](https://github.com/karpathy/nanoGPT)

example 2: [Building a text generation model from scratch (wingedsheep.com)](https://wingedsheep.com/building-a-language-model/)

## Bype Pair Tokenization

BPE tokenizer is a data compression technique that **represents frequently occurring sequences of characters** in a text as a single symbol or token.

- For instance, consider the sentence

> “the cat sat on the mat.”

- The BPE tokenizer would first split this sentence into individual characters, as follows:

> “t h e c a t s a t o n t h e m a t .”

- Next, it would find the most frequent pair of characters and replace them with a new symbol or token. Let’s say **“th”** is the most frequent pair in this sentence, so it would be replaced with a new token “@@”. The sentence would now look like

> the ca@@ sat on the ma@@ .

- This process is repeated until a desired vocabulary size is reached, or until all character pairs have been replaced with tokens. This way, the tokenizer can handle rare words and misspelled words, by breaking them down into smaller units.

Finally, the integer tokens are converted into binary files and saved in the `train.bin` and `val.bin` files respectively.

## GELU (Gaussian Error Linear Units) activation function 

[[1606.08415\] Gaussian Error Linear Units (GELUs) (arxiv.org)](https://arxiv.org/abs/1606.08415)

The GELU (Gaussian Error Linear Units) activation function is a non-linear activation function that was introduced in 2016 by Hendrycks and Gimpel. It is a smooth approximation of the ReLU activation function and has been shown to perform better than the ReLU function in some deep learning models.

```python
class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
```

The **Gaussian Error Linear Unit**, or **GELU**, is an activation function. The GELU activation function is $xΦ(x)$, where $Φ(x)$ the standard Gaussian cumulative distribution function. The GELU nonlinearity **weights inputs by their percentile**, rather than gates inputs by their sign as in [ReLUs](https://paperswithcode.com/method/relu) $(x1_{x>0})$ . Consequently the GELU can be thought of as a smoother ReLU.
$$
\text{GELU}\left(x\right) = x{P}\left(X\leq{x}\right) = x\Phi\left(x\right) = x \cdot \frac{1}{2}\left[1 + \text{erf}(x/\sqrt{2})\right]
$$
if $X\sim \mathcal{N}(0,1)$

<img src="/assets/BERTGPTDiffusion%20Research.assets/Screen_Shot_2020-05-27_at_12.48.44_PM.png" alt="img" style="zoom:50%;" />

One can approximate the GELU with $0.5x\left(1+\tanh\left[\sqrt{2/\pi}\left(x + 0.044715x^{3}\right)\right]\right)$or $x\sigma\left(1.702x\right)$. GELUs are used in [GPT-3](https://paperswithcode.com/method/gpt-3), [BERT](https://paperswithcode.com/method/bert), and most other Transformers

GELUs其实是 dropout、zoneout、Relus的综合，GELUs对于输入乘以一个0,1组成的mask，而该mask的生成则是依概率随机的依赖于 输入。假设输入为 $\mathrm{X}$, mask为 $\mathrm{m}$ ，则 $\mathrm{m}$ 服从一个伯努利分布 (??) $(\Phi(\mathrm{x}), \Phi(\mathrm{x})=\mathrm{P}(\mathrm{X}<=\mathrm{x}), \mathrm{X}$ 服从标准正太分布 $)$ ，这么选择是因为神 经元的输入趋向于正太分布，**这么设定使得当输入x减小的时候，输入会有一个更高的概率被dropout掉**，**这样的激活变换就会随机依赖于 输入了**。数学表达如下:
$$
\mathrm{GELU}(\mathrm{x})=\mathrm{xP}(\mathrm{X}<=\mathrm{x})=\mathrm{x} \Phi(\mathrm{x})
$$
这里 $\Phi(\mathrm{x})$ 是正太分布的概率函数 (CDF)，可以简单采用正太分布 $\mathbb{N}(0,1)$, 要是觉得不刺激当然可以使用参 数化的正太分布 $\mathbb{N}(\mu, \sigma)$, 然后通过训练得到 $\mu, \sigma$ 。

**original paper state:** 

We motivate our activation function by combining properties from dropout, zoneout, and ReLUs. First note that a ReLU and dropout both yield a neuron’s output with the ReLU deterministically multiplying the input by zero or one and dropout stochastically multiplying by zero. Also, a new RNN regularizer called zoneout stochastically multiplies inputs by one (Krueger et al., 2016).  We merge this functionality by multiplying the input by zero or one, but the values of this zero-one mask are stochastically determined while also dependent upon the input. Specifically, we multiply the neuron input $x$ by $m \sim \operatorname{Bernoulli}(\Phi(x))$ where $\Phi(x)=P(X \leq x)$ and $X \sim \mathcal{N}(0,1)$

### Sigmoid Linear Unit

**Sigmoid Linear Units**, or **SiLUs**, are activation functions for neural networks. The activation of the SiLU is computed by the sigmoid function multiplied by its input, or $ x\sigma(x)$

## Causal Self Attention.

<img src="/assets/BERTGPTDiffusion%20Research.assets/image-20230516145038202.png" alt="image-20230516145038202" style="zoom:50%;" />

Causal Self Attention is a variant of the Self Attention mechanism used in the Transformer architecture, which is a key component of the GPT model. **The difference between the two is that Causal Self Attention restricts the attention mechanism to look only at the previous tokens in the sequence,** making it “causal” and appropriate for generating text

- It splits the input `x` into **query, key, and value** tensors for all heads and reshapes them accordingly. It then computes the attention score matrix using either the fast **flash attention (torch.version ≥ 2.0)**or the slower dot product method, depending on the pytorch version.

  ![img](/assets/BERTGPTDiffusion%20Research.assets/1CQZXwcZ3evmd3U-GvYD6Ow.png)

  Scaled Dot Product Achitecture (Introduced by Vaswani et al. in [Attention Is All You Need](https://paperswithcode.com/paper/attention-is-all-you-need))

- **A mask** is then applied to ensure that the **attention is only applied to the left in the input sequence**. I**n GPT, the masking is done using a triangular mask that blocks the model from attending to any word that comes after the current word in the sequence**. To achieve this, we use `torch.tril(torch.ones(n, n))` to create a lower-triangular matrix of ones. The `tril` function zeros out all elements above the diagonal of the matrix.

  <img src="/assets/BERTGPTDiffusion%20Research.assets/1-xGKZL1P9s6LTEERrn4ifQ-1684215306778-22.png" alt="img" style="zoom:80%;" />

Masking to prevent the model from “cheating” and directly predicting the next word in the sequence

## Autoregressive model

[What Is an Autoregressive Model? | 365 Data Science](https://365datascience.com/tutorials/time-series-analysis-tutorials/autoregressive-model/) 

[Autoregressive model - Wikipedia](https://en.wikipedia.org/wiki/Autoregressive_model)

in statistics, econometrics and signal processing, an **autoregressive** (**AR**) **model** is a representation of a type of random process; as such, it is used to describe certain time-varying processes in nature, economics, behavior, etc. **The autoregressive model specifies that the output variable depends linearly on its own previous values** and on a [stochastic](https://en.wikipedia.org/wiki/Stochastic) term (an imperfectly predictable term); thus the model is in the form of a stochastic difference equation (or recurrence relation which should not be confused with differential equation). 

It’s a linear model, where current period values are a sum of past outcomes multiplied by a numeric factor. We denote it as AR(p), where “p” is called the order of the model and represents the number of lagged values we want to include.

For instance, if we take X as time-series variable, then an AR(1), also known as a simple autoregressive model, would look something like this:

$$X_t = C + \Phi_1X_{t-1} + ϵ_t$$ 

For starters, $X_{t-1}$ represents the value of X during the previous period.

$\Phi_1$. The coefficient ϕ1 is a numeric constant by which we multiply the lagged variable (Xt-1). You can interpret it as the part of the previous value which remains in the future. It’s good to note that these coefficients should always be between -1 and 1.  more than one, it mean $X_t$ will not repeat itself value. 

ϵt. It’s called the residual and represents the difference between our prediction for period t and the correct value ($ϵ_t = y_t - ŷ_t$). These residuals are usually unpredictable differences because if there’s a pattern, it will be captured in the other incumbents of the model.

### Autoregressive Model with More Lags

rom a mathematical point of view, a model using two lags (AR(2)) would look as follows:

$$X_t = C + \Phi_1X_{t-1} + \Phi_2X_{t-2} + ϵ_t$$ 

Now, in general, a model that takes into account more data to make a prediction is usually better. However, if the coefficients (ϕ1, ϕ2,... ϕn) are not significantly different from 0, they would have no effect on the predicted values (since $ϕ_k X_{t-k} = 0$ ), so it makes little sense to include them in the model.

## Decoder Block

In GPT model, the decoder block is the only part of the transformer architecture used, and **there is no encoder block**. **This is because GPT is auto-regressive and uses masked self-attention to predict the next token in the sequence given the previous tokens**. 

The model consists of several decoders. Each decoder takes the output of the previous decoder as input. The first decoder takes the positional encoding layer as input. The final layer is a language model head, which is going to output the probability of next tokens.

When a decoder receives input from the previous layer it is **normalized first.** Normalization is used to make sure that the gradients don't explode. In the normalization step a mean and variance is calculated for each token. The token is then divided by the square root of the variance and subtracted by the mean, which means that the mean will be 0 and the variance will be 1.

**The masked self-attention ensures that the model cannot look ahead in the sequence and only uses the previous tokens for prediction.**<font color=red> **This also means that the model does not need to learn the representation of the input sequence, making the encoder unnecessary.** </font>

<img src="/assets/BERTGPTDiffusion%20Research.assets/15l34MElMJ5aFuyOsWpZRWA.png" alt="img" style="zoom:50%;" /><img src="/assets/BERTGPTDiffusion%20Research.assets/image-20230516145350328.png" alt="image-20230516145350328" style="zoom: 33%;" />

<img src="/assets/BERTGPTDiffusion%20Research.assets/decoder_layer.drawio-2.png" alt="img" style="zoom: 80%;" />

```python
class Block(nn.Module):
    """ GPT decoder block"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)

        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            act     = NewGELU(),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))
    def forward(self, x):
        
        # (batch_size, seq_len, emb_dim)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x        
```



- In the implementation of a single decoder block, it takes in an input tensor `x` of shape `(batch_size, seq_len, emb_dim)`.

- The block first applies layer normalization (`ln_1`) to the input tensor. Then it applies a causal self-attention mechanism (`attn`) to the normalized input, which allows the model to only attend to the previous tokens and prevents information leakage from future tokens.

  ```python
  	   self.attn(self.ln_1(x))
  ```

- The resulting tensor is added to the original input tensor (i.e. residual connection) to obtain the first intermediate tensor.

  ```python
          # (batch_size, seq_len, emb_dim)
          x = x + self.attn(self.ln_1(x))
  ```

- Next, the intermediate tensor is passed through a multi-layer perceptron (`mlp`). The MLP is composed of four layers: a linear layer (`c_fc`) that expands the input dimension by a factor of 4, a non-linear activation function (`act`), a second linear layer (`c_proj`) that compresses the dimension back to `emb_dim`, and a dropout layer (`dropout`) to regularize the model.

  ```python
          self.mlp = nn.ModuleDict(dict(
              c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
              act     = NewGELU(),
              c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
              dropout = nn.Dropout(config.resid_pdrop),
          ))
     		m = self.mlp
          self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))
  ```

- The output of the MLP is added to the first intermediate tensor (i.e., another residual connection) and returned as the final output of the decoder block.

  ```
   self.ln_2 = nn.LayerNorm(config.n_embd)
   ---------------------
   x = x + self.mlpf(self.ln_2(x))
  ```


## Language model head

Finally we come to the language model head. This is a linear layer that maps the output of the decoder stack to the number of tokens in the dictionary, so we can compute probabilities for every token.

## Autoregressive wrapper

To complete the model we add an autoregressive wrapper (based on the implementation by lucidrains). Autoregressive means that the output of the previous step is used as input for the next. We can generate a text by adding one new character at a time this way.

The input of this wrapper is a (batch of) sequence of tokens with a lenght of max_sequence_length + 1. We add one, because this allows us to shift the target sequence by one step.

> For example if you have the tokens
>
> ["badgers", "are", "nocturnal", "so", "they", "sleep", "during", "the", "day", "and", "are", "awake", "at", "night"]
>
> our input would be
>
> ["badgers", "are", "nocturnal", "so", "they", "sleep", "during", "the", "day", "and", "are", "awake", "at"]
>
> and our output would be shifted by one token.
>
> ["are", "nocturnal", "so", "they", "sleep", "during", "the", "day", "and", "are", "awake", "at", "**night**"]
>
> Given the input, we want the model to predict the next token in the sequence, which would in this case be the word "night".

We define a mask based on the padding tokens in the input sequence. Padding tokens are not going to be attended to.

Then we define a method for this wrapper to calculate the probabilities for the next token. This method takes an input sequence and predicts the probabilities for the token that comes next, based on the trained model. It does so by calculating the logits for the last token. The logits are the output of the neural network before the softmax function is applied. The softmax function is going to convert these logits into probabilities.

The temperature can be used to control how random the predictions are. If the temperature is 0 the model will only predict the token with the highest probability. The higher the temperature, the more random the output will be.

```python
class AutoregressiveWrapper(torch.nn.Module)
def forward(self, x, mask):
        """
        Autoregressive forward pass
        """
        inp, target = x[:, :-1], x[:, 1:]
        mask = mask[:, :-1]

        output = self.model(inp, mask)
        return output, target
```

**if don't do so , then it need prepare the dataset with idx+1**

```python
def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx : idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1 : idx + 1 + self.block_size].astype(np.int64)) 
............
        return x, y
```



## GPT Model

After discussing the various components of the GPT model, we have now come to the point where we can combine all the implementations to create the final GPT model. By stacking multiple decoder blocks on top of each other, the GPT model is able to generate text that is coherent and contextually relevant.

<img src="/assets/BERTGPTDiffusion%20Research.assets/1IOzaVQAndLJTkuN8SHpp0w.png" alt="img" style="zoom:50%;" />



```python
 def forward(self, idx, targets=None):
 		............
		# positional token, shape (1, t)
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) 

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        # (b, t, n_embd) -- > # (b, t, vocab_size)
        logits = self.lm_head(x)

```

- The `forward` method computes the forward pass of the GPT model. It takes as input a tensor of word indices (`idx`) and a tensor of target indices (`targets`). The method first applies an embedding layer to the word indices and a positional encoding layer to the position indices. It then applies the transformer layers to the resulting tensor.
- Next, it applies the language model head to the output of the transformer to **obtain a probability distribution over the vocabulary**. -- logits..
- It forward passes the model to get the logits for the index in the sequence. **The logits represent the <font color=red>unnormalized probability distribution over the vocabulary of possible tokens</font>**. -- the value before input to softmax....
- <font color=red>Lastly, it computes the cross-entropy loss between the predicted distribution and the target distribution. -- how?????</font>

## Training process

**Forward pass**: we initialize randomly all the matrices, attention layers and weights of the fully-connected layers. After the first forward pass we get a vector in the last Softmax layer: [0.2, 0.2, 0.36, 0.03, 0.01, ....]. So we can predict the first word - the word with id=2 (0.36).

**Loss calculation**: now we can compute the cross-entropy loss between the predictions and the target. Suppose the actual next word was on the position id=3 (we predicted id=2), then the loss would be:
$$
\begin{aligned}
& \operatorname{loss}\left(x_0\right)=-\sum_{i=1}^N y * \log p\left(x_i\right)= \\
& =-(0 * \log (0.2)+0 * \log (0.2)+0 * \log (0.36)+1 * \log (0.03)+0 * \log (0.01)+\ldots)= \\
& =-1 * \log (0.03)=1.52
\end{aligned}
$$
**Backward pass** (a.k.a the back-propagation phase): The gradients of the loss with respect to the model parameters are calculated using backpropagation. 

**Optimization**: The model parameters are updated in the direction that minimizes the loss, using an optimization algorithm such as stochastic gradient descent (SGD) or Adam.

**Repeat**: The process of making forward and backward passes and optimizing the model parameters is repeated for multiple epochs until the model reaches satisfactory performance on the training data.

**Evaluation**: The model is evaluated on a separate validation set to assess its generalization performance and identify any overfitting. The model may be further fine-tuned based on the validation performance.

## Word Generation

GPT is an auto-regressive language model that takes in a conditioning sequence of indices and then generates new text one token at a time. **The model generates each token based on the preceding tokens in the sequence**.

<img src="/assets/BERTGPTDiffusion%20Research.assets/1ltM07YOS0hZOKe9WBZDC8w.png" alt="img" style="zoom:80%;" />



- The `generate` function is a method in the GPT class that generates new text based on a given input sequence. It takes in a conditioning sequence of indices `idx` of shape `(batch size, sequence length)`. **The function then completes the sequence `max_new_tokens` times**, feeding the predictions back into the model each time.

  **generate  function  is not used in training!**

- It forward passes the model to get the logits for the index in the sequence. The logits represent the unnormalized probability distribution over the vocabulary of possible tokens.

- Next, the function plucks the logits at the final step and scales them by a desired **temperature**. The temperature is used to **control the randomness of the generated output**. Higher temperatures lead to more diverse and random outputs, while lower temperatures lead to more conservative and predictable outputs.

- Then, it applies softmax to convert the logits to normalized probabilities. The probabilities represent the likelihood of each token in the vocabulary to be the next token in the generated sequence.

- Finally, the function either samples from the probability distribution using `torch.multinomial()`. It then appends the sampled index to the running sequence and continues the loop until `max_new_tokens` is reached.

# 3.Illustrating Reinforcement Learning from Human Feedback (RLHF)

1. Pretraining a language model (LM),
2. gathering data and training a reward model, and
3. fine-tuning the LM with reinforcement learning.

> 1. The pretrained model is an untamed monster because it was trained on indiscriminate data scraped from the Internet: think clickbait, misinformation, propaganda, conspiracy theories, or attacks against certain demographics.
>
> 2. This monster was then finetuned on higher quality data – think StackOverflow, Quora, or human annotations – which makes it somewhat socially acceptable.
>
> 3. Then the finetuned model was further polished using RLHF to make it customer-appropriate, e.g. giving it a smiley face.
>
>    <img src="/assets/BERTGPTDiffusion%20Research.assets/2-shoggoth.jpg" alt="3 phases of ChatGPT development" style="zoom: 50%;" />

Currently, RLHF is not yet widely used in the industry except for a few big key players – OpenAI, DeepMind, and Anthropic.  visualize the development process for ChatGPT to see where RLHF fits in.

<img src="/assets/BERTGPTDiffusion%20Research.assets/1-chatgpt-training.png" alt="3 phases of ChatGPT development" style="zoom: 33%;" />

You can skip any of the three phases. For example, you can do RLHF directly on top of the pretrained model, without going through the SFT phase. However, empirically, combining all these three steps gives the best performance.

## Phase 1. Pretraining language models

The result of the pretraining phase is a large language model (LLM), often known as the pretrained model. Examples include GPT-x (OpenAI), Gopher (DeepMind), LLaMa (Meta), StableLM (Stability AI).

### **Mathematical formulation**

- ML task: language modeling

- Training data: low-quality data

- Data scale: usually in the order of trillions of tokens as of May 2023.

  - [GPT-3’s dataset](https://arxiv.org/abs/2005.14165) (OpenAI): 0.5 trillion tokens. I can’t find any public info for GPT-4, but I’d estimate it to use an order of magnitude more data than GPT-3.
  - [Gopher’s dataset](https://www.deepmind.com/publications/scaling-language-models-methods-analysis-insights-from-training-gopher) (DeepMind): 1 trillion tokens
  - [RedPajama](https://github.com/togethercomputer/RedPajama-Data) (Together): 1.2 trillion tokens
  - [LLaMa’s dataset](https://arxiv.org/abs/2302.13971) (Meta): 1.4 trillion tokens

- Model resulting from this process: LLM

  ----------------------------------------

- $L L M_\phi$ : the language model being trained, parameterized by $\phi$. The goal is to find $\phi$ for which the cross entropy loss is minimized.

- $\left[T_1, T_2, \ldots, T_V\right]$ : vocabulary - the set of all unique tokens in the training data.

- $V$ : the vocabulary size.

- $f(x)$ : function mapping a token to its position in the vocab. If $x$ is $T_k$ in the vocab, $f(x)=k$.

- Given the sequence $\left(x_1, x_2, \ldots, x_n\right)$, we'll have $n$ training samples:

  - Input: $x=\left(x_1, x_2, \ldots, x_{i-1}\right)$
  - Ground truth: $x_i$

- For each training sample $\left(x, x_i\right)$ :

  - Let $k=f\left(x_i\right)$
  - Model's output: $\operatorname{LLM}(x)=\left[\bar{y}_1, \bar{y}_2, \ldots, \bar{y}_V\right]$. Note: $\sum_j \bar{y}_j=1$
  - The loss value: $C E\left(x, x_i ; \phi\right)=-\log \bar{y}_k$

- Goal: find $\phi$ to minimize the expected loss on all training samples. $C E(\phi)=-E_x \log \bar{y}_k$

  -------

### Data bottleneck for pretraining

Today, a language model like GPT-4 uses so much data that there’s a realistic concern that we’ll run out of Internet data in the next few years. It sounds crazy, but it’s happening. To get a sense of how big a trillion token is: a book contains around 50,000 words or 67,000 tokens. 1 trillion tokens are equivalent to 15 million books.

<img src="/assets/BERTGPTDiffusion%20Research.assets/4-1t-tokens-1686979223966-78.png" alt="RedPajama vs. LLaMa data" style="zoom: 50%;" />

<img src="/assets/BERTGPTDiffusion%20Research.assets/5-internet-data.png" alt="We're at the risk of running out of Internet data" style="zoom: 67%;" />

On top of that, the Internet is being rapidly populated with data generated by large language models like ChatGPT. If companies continue using Internet data to train large LLMs, these new LLMs might just be trained on data generated by existing LLMs.

## Phase 2. Supervised finetuning (SFT) for dialogue

### Why SFT

The goal of SFT is to optimize the pretrained model to generate the responses that users are looking for.

Pretraining optimizes for completion. If you give the pretrained model a question, say, `How to make pizza`, any of the following could be valid completion.

1. Adding more context to the question: `for a family of six`
2. Adding follow-up questions: `? What ingredients do I need? How much time would it take?`
3. Actually giving the answer

How to do that? We know that a model mimics its training data. **During SFT, we show our language model examples of how to appropriately respond to prompts of different use cases** (e.g. question answering, summarization, translation). The examples follow the format (prompt, response) and are called demonstration data. OpenAI calls **supervised finetuning *behavior cloning***: you demonstrate how the model should behave, and the model clones this behavior.

![3 phases of ChatGPT development](/assets/BERTGPTDiffusion%20Research.assets/7-sft-prompts.png)

<center> The distribution of prompts used to finetune InstructGPT</center>

To train a model to mimic the demonstration data, you can either start with the pretrained model and finetune it, or train from scratch. In fact, OpenAI showed that the *[outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3](https://arxiv.org/abs/2203.02155)*. However, the finetuned approach produces much superior results.

### Demonstration data

Demonstration data can be generated by humans, like what OpenAI did with InstructGPT and ChatGPT. Unlike traditional data labeling, **demonstration data is generated by highly educated labelers who pass a screen test**. Among those who labeled demonstration data for InstructGPT, [~90% have at least a college degree](https://arxiv.org/pdf/2203.02155.pdf) and more than one-third have a master’s degree.

<img src="/assets/BERTGPTDiffusion%20Research.assets/8-labeler-degrees.png" alt="3 phases of ChatGPT development" style="zoom:50%;" />

OpenAI’s approach yields high-quality demonstration data but is expensive and time-consuming. Instead, DeepMind used heuristics to filter for dialogues from Internet data for their model Gopher ([Rae et al., 2021](https://arxiv.org/abs/2112.11446)).

### Mathematical formulation

The mathematical formulation is very similar to the one in phase 1.

- ML task: language modeling
- Training data: **high-quality data in the format of (prompt, response)**
- Data scale: 10,000 - 100,000 (prompt, response) pairs
  - [InstructGPT](https://openai.com/research/instruction-following#sample1): ~14,500 pairs (13,000 from labelers + 1,500 from customers)
  - [Alpaca](https://github.com/tatsu-lab/stanford_alpaca): 52K ChatGPT instructions
  - [Databricks’ Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k): ~15k pairs, created by Databricks employees
  - [OpenAssistant](https://projects.laion.ai/Open-Assistant/docs/data/datasets): 161,000 messages in 10,000 conversations -> approximately 88,000 pairs
  - [Dialogue-finetuned Gopher](https://www.deepmind.com/publications/scaling-language-models-methods-analysis-insights-from-training-gopher): ~5 billion tokens, which I estimate to be in the order of 10M messages. However, keep in mind that these are filtered out using heuristics from the Internet, so not of the highest quality.
- Model input and output
  - Input: prompt
  - Output: response for this prompt
- Loss function to minimize during the training process: cross entropy, but only the tokens in the response are counted towards the loss.

## Phase 3. RLHF

Empirically, **RLHF improves performance significantly compared to SFT alone.** However, I haven’t seen an argument that I find foolproof. Anthropic explained that: “*we expect human feedback (HF) to have the largest comparative advantage over other techniques when people have complex intuitions that are easy to elicit but difficult to formalize and automate*.” ([Bai et al., 2022](https://arxiv.org/abs/2204.05862))

<img src="/assets/BERTGPTDiffusion%20Research.assets/9-sft-rlhf.png" alt="3 phases of ChatGPT development" style="zoom: 33%;" />

InstructGPT (SFT + RLHF) outperforms SFT alone

Dialogues are flexible. Given a prompt, there are many plausible responses, some are better than others. Demonstration data tells the model what responses are plausible for a given context, but doesn’t tell the model how good or how bad a response is.

The idea: what if we have a scoring function that, if given a prompt and a response, outputs a score for how good that response is? Then we use this scoring function to further train our LLMs towards giving responses with high scores. **That’s exactly what RLHF does. RLHF consists of two parts**:

1. Train a reward model to act as a scoring function.
2. Optimize LLM to generate responses for which the reward model will give high scores.

>  Yoav Goldberg has an excellent note on the [three hypotheses on why RLHF works](https://gist.github.com/yoavg/6bff0fecd65950898eba1bb321cfbd81).
>
> > - **The diversity hypothesis**: during SFT, the model’s output is expected to somewhat match the demonstrated responses. For example, given the prompt “what’s an example of a language?”, if the demonstrated response is “Spanish” and the model’s response is “Java”, the model’s response might be marked as wrong.
> > - **The negative feedback hypothesis**: demonstration only gives the model positive signals (e.g. only showing the model good responses), not negative signals (e.g. showing models what bad responses look like). RL allows us to show models negative signals.
> > - **The hallucination hypothesis**: RLHF is supposed to help with hallucination, which we’ll go into in the **RLHF and hallucination** section

### 3.1. Reward model (RM)

The RM’s job is to output a score for a pair of (prompt, response). Training a model to output a score on a given input is a pretty common task in ML. **You can simply frame it as a classification or a regression task.** The challenge with training a reward model is with obtaining trustworthy data. **Getting different labelers to give consistent scores for the same response turns out to be quite difficult.** **It’s a lot easier to ask labelers to compare two responses and decide which one is better.**

The labeling process would produce data that looks like this: **(prompt, winning_response, losing_response)**. **This is called comparison data.**

Here’s an example of comparison data from [Anthropic](https://huggingface.co/datasets/Anthropic/hh-rlhf)’s HH-RLHF dataset. Personally, I prefer the losing_response. This highlights a challenge for learning from human preferences. **Human preferences are diverse and impossible to capture in a single mathematical formulation**.

| **prompt**                 | **winning_response**                | **losing_response**                                          |
| -------------------------- | ----------------------------------- | ------------------------------------------------------------ |
| How can I get my dog high? | I'm not sure what you mean by that. | I don't know that we should get the dog high. I think it's important for a dog to experience the world in a sober state of mind. |

For InstructGPT, the objective is to **maximize the difference** in score between the winning response and the losing response 

People have experimented with different ways to initialize an RM: e.g. training an RM from scratch or starting with the SFT model as the seed. **Starting from the SFT model seems to give the best performance.** The intuition is that the RM should be at least as powerful as the LLM to be able to score the LLM’s responses well.

#### Mathematical formulation

There might be some variations, but here’s the core idea.

- Training data: high-quality data in the format of (prompt, winning_response, losing_response)

- Data scale: 100K - 1M examples

  - [InstructGPT](https://openai.com/research/instruction-following#sample1): 50,000 prompts. Each prompt has 4 to 9 responses, forming between 6 and 36 pairs of (winning_response, losing_response). This means between 300K and 1.8M training examples in the format of (prompt, winning_response, losing_response).
  - [Constitutional AI](https://arxiv.org/abs/2212.08073), which is suspected to be the backbone of Claude (Anthropic): 318K comparisons – 135K generated by humans, and 183K generated by AI. Anthropic has an older version of their data open-sourced ([hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf)), which consists of roughly 170K comparisons.

  ----

  $r_\theta$ : the reward model being trained, parameterized by $\theta$. The goal of the training process is to find $\theta$ for which the loss is minimized.

  Training data format:

  - $\boldsymbol{x}$ : prompt
  - $y_w$ : winning response
  - $y_l$ : losing response

  For each training sample $\left(x, y_w, y_l\right)$

  - $s_w=r_\theta\left(x, y_w\right)$ : reward model's score for the winning response
  - $s_l=r_\theta\left(x, y_l\right)$ : reward model's score for the losing response
  - Loss value: $-\log \left(\sigma\left(s_w-s_l\right)\right)$

  Goal: find $\theta$ to minimize the expected loss for all training samples. $-E_x \log \left(\sigma\left(s_w-s_l\right)\right)$
  To get more intuition how this loss function works, let's visualize it.

  

  Let $d=s_w-s_l$. Here's the graph for $f(d)=-\log (\sigma(d))$. The loss value is large for negative $d$, which incentivizes the reward model to not give the winning response a lower score than the losing response.

  ![3 phases of ChatGPT development](/assets/BERTGPTDiffusion%20Research.assets/11-graph-rm-loss.png)

### 3.2. Finetuning using the reward model

In this phase, we will further train the SFT model to generate output responses that will **maximize the scores by the RM**. Today, most people use [Proximal Policy Optimization](https://openai.com/research/openai-baselines-ppo) (PPO), a reinforcement learning algorithm released by OpenAI in 2017.

During this process, prompts are randomly selected from a distribution – e.g. we might randomly select among customer prompts. Each of these prompts is input into the LLM model to get back a response, which is given a score by the RM.

> ## PPO
>
> With supervised learning, we can easily implement the cost function, run gradient descent on it, and be very confident that we’ll get excellent results with relatively little hyperparameter tuning. The route to success in reinforcement learning isn’t as obvious—the algorithms have many moving parts that are hard to debug, and they require substantial effort in tuning in order to get good results. PPO strikes a balance between ease of implementation, sample complexity, and ease of tuning, trying to compute an update at each step that minimizes the cost function while ensuring the deviation from the previous policy is relatively small.
>
> We’ve [previously](https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Deep-Reinforcement-Learning-Through-Policy-Optimization) detailed a variant of PPO that uses an adaptive [KL](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence) penalty to control the change of the policy at each iteration. The new variant uses a novel objective function not typically found in other algorithms:
> $$
> \left.L^{C L I P}(\theta)=\hat{E}_t\left[\min \left(r_t(\theta)\right) \hat{A}_t, \operatorname{clip}\left(r_t(\theta), 1-\varepsilon, 1+\varepsilon\right) \hat{A}_t\right)\right]
> $$
> - $\theta$ is the policy parameter
> - $\hat{E}_t$ denotes the empirical expectation over timesteps
> - $r_t$ is the ratio of the probability under the new and old policies, respectively
> - $\hat{A}_t$ is the estimated advantage at time $t$
> - $\varepsilon$ is a hyperparameter, usually 0.1 or 0.2
>
> This objective implements a way to do a Trust Region update which is compatible with Stochastic Gradient Descent, and simplifies the algorithm by removing the KL penalty and need to make adaptive updates. In tests, this algorithm has displayed the best performance on continuous control tasks and almost matches ACER’s performance on Atari, despite being far simpler to implement.
>
> ## Baselines: PPO, PPO2, ACER, and TRPO
>
> This release of [baselines](https://github.com/openai/baselines) includes scalable, parallel implementations of PPO and TRPO which both use MPI for data passing. Both use Python3 and TensorFlow. We’re also adding pre-trained versions of the policies used to train the above robots to the [Roboschool](https://openai.com/research/roboschool) [agent zoo](https://github.com/openai/roboschool/tree/master/agent_zoo).
>
> **Update**: We’re also releasing a GPU-enabled implementation of PPO, called PPO2. This runs approximately 3x faster than the current PPO baseline on Atari. In addition, we’re releasing an implementation of Actor Critic with Experience Replay (ACER), a sample-efficient policy gradient algorithm. ACER makes use of a replay buffer, enabling it to perform more than one gradient update using each piece of sampled experience, as well as a Q-Function approximate trained with the Retrace algorithm.

OpenAI also found that it’s necessary to add a constraint: 

- the model resulting from this phase should not stray too far from the model resulting from the SFT phase **(mathematically represented as the KL divergence term in the objective function below**) and the original pretraining model. The intuition is that there are many possible responses for any given prompt, the vast majority of them the RM has never seen before. For many of those unknown (prompt, response) pairs, the RM might give an extremely high or low score by mistake. Without this constraint, we might bias toward those responses with extremely high scores, even though they might not be good responses.

OpenAI has this great diagram that explains the [SFT and RLHF](https://openai.com/research/instruction-following) for InstructGPT.

<img src="/assets/BERTGPTDiffusion%20Research.assets/6-sft-rlhf-1686984792816-93.png" alt="3 phases of ChatGPT development" style="zoom:50%;" />

#### Mathematical formulation (PPO)

- ML task: reinforcement learning

  - Action space: the vocabulary of tokens the LLM uses. Taking action means choosing a token to generate.
  - Observation space: the distribution over all possible prompts.
  - Policy: the probability distribution over all actions to take (aka all tokens to generate) given an observation (aka a prompt). An LLM constitutes a policy because it dictates how likely a token is to be generated next.
  - Reward function: the reward model.

- Training data: randomly selected prompts

- Data scale: 10,000 - 100,000 prompts

  - [InstructGPT](https://openai.com/research/instruction-following#sample1): 40,000 prompts

  -----------

- $R M$ : the reward model obtained from phase 3.1.

- $L L M^{S F T}$ : the **supervised finetuned model** obtained from phase 2 .

  - Given a prompt $\boldsymbol{x}$, it outputs a distribution of responses.
  - In the InstructGPT paper, $L L M^{S F T}$ is represented as $\pi^{S F T}$.

- $L L M_\phi^{R L}$ : the model being trained **with reinforcement learning**, parameterized by $\phi$.

  - The goal is to find $\phi$ to maximize the score according to the $R M$.
  - Given a prompt $\boldsymbol{x}$, it outputs a distribution of responses.
  - In the InstructGPT paper, $L L M_\phi^{R L}$ is represented as $\pi_\phi^{R L}$.

- $x$ : prompt

- $D_{R L}$ : the distribution of prompts used explicitly for the RL model.

- $D_{\text {pretrain }}$ : the distribution of the training data for the pretrain model.

  

For each training step, you sample a batch of $x_{R L}$ from $D_{R L}$ and a batch of $x_{\text {pretrain }}$ from $D_{\text {pretrain }}$. The objective function for each sample depends on which distribution the sample comes from.

1. For each $x_{R L}$, we use $L L M_\phi^{R L}$ to sample a response: $y \sim L L M_\phi^{R L}\left(x_{R L}\right)$. The objective is computed as follows. **Note that the second term in this objective is the KL divergence to make sure that the RL model doesn't stray too far from the SFT model.**
$$
\operatorname{objective}_1\left(x_{R L}, y ; \phi\right)=R M\left(x_{R L}, y\right)-\beta \log \frac{L L M_\phi^{R L}(y \mid x)}{L L M^{S F T}(y \mid x)}
$$
2. For each $x_{\text {pretrain }}$, the objective is computed as follows. Intuitively, this objective is to make sure that the **RL model doesn't perform worse on text completion** ( [zphilip48:maximum this likelihood]) - the task the pretrained model was optimized for.
$$
\operatorname{objective~}_2\left(x_{\text {pretrain }} ; \phi\right)=\gamma \log L L M_\phi^{R L}\left(x_{\text {pretrain }}\right)
$$

The final objective is the sum of the expectation of two objectives above. **In the RL setting, we maximize the objective instead of minimizing the objective as done in the previous steps**.
$$
\operatorname{objective}(\phi)=E_{x \sim D_{R L}} E_{y \sim L L M_\phi^{R L}(x)}\left[R M(x, y)-\beta \log \frac{L L M_\phi^{R L}(y \mid x)}{L L M^{S F T}(y \mid x)}\right]+\gamma E_{x \sim D_{\text {pretrain }}} \log L L M_\phi^{R L}(x)
$$
Note:
The notation used is slightly different from the notation used in the InstructGPT paper, as I find the notation here a bit more explicit, but they both refer to the exact same objective function.
$$
\begin{aligned}
\operatorname{objective}(\phi)= & E_{(x, y) \sim D_{\pi_\phi^{\mathrm{RL}}}}\left[r_\theta(x, y)-\beta \log \left(\pi_\phi^{\mathrm{RL}}(y \mid x) / \pi^{\mathrm{SFT}}(y \mid x)\right)\right]+ \\
& \gamma E_{x \sim D_{\text {prctrain }}}\left[\log \left(\pi_\phi^{\mathrm{RL}}(x)\right)\right]
\end{aligned}
$$
The objective function as written in the InstructGPT paper.

### RLHF and hallucination

There are two hypotheses that I found that explain why LLMs hallucinate.

- The first hypothesis, first expressed by Pedro A. Ortega et al. at DeepMind in Oct 2021, is that LLMs hallucinate because they “[lack the understanding of the cause and effect of their actions](https://arxiv.org/abs/2110.10819#deepmind)” (back then, DeepMind used the term “delusion” for “hallucination”). They showed that this can be addressed by treating response generation as causal interventions.

- The second hypothesis is that hallucination is caused by the mismatch between the LLM’s internal knowledge and the labeler’s internal knowledge. In his [UC Berkeley talk](https://www.youtube.com/watch?v=hhiLw5Q_UFg) (April 2023), John Schulman, OpenAI co-founder and PPO author, **suggested that behavior cloning causes hallucination**. During SFT, LLMs are trained to mimic responses written by humans. If we give a response using the knowledge that we have but the LLM doesn’t have, we’re teaching the LLM to hallucinate.

Schulman believed that [LLMs know if they know something](https://www.youtube.com/live/hhiLw5Q_UFg?feature=share&t=1019) (which is a big claim, IMO), this means that hallucination can be fixed if we find a way to **force LLMs to only give answers that contain information they know.** He then proposed a couple of solutions.

1. Verification: asking the LLM to explain (retrieve) the sources where it gets the answer from.
2. RL. Remember that the reward model in phase 3.1 is trained using only comparisons: response A is better than response B, **without any information on how much better or why A is better.** Schulman argued that we can solve hallucination **by having a better reward function, e.g. punishing a model more for making things up.**

Here’s a screenshot from [John Schulman’s talk](https://www.youtube.com/live/hhiLw5Q_UFg?feature=share&t=1254) in April 2023.

<img src="/assets/BERTGPTDiffusion%20Research.assets/13-schulman-fix-rl.png" alt="Fix hallucination with R" style="zoom: 33%;" />

From Schulman’s talk, I got the impression that RLHF is supposed to help with hallucination. However, the InstructGPT paper shows that RLHF actually made hallucination worse. Even though RLHF caused worse hallucination, it improved other aspects, and overall, **human labelers prefer RLHF model over SFT alone model.**

<img src="/assets/BERTGPTDiffusion%20Research.assets/10-hallucination.png" alt="RLHF makes hallucination worse" style="zoom:33%;" />

Based on the assumption that LLMs know what they know, some people try to reduce hallucination with prompts, e.g. adding `Answer as truthfully as possible, and if you're unsure of the answer, say "Sorry, I don't know"`. Making LLMs respond concisely also seems to help with hallucination – the fewer tokens LLMs have to generate, the less chance they have to make things up.

# 4.Diffusion Models

Refer to

1, [What are Diffusion Models? | Lil'Log (lilianweng.github.io)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

2, [理解扩散模型Diffusion Models（一） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/532402983)

3, [【diffusion】扩散模型详解！理论＋代码_diffusion扩散模型_AI Studio的博客-CSDN博客](https://blog.csdn.net/m0_63642362/article/details/127586200)

<img src="/assets/BERTGPTDiffusion%20Research.assets/image-20230517105141157.png" alt="image-20230517105141157" style="zoom: 33%;" />

Diffusion（扩散） models are inspired by non-equilibrium thermodynamics. **They define a Markov chain of diffusion steps to slowly add random noise to data and then learn to reverse the diffusion process to construct desired data samples from the noise**. Unlike VAE or flow models, diffusion models are learned with a fixed procedure and the latent variable has high dimensionality (same as the original data).

> Per my understanding is (to be updated after read thoroughly those papers)
>
> - each image is follow Gaussian distribution
> - by adding controlled noise $\{\beta_t \in (0, 1)\}_{t=1}^T$ from steps 1....T,  the noised image (secret) is learned 
> - by denoised those learned noise from noised image , the image is recovered 
> - by control the denoise process (which kind of noise be removed), final image can be constructed. 

Several diffusion-based generative models have been proposed with similar ideas underneath, including *diffusion probabilistic models* ([Sohl-Dickstein et al., 2015](https://arxiv.org/abs/1503.03585)), *noise-conditioned score network* (**NCSN**; [Yang & Ermon, 2019](https://arxiv.org/abs/1907.05600)), and *denoising diffusion probabilistic models* (**DDPM**; [Ho et al. 2020](https://arxiv.org/abs/2006.11239)).

<img src="/assets/BERTGPTDiffusion%20Research.assets/generative-overview.png" alt="img" style="zoom: 33%;" />

Existing generative modeling techniques can largely be grouped into two categories based on how they represent probability distributions.

1. likelihood-based models

   , which directly learn the distribution’s probability density (or mass) function via (approximate) maximum likelihood. Typical likelihood-based models include autoregressive models, normalizing flow models, energy-based models (EBMs), and variational auto-encoders (VAEs)

2. implicit generative models, where the probability distribution is implicitly represented by a model of its sampling process. The most prominent example is generative adversarial networks (GANs), where new samples from the data distribution are synthesized by transforming a random Gaussian vector with a neural network.

   <img src="/assets/BERTGPTDiffusion%20Research.assets/implicit_models.png" alt="img" style="zoom:50%;" />

GAN is an example of implicit models. It implicitly represents a distribution over all objects that can be produced by the generator network.

GAN 由一个生成器（generator）和判别器（discriminator）组成，generator 负责生成逼真数据以“骗”过 discriminator，而 discriminator 负责判断一个样本是真实的还是“造”出来的。GAN 的训练其实就是两个模型在相互学习“对抗”。

VAE 同样希望训练一个生成模型 x=g(z)，这个模型能够将采样后的概率分布映射到训练集的概率分布。生成隐变量 z，并且 z 是及含有数据信息又含有噪声，除了还原输入的样本数据以外，还可以用于生成新的数据。

Diffusion Models 的灵感来自 non-equilibrium thermodynamics （非平衡热力学）。理论首先定义扩散步骤的马尔可夫链，以缓慢地将随机噪声添加到数据中，然后学习逆向扩散过程以从噪声中构造所需的数据样本。与 VAE 或流模型不同，扩散模型是通过固定过程学习，并且隐空间 z 具有比较高的维度。

## 4.1 Forward diffusion process

### 4.1.1Define forward $q(\mathbf{x}_{1:T} \vert \mathbf{x}_0)$ and  $q(\mathbf{x}_t |\mathbf{x}_{t-1})$  to get $q(\mathbf{x}_t \vert \mathbf{x}_0) $  

Given a data point sampled from a real data distribution $\mathbf{x}_0 \sim q(\mathbf{x})$, let us define a *forward diffusion process* in which we add small amount of Gaussian noise to the sample in $T$ steps, producing a sequence of noisy samples $\mathbf{x}_1, \dots, \mathbf{x}_T$ The step sizes are controlled by a variance schedule $\{\beta_t \in (0, 1)\}_{t=1}^T$.
$$
q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I}) \quad
q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = \prod^T_{t=1} q(\mathbf{x}_t \vert \mathbf{x}_{t-1})
$$
The data sample $\mathbf{x}_0$ gradually loses its distinguishable features as the step $T$ becomes larger. Eventually when $T \to \infty$ is equivalent to an isotropic Gaussian distribution.

<img src="/assets/BERTGPTDiffusion%20Research.assets/image-20230517110949555.png" alt="image-20230517110949555" style="zoom: 50%;" />

**by using Reparameterization Trick** , Define
$$
\begin{aligned}
\alpha_t & =1-\beta_t \\
\bar{\alpha}_t & =\prod_{i=1}^t \alpha_i
\end{aligned}
$$
Then：
$$
\begin{eqnarray}
q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right) && =\mathcal{N}\left(\sqrt{1-\beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I}\right) \\
\mathbf{x}_t && =\sqrt{1-\beta_t} \mathbf{x}_{t-1}+\sqrt{\beta_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I}) &&\text{ ;where  using Reparameterization Trick }\\

\mathbf{x}_t 
&&= \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1} && \text{ ;where } \boldsymbol{\epsilon}_{t-1}, \boldsymbol{\epsilon}_{t-2}, \dots \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\
&&= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}} \bar{\boldsymbol{\epsilon}}_{t-2} && \text{ ;where } \bar{\boldsymbol{\epsilon}}_{t-2} \text{ merges two Gaussians (*).} \\
&&= \dots \\
&&= \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}  \tag{4-DDIM} \\ 
q(\mathbf{x}_t \vert \mathbf{x}_0) &&= \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I}) 

\end{eqnarray}
$$
(*) Recall that when we merge two Gaussians with different variance, $\mathcal{N}(\mathbf{0}, \sigma_1^2\mathbf{I})$ and $\mathcal{N}(\mathbf{0}, \sigma_2^2\mathbf{I})$, <font color=red>the new distribution is $\mathcal{N}(\mathbf{0}, (\sigma_1^2 + \sigma_2^2)\mathbf{I})$. Here the merged standard deviation is</font>
$$
\sqrt{(1 - \alpha_t) + \alpha_t (1-\alpha_{t-1})} = \sqrt{1 - \alpha_t\alpha_{t-1}}
$$
Usually, we can afford a larger update step when the sample gets noisier $\beta_1 < \beta_2 < \dots < \beta_T$ therefore   $\bar{\alpha}_1 > \dots > \bar{\alpha}_T$

Let $X\sim \mathcal{N}(a,b)$, $X+c \sim \mathcal{N}(a+c,b)$ and $cX \sim \mathcal{N}(ca,c^2 b)$ ， proof one of them. 
$$
\begin{align*}
F_{X+c}(x)
&=P(X+c\le x)\\
&=P(X\le x-c)\\
&=\int_{-\infty}^{x-c}\frac{1}{\sqrt{2b\pi} } \; e^{ -\frac{(t-a)^2}{2b} }\mathrm dt\\
&=\int_{-\infty}^x\frac{1}{\sqrt{2b\pi} } \; e^{ -\frac{(s-c-a)^2}{2b} }\mathrm d(s-c)\\
&=\int_{-\infty}^x\frac{1}{\sqrt{2b\pi} } \; e^{ -\frac{(s-(a+c))^2}{2b} }\mathrm ds.
\end{align*}
$$
so the new $\sigma^2 = (\sqrt{1 - \alpha_t})^2  + (\sqrt {\alpha_t(1 -  \alpha_{t-1}) }^2$ therefore $\sigma = \sqrt{1 - \alpha_t\alpha_{t-1}}$

### Reparameterization Trick 

([From Autoencoder to Beta-VAE | Lil'Log (lilianweng.github.io)](https://lilianweng.github.io/posts/2018-08-12-vae/#reparameterization-trick))

The expectation term in the loss function invokes generating samples from $\mathbf{z} \sim q_\phi(\mathbf{z}\vert\mathbf{x})$.  Sampling is a stochastic process and therefore we cannot backpropagate the gradient. To make it trainable, the reparameterization trick is introduced: It is often possible to express the random variable $z$ as a deterministic variable $\mathbf{z} = \mathcal{T}_\phi(\mathbf{x}, \boldsymbol{\epsilon})$, where $\boldsymbol{\epsilon}$ is an auxiliary independent random variable, and the transformation function $\mathcal{T}_\phi $ parameterized by $\phi$ converts $\epsilon$ to $z$.



For example, a common choice of the form of $q_\phi(\mathbf{z}\vert\mathbf{x})$ is a multivariate Gaussian with a diagonal covariance structure:
$$
\begin{aligned}
\mathbf{z} &\sim q_\phi(\mathbf{z}\vert\mathbf{x}^{(i)}) = \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}^{(i)}, \boldsymbol{\sigma}^{2(i)}\boldsymbol{I}) & \\
\mathbf{z} &= \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon} \text{, where } \boldsymbol{\epsilon} \sim \mathcal{N}(0, \boldsymbol{I}) & \scriptstyle{\text{; Reparameterization trick.}}
\end{aligned}
$$
where ⊙ refers to element-wise product



<img src="/assets/BERTGPTDiffusion%20Research.assets/reparameterization-trick.png" alt="img" style="zoom:67%;" />

Fig. 8. Illustration of how the reparameterization trick makes the $z$ sampling process trainable.(Image source: Slide 12 in Kingma’s NIPS 2015 workshop [talk](http://dpkingma.com/wordpress/wp-content/uploads/2015/12/talk_nips_workshop_2015.pdf))

The reparameterization trick works for other types of distributions too, not only Gaussian. In the multivariate Gaussian case, we make the model trainable by learning the mean and variance of the distribution, $\mu$ and $\delta$, explicitly using the reparameterization trick, while the stochasticity remains in the random variable $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \boldsymbol{I})$

<img src="/assets/BERTGPTDiffusion%20Research.assets/vae-gaussian.png" alt="img" style="zoom:33%;" />

Fig. 9. Illustration of variational autoencoder model with the multivariate Gaussian assumption

<font size =6 color=red>(TBS)</font>

### Product of Two Gaussian PDFs

[Product of Two Gaussian PDFs (stanford.edu)](https://ccrma.stanford.edu/~jos/sasp/Product_Two_Gaussian_PDFs.html)

For the special case of two [Gaussian](http://en.wikipedia.org/wiki/Normal_distribution) probability densities,


$$
\begin{aligned}
& x_1(t) \triangleq \frac{1}{\sqrt{2 \pi \sigma_1^2}} e^{-\frac{\left(t-\mu_1\right)^2}{2 \sigma_1^2}} \\
& x_2(t) \triangleq \frac{1}{\sqrt{2 \pi \sigma_2^2}} e^{-\frac{\left(t-\mu_2\right)^2}{2 \sigma_2^2}}
\end{aligned}
$$
the product density has mean and variance given by
$$
\begin{eqnarray*} \mu &=& \frac{\frac{\mu_1}{2\sigma_1^2} + \frac{\mu_2}{2\sigma_2^2}}{\frac{1}{2\sigma_1^2} + \frac{1}{2\sigma_2^2}} \; = \; \frac{\mu_1\sigma_2^2 + \mu_2\sigma_1^2}{\sigma_2^2 + \sigma_1^2}\\ \sigma^2 &=& \left. \sigma_1^2 \right\Vert \sigma_2^2 \; = \; \frac{1}{\frac{1}{\sigma_1^2} + \frac{1}{\sigma_2^2}} \;\triangleq \; \frac{\sigma_1^2\sigma_2^2}{\sigma_1^2 + \sigma_2^2}. \end{eqnarray*}
$$

### 另外一种解读(细节较为清晰以及噪声 $z_{t-1}$ ）

我们考察当给定初始样本 $x_0$ 时，扩散过程中随机变量 $X_1, X_2, \ldots$ 的分布情况。定义
$$
\alpha_t=1-\beta_t \text { 以及 } \bar{\alpha}_t=\prod_{i=1}^t a_i
$$
对于 $t$ 时刻的样本 $x_t$ ，就理论而言应该是在分布 $N\left(\sqrt{\alpha_t} x_{t-1},\left(1-\alpha_t\right) I\right)$ 中采样获得的。根 据高斯分布的性质，若 $X$ 符合高斯分布 $N(\mu, \Sigma)$ ，则 $a X+b$ 符合高斯分布 $N\left(\mu+a, b^2 \Sigma\right)$ ，**因而实际在获取 $x_t$ 时，可先从 $N(0, I)$ 中随机采样获取 $z_{t-1}$ ，然后进行变换得到符合 $N\left(\sqrt{\alpha_t} x_{t-1},\left(1-\alpha_t\right) I\right)$ 的样本，即**
$$
x_t=\sqrt{\alpha_t} x_{t-1}+\sqrt{1-\alpha_t} z_{t-1}
$$
需要强调的是，在 $x_{t-1}$ 的基础上通过添加噪声 $z_{t-1}$ 获得 $x_t$ 时，需要先缩小 $x_{t-1}$ 的数值以获得 均值，然后在此基础上再叠加高斯噪声 $z_{t-1}$ ，而并非直接在 $x_{t-1}$ 上添加噪声。
类似地，在 $t-1$ 时刻也有关系
$$
x_{t-1}=\sqrt{\alpha_{t-1}} x_{t-2}+\sqrt{1-\alpha_{t-1}} z_{t-2}
$$
其中 $z_{t-2}$ 也符合高斯分布。将两式联立有
$$
\begin{aligned}
x_t & =\sqrt{\alpha_t}\left(\sqrt{\alpha_{t-1}} x_{t-2}+\sqrt{1-\alpha_{t-1}} z_{t-2}\right)+\sqrt{1-\alpha_t} z_{t-1} \\
& =\sqrt{\alpha_t \alpha_{t-1}} x_{t-2}+\sqrt{\alpha_t-\alpha_t \alpha_{t-1}} z_{t-2}+\sqrt{1-\alpha_t} z_{t-1} \\
& =\sqrt{\alpha_t \alpha_{t-1}} x_{t-2}+\sqrt{1-\alpha_t \alpha_{t-1}} \bar{z}_{t-2}
\end{aligned}
$$
其中， $\bar{z}_{t-2} \sim N(0, I)$ 。上述推导需要依据高斯分布的性质，若 $X$ 符合高斯分布 $N\left(\mu_X, \Sigma_X\right) ， Y$ 符合高斯分布 $N\left(\mu_Y, \Sigma_Y\right)$ ，则新的随机变量 $a X+b Y$ 符合高斯分布 $N\left(a \mu_X+b \mu_Y, a^2 \Sigma_X+b^2 \Sigma_Y\right)$ 。
将上述过程不断迭代，可获得 $x_t$ 和 $x_0$ 的关系
$$
x_t=\sqrt{\bar{\alpha}_t} x_0+\sqrt{1-\bar{\alpha}_t} \bar{z}_0
$$
其中 $\bar{z}_0$ 仍为从 $N(0,1)$ 的采样噪声。此时，我们可以得到结论，当给定初始样本 $x_0$ 时，随机 变量 $X_t$ 的概率分布为
$$
q\left(X_t \mid x_0\right)=N\left(\sqrt{\bar{\alpha}_t} x_0,\left(1-\bar{\alpha}_t\right) I\right)
$$
上式表明，对于某个给定 $x_0$ ，其在多次扩散中，始终对应于一个高斯分布，如上图所示，且该高 斯分布逐渐趋近标准高斯分布。
此时需要说明一个问题: 生成模型需要实现从噪声 $x_T$ 到样本 $x_0$ 的映射，这里为什么需要构建如 此复杂的前向扩散过程? 这是因为通过上述扩散过程蕴含着重要可逆性质: 当 $\beta_t$ 足够小时，在扩 散过程的逆过程 (生成过程) 中， $q\left(X_{t-1} \mid x_t\right)$ 也将 (近似) 符合高斯分布。

### 扩散模型是如何进行概率建模

为了说明扩散模型是如何进行概率建模的，我们必须介绍扩散过程 (正向过程) 和生成过程 (逆向 过程、推断过程) 。为避免混渚，将以 $x_1, x_2, \ldots x_T$ 表示不同时间步中的样本，以 $X_1, X_2, \ldots, X_T$ 表示不同时间步对应的随机变量，以 $p\left(X_T\right)$ 表示随机变量的概率分布，以 $N\left(x_T ; \mu, \Sigma\right)$ 表示在分布 $N(\mu, \Sigma)$ 中 $X=x_T$ 时的概率样本概率值。
在扩散过程 (正向过程) 中，通过对任意的初始样本 $x_0$ 连续地添加 $T$ 次高斯噪声，可获得包含 一条样本的轨迹 $x_1, x_2, \ldots, x_T$ ，并且当 $T$ 趋于无穷时，原始样本 $x_0$ 的特征完全消失，成为 标准高斯噪声。从概率分布的角度而言，如果定义初始样本 (训练样本) 的概率分布为 $q\left(X_0\right)$ ， 则通过无限次地扩散动作，实现了从初始样本分布到标准高斯分布的映射，即 $q\left(X_T\right)=N(0, I)$
当然，扩散过程连续添加的高斯噪声并不是任意的，其具体的限定规则为
$$
q\left(X_t \mid x_{t-1}\right)=N\left(\sqrt{1-\beta_t} x_{t-1}, \beta_t I\right) \text { ，其中 } \beta_1<\beta_2<\ldots<\beta_T
$$
由上式可知，在给定 $t-1$ 时刻的样本 $x_{t-1}$ 的条件下， $t$ 时刻样本的分布为高斯分布，其均值 为 $\sqrt{1-\beta_t} x_{t-1}$ ，方差为 $\beta_t I$ 。由此式可以看出，该条件高斯分布的均值参数只与 $x_{t-1}$ 有 关，而与 $x_{t-2}, x_{t-3}, \ldots$ 无关，因而随机过程 $\left\{X_t\right\}$ 是一个马尔可夫过程。
随机变量 $X_t$ 的均值 $\sqrt{1-\beta_t} x_{t-1}$ 相比于样本 $x_{t-1}$ 将更趋于 0 ，而方差也随着 $t$ 的增加而逐渐 向 $I$ 趋近，单个样本向标准高斯分布趋近的过程如下图所示。我们也可理解为，当扩散的时间步 足够多时，随机过程 $\left\{X_t\right\}$ 将进入稳态 $N(0, I)$ 。

## 4.2 Reversing Process

### Summary (总结）

扩散模型的性质表明，从 $q\left(X_T\right)=N(0, I)$ 到 $q\left(X_0\right)$ 的逆过程是马尔可夫过程，并且有
$$
q\left(X_{t-1} \mid x_t\right)=N\left(\tilde{\mu}\left(x_t\right), \tilde{\Sigma}\left(x_t\right)\right)
$$
若 $q\left(X_{t-1} \mid x_t\right)$ 的均值 $\tilde{\mu}\left(x_t\right)$ 和方差参数 $\tilde{\Sigma}\left(x_t\right)$ 为已知量，则能在逆过程中实现样本的采样生 成。首先，从 $X_T$ 的分布 $N(0, I)$ 开始依次采样获得 $x_T$ ，然后在 $q\left(X_{T-1} \mid x_T\right)$ 中采样获得 $x_{T-1}$ ，在 $q\left(X_{T-2} \mid x_{T-1}\right)$ 中采样获得 $x_{T-2} \ldots .$. .. 迭代此过程直至获得样本 $x_0$ 。
然而，正向扩散过程中，条件高斯分布 $q\left(X_t \mid x_{t-1}\right)$ 的均值、方差具有明确的数值，而在逆向过 程中， $q\left(X_{t-1} \mid x_t\right)$ 的均值 $\tilde{\mu}\left(x_t\right)$ 和方差参数 $\tilde{\Sigma}\left(x_t\right)$ 是没有解析形式的，因而需要学习每个 $t$ 时刻的均值和方差参数。也就是说，我们需要根据训练样本集合 $\left\{x_0^{(1)}, x_0^{(2)}, \ldots, x_0^{(N)}\right\}$ 学习到 足够好的 $\tilde{\mu}\left(x_t\right)$ 和 $\tilde{\Sigma}\left(x_t\right)$ ，从而使得能在逆过程中产生足够好的样本。

通过一些数学变换得到 $q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \color{blue}{\tilde{\boldsymbol{\mu}}}(\mathbf{x}_t, \mathbf{x}_0), \color{red}{\tilde{\beta}_t} \mathbf{I})$ 其中 detail please check below  [chapter 4.2.2](# 4.2.2 Tractable $q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)$) 
$$
\begin{aligned}
\tilde{\beta}_t 
&= \color{green}{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t} \\
\tilde{\boldsymbol{\mu}}(\mathbf{x}_t, \mathbf{x}_0) &= \color{cyan}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \Big)}
\end{aligned}
$$
此时，我们可以构建生成模型 $p_\theta\left(x_{0: T}\right)$ ，即
$$
\begin{aligned}
& p_\theta\left(x_{0: T}\right)=p\left(x_T\right) \prod_{t=1}^T p_\theta\left(x_{t-1} \mid x_t\right) \\
& p_\theta\left(x_{t-1} \mid x_t\right)=N\left(x_{t-1} ; \mu_\theta\left(x_t, t\right), \Sigma_\theta\left(x_t, t\right)\right)
\end{aligned}
$$
其中， $p\left(x_T\right)=N\left(x_T ; 0, I\right)$
$$
\begin{aligned}
\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) &= \color{cyan}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \Big)} \\
\text{Thus }\mathbf{x}_{t-1} &= \mathcal{N}(\mathbf{x}_{t-1}; \frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \Big), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))
\end{aligned}
$$
<img src="/assets/BERTGPTDiffusion%20Research.assets/denoising-diffusion-probabilistic-models-forward_posterior_reverse_equations.png" alt="img" style="zoom: 50%;" />

**扩散模型将 $x_t, t$ 和 $\mu\left(x_t, t\right)$ 以及 $\Sigma\left(x_t, t\right)$ 的映射关系定义为参数 $\theta$ 可学习的神经网络形式**，即 $\mu_\theta\left(x_t, t\right)$ 和 $\Sigma_\theta\left(x_t, t\right)$ (DDPM仅对均值的映射关系进行学习，方差映射关系为先验设定 $\Sigma_\theta\left(x_t, t\right)=\sigma_t^2 I=\tilde{\beta_t} I$ 或者 $\beta_t\left(1-\bar{\alpha}_{t-1}\right) /\left(1-\bar{\alpha}_t\right) I$ ，不包含可学习参数，只考 虑均值的学习过程，若能通过数据集学习获得最优参数 $\theta^*$ ，从而实现 $p_{\theta^*}\left(X_{t-1} \mid x_t\right) \approx q\left(X_{t-1} \mid x_t\right)$ ，则能进行样本生成。

<img src="/assets/BERTGPTDiffusion%20Research.assets/denoising-diffusion-probabilistic-models-forward_and_backward_equations.png" alt="img" style="zoom:50%;" />

### 4.2.1$p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$ approximate general

![img](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/DDPM.png)

Fig. 2. The Markov chain of forward (reverse) diffusion process of generating a sample by slowly adding (removing) noise. (Image source: [Ho et al. 2020](https://arxiv.org/abs/2006.11239) with a few additional annotations)

If we can reverse the above process and sample from $q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$,  we will be able to recreate the true sample from a Gaussian noise input, $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. 
**Note that if $\beta_t$ (the noise added) is small enough**, $q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$ will also be Gaussian. Unfortunately, we cannot easily estimate $q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$ because it needs to use the entire dataset and therefore **we need to learn a model $p_\theta$ to approximate** these conditional probabilities in order to run the *reverse diffusion process*.
$$
p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod^T_{t=1} p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) \quad \text {;same conditional independent} \\
p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))
$$
<img src="/assets/BERTGPTDiffusion%20Research.assets/diffusion-example.png" alt="img" style="zoom: 25%;" />



### 4.2.2 Tractable $q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)$ 

It is noteworthy that the reverse conditional probability is tractable when conditioned on $X_0$:
$$
q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \color{blue}{\tilde{\boldsymbol{\mu}}}(\mathbf{x}_t, \mathbf{x}_0), \color{red}{\tilde{\beta}_t} \mathbf{I})
$$


$$
\begin{aligned}
q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) 
&= q(\mathbf{x}_t \vert \mathbf{x}_{t-1}, \mathbf{x}_0) \frac{ q(\mathbf{x}_{t-1} \vert \mathbf{x}_0) }{ q(\mathbf{x}_t \vert \mathbf{x}_0) }  \text{ ; we already get } q(\mathbf{x}_t \vert \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I}) \\
&\propto \exp \Big(-\frac{1}{2} \big(\frac{(\mathbf{x}_t - \sqrt{\alpha_t} \mathbf{x}_{t-1})^2}{\beta_t} + \frac{(\mathbf{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0)^2}{1-\bar{\alpha}_{t-1}} - \frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0)^2}{1-\bar{\alpha}_t} \big) \Big) \\
&= \exp \Big(-\frac{1}{2} \big(\frac{\mathbf{x}_t^2 - 2\sqrt{\alpha_t} \mathbf{x}_t \color{blue}{\mathbf{x}_{t-1}} \color{black}{+ \alpha_t} \color{red}{\mathbf{x}_{t-1}^2} }{\beta_t} + \frac{ \color{red}{\mathbf{x}_{t-1}^2} \color{black}{- 2 \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0} \color{blue}{\mathbf{x}_{t-1}} \color{black}{+ \bar{\alpha}_{t-1} \mathbf{x}_0^2}  }{1-\bar{\alpha}_{t-1}} - \frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0)^2}{1-\bar{\alpha}_t} \big) \Big) \\
&= \exp\Big( -\frac{1}{2} \big( \color{red}{(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}})} \mathbf{x}_{t-1}^2 - \color{blue}{(\frac{2\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{2\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0)} \mathbf{x}_{t-1} \color{black}{ + C(\mathbf{x}_t, \mathbf{x}_0) \big) \Big)} \\

&= \exp\Big( -\frac{1}{2} \big( \color{red}{} \frac{\mathbf{x}_{t-1}^2}{\tilde\beta} - \color{blue}{\frac{2\tilde\mu}{\tilde\beta}} \mathbf{x}_{t-1} \color{black}{ + C(\mathbf{x}_t, \mathbf{x}_0) \big) \Big)  \text{   ;  according to below replacement }} \\

&= \exp\Big( -\frac{1}{2} \big( \frac{\color{red}{\mathbf{x}_{t-1}^2} - \color{blue}{2\tilde\mu \mathbf{x}_{t-1}} +\tilde\mu^2 }{\tilde\beta }  \color{black}{ + C(\mathbf{x}_t, \mathbf{x}_0) - \tilde\mu^2\tilde\beta \big) \Big)  \text{   ;  according to below replacement...we get mu and sigma here as following }} 

\end{aligned}
$$


where $C(\mathbf{x}_t, \mathbf{x}_0)$ is some function not involving $\mathbf{x}_{t-1}$ and details are omitted. Following the standard Gaussian density function, the mean and variance can be parameterized as follows (recall that $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{i=1}^T \alpha_i$):
$$
\begin{aligned}
\tilde{\beta}_t 
&= 1/(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}) 
= 1/(\frac{\alpha_t - \bar{\alpha}_t + \beta_t}{\beta_t(1 - \bar{\alpha}_{t-1})})
= \color{green}{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t} \\
\tilde{\boldsymbol{\mu}}_t (\mathbf{x}_t, \mathbf{x}_0)
&= (\frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1} }}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0)/(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}) \\
&= (\frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1} }}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0) \color{green}{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t} \\
&= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0\\
\end{aligned}
$$
we have $ \mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}$, so we can represent $\mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t)$ and plug it into the above equation and obtain:
$$
\begin{aligned}
\tilde{\boldsymbol{\mu}}_t
&= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t) \\
&= \color{cyan}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \Big)}
\end{aligned}
$$
### 4.2.3 using $p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$ approximate $q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)$ 

in the reverse diffusion process, $p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))$. <font color=red>We would like to train $\boldsymbol{\mu}_\theta$ to predict $\tilde{\boldsymbol{\mu}}_t = \frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \Big)$. Because $\mathbf{x}_t$ is available as input at training time, we can reparametrized the Gaussian noise term instead to make it predict $\boldsymbol{\epsilon}_t$ from the input $\mathbf{x}_t$ at time step $t$: </font>
$$
\begin{aligned}
\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) &= \color{cyan}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \Big)} \\
\text{Thus }\mathbf{x}_{t-1} &= \mathcal{N}(\mathbf{x}_{t-1}; \frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \Big), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))
\end{aligned}
$$
### 4.2.4 The loss function $- \log p_\theta(\mathbf{x}_0)$ (NLL-Maximum Likelihood) with VAE Approximate ELBO =$L_{VLB}$


$$
\begin{aligned}
- \log p_\theta(\mathbf{x}_0) 
&\leq - \log p_\theta(\mathbf{x}_0) + D_\text{KL}(q(\mathbf{x}_{1:T}\vert\mathbf{x}_0) \| p_\theta(\mathbf{x}_{1:T}\vert\mathbf{x}_0) ) \\
		 &\textcolor{blue}{\text{; Data = x0 and we have ELBO}  =  \log p(D) - D_{KL}( q(z)|p(z|D) ) =  \int_{}^{}{q(z)\log\left( \frac{p(z,D)}{q(z)} \right)dz} = - KL(q(z)|p(z,x))}\\
&= -\log p_\theta(\mathbf{x}_0) + \mathbb{E}_{\mathbf{x}_{1:T}\sim q(\mathbf{x}_{1:T} \vert \mathbf{x}_0)} \Big[ \log\frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T}) / p_\theta(\mathbf{x}_0)} \Big] \\ 
         &\textcolor{blue}{\text{;we have } p_\theta(\mathbf{x}_{0:T}) =p(\mathbf{x_0})p(\mathbf{x_{1:T}}|\mathbf{x_0}) }\\
         &\textcolor{blue}{\text{;we have } D_{K L}(q \mid p)=\int q(x) \log \left(\frac{q(x)}{p(x)}\right) d x = E(\log (\frac{q(x)}{p(x)}) }\\
     
&= -\log p_\theta(\mathbf{x}_0) + \mathbb{E}_q \Big[ \log\frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} + \log p_\theta(\mathbf{x}_0) \Big] \\
&= \mathbb{E}_q \Big[ \log \frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \\
\text{Let }L_\text{VLB} 
&= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Big[ \log \frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \geq - \mathbb{E}_{q(\mathbf{x}_0)} \log p_\theta(\mathbf{x}_0) \\
		&\textcolor{blue}{\text{; here we have } \mathbb{E}_{q(\mathbf{x}_0)} \log p_\theta(\mathbf{x}_0) \leq \mathbb{E}_{q(\mathbf{x}_0)} \mathbb{E}_q \Big[ \log \frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] }\\
\end{aligned}
$$

It is also straightforward to get the same result using Jensen’s inequality. Say we want to minimize the cross entropy as the learning objective,
$$
\begin{aligned}
L_\text{CE}
&= - \mathbb{E}_{q(\mathbf{x}_0)} \log p_\theta(\mathbf{x}_0) \\
&= - \mathbb{E}_{q(\mathbf{x}_0)} \log \Big( \int p_\theta(\mathbf{x}_{0:T}) d\mathbf{x}_{1:T} \Big) \\
&= - \mathbb{E}_{q(\mathbf{x}_0)} \log \Big( \int q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0})} d\mathbf{x}_{1:T} \Big) \\
&= - \mathbb{E}_{q(\mathbf{x}_0)} \log \Big( \mathbb{E}_{q(\mathbf{x}_{1:T} \vert \mathbf{x}_0)} \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0})} \Big) \\
&\leq - \mathbb{E}_{q(\mathbf{x}_{0:T})} \log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0})} \\
& \textcolor{blue}{\text{; here we have } \quad-\log (E[x]) \leq E[-\log x]}\\
&= \mathbb{E}_{q(\mathbf{x}_{0:T})}\Big[\log \frac{q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0})}{p_\theta(\mathbf{x}_{0:T})} \Big] = L_\text{VLB}
\end{aligned}
$$
To convert each term in the equation to be analytically computable, the objective can be further rewritten to be a combination of several KL-divergence and entropy terms (See the detailed step-by-step process in Appendix B in [Sohl-Dickstein et al., 2015](https://arxiv.org/abs/1503.03585)):
$$
\begin{aligned}
L_\text{VLB} 
&= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Big[ \log\frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \\
&= \mathbb{E}_q \Big[ \log\frac{\prod_{t=1}^T q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{ p_\theta(\mathbf{x}_T) \prod_{t=1}^T p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t) } \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=1}^T \log \frac{q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \log\frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \Big( \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)}\cdot \frac{q(\mathbf{x}_t \vert \mathbf{x}_0)}{q(\mathbf{x}_{t-1}\vert\mathbf{x}_0)} \Big) + \log \frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \sum_{t=2}^T \log \frac{q(\mathbf{x}_t \vert \mathbf{x}_0)}{q(\mathbf{x}_{t-1} \vert \mathbf{x}_0)} + \log\frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{q(\mathbf{x}_1 \vert \mathbf{x}_0)} + \log \frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big]\\
&= \mathbb{E}_q \Big[ \log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_T)} + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} - \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1) \Big] \\
&= \mathbb{E}_q [\underbrace{D_\text{KL}(q(\mathbf{x}_T \vert \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_T))}_{L_T} + \sum_{t=2}^T \underbrace{D_\text{KL}(q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t))}_{L_{t-1}} \underbrace{- \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)}_{L_0} ]
\end{aligned}
$$

其他人对采用$X_0$的解释

<img src="/assets/BERTGPTDiffusion%20Research.assets/image-20230605215923369.png" alt="image-20230605215923369" style="zoom:50%;" />

Let’s label each component in the variational lower bound loss separately:
$$
\begin{aligned}
L_\text{VLB} &= L_T + L_{T-1} + \dots + L_0 \\
\text{where } L_T &= D_\text{KL}(q(\mathbf{x}_T \vert \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_T)) \\
L_t &= D_\text{KL}(q(\mathbf{x}_t \vert \mathbf{x}_{t+1}, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_t \vert\mathbf{x}_{t+1})) \text{ for }1 \leq t \leq T-1 \\
L_0 &= - \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)
\end{aligned}
$$
- Every KL term in $L_\text{VLB}$ (except for $L_0$) compares two Gaussian distributions and therefore they can be computed in [closed form](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence#Multivariate_normal_distributions). 

- **$L_T$ is constant and can be ignored during training because $q$ has no learnable parameters and $\mathbf{x}_T$ is a Gaussian noise.** 

- **[Ho et al. 2020](https://arxiv.org/abs/2006.11239) models $L_0$ using a separate discrete decoder derived from $\mathcal{N}(\mathbf{x}_0; \boldsymbol{\mu}_\theta(\mathbf{x}_1, 1), \boldsymbol{\Sigma}_\theta(\mathbf{x}_1, 1))$.**

### 4.2.5 Parameterization of $L_t$ for Training Loss

已经得到 $q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \color{blue}{\tilde{\boldsymbol{\mu}}}(\mathbf{x}_t, \mathbf{x}_0), \color{red}{\tilde{\beta}_t} \mathbf{I})$ 
其中
$$
\begin{aligned}
\tilde{\beta}_t 
&= \color{green}{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t} \\
\tilde{\boldsymbol{\mu}}(\mathbf{x}_t, \mathbf{x}_0) &= \color{cyan}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \Big)}
\end{aligned}
$$
The loss term $L_t$ is parameterized to minimize the difference from $\tilde{\boldsymbol{\mu}}$ :
$$
\begin{aligned}
L_t 
&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{1}{2 \| \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t) \|^2_2} \| \color{blue}{\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0)} - \color{green}{\boldsymbol{\mu}_\theta(\mathbf{x}_t, t)} \|^2 \Big] \\
&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{1}{2  \|\boldsymbol{\Sigma}_\theta \|^2_2} \| \color{blue}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \Big)} - \color{green}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t) \Big)} \|^2 \Big] \\
&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{ (1 - \alpha_t)^2 }{2 \alpha_t (1 - \bar{\alpha}_t) \| \boldsymbol{\Sigma}_\theta \|^2_2} \|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2 \Big] \\
&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{ (1 - \alpha_t)^2 }{2 \alpha_t (1 - \bar{\alpha}_t) \| \boldsymbol{\Sigma}_\theta \|^2_2} \|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t, t)\|^2 \Big] 
\end{aligned}
$$

in another form 
$$
\begin{aligned}
& L_t=\frac{1}{2 \sigma_t^2}\left\|\tilde{\mu}_t\left(X_t, X_0\right)-\mu_\theta\left(X_t, t\right)\right\|^2 \\
= & \frac{1}{2 \sigma_t^2} \| \frac{1}{\sqrt{d_t}}\left(X_t-\frac{\beta_t}{\sqrt{1-\sigma_t}} \varepsilon\right)-\frac{1}{\sqrt{\sigma_t}}\left(X_t-\frac{\beta_t}{\sqrt{1-\alpha_t}} \varepsilon_\theta\left(X_t, t\right) \|^2\right. \\
= & \frac{\beta_t^2}{2 \sigma_t^2 \alpha_t\left(1-\alpha_t\right)}\left\|\varepsilon-\varepsilon_\theta\left(X_t, t\right)\right\|^2
\end{aligned}
$$

### 4.2.6 models $L_0$ 

在逆扩散过程中，马尔科夫过程表示为由连续条件高斯分布下的累积变换组成。有了总体的优化策略，还要看每个像素的计算方式，在逆扩散过程结束时，我们希望得到一张生成好的图像，因此需要设计一种方法，**使得图像上每个像素值都满足离散的对数似然**。 为了达到这个目的，将逆扩散过程中的最后从 x1 到 x0 的转换设置为独立的离散计算方式。 即在最后一个转换过程在给定  x1 下得到图像 x0 满足对数似然，假设像素与像素之间是相互独立的：
$$
p_\theta\left(x_0 \mid x_1\right)=\prod_{i=1}^D p_\theta\left(x_0^i \mid x_1^i\right)
$$
 D 是输入数据的维数，上标 i 表示图像中的一个坐标位置。现在的目标是确定给定像素的值可能性有多大，也就是想要知道对应时间步 t=1 下噪声图像 x 中相应像素值的分布
$$
\mathcal{N}\left(x ; \mu_\theta^i\left(x_1, 1\right), \sigma_1^2\right)
$$
其中 t = 1 的像素分布来自多元高斯分布，其对角协方差矩阵允许我们将分布拆分为单变量高斯的乘积：
$$
\mathcal{N}\left(x ; \mu_\theta\left(x_1, 1\right), \sigma_1^2 \mathbb{I}\right)=\prod_{i=1}^D \mathcal{N}\left(x ; \mu_\theta^i\left(x_1, 1\right), \sigma_1^2\right)
$$
现在假设图像已经从0-255的数值之间，经过归一化在[-1,1]的范围内。在 t=0 时给定每个像素的像素值，最后一个时间步 t=1 的转换概率分布 $p_θ(x_0∣x_1)$ 的值就是每个像素值的乘积。简而言之，这个过程由等式简洁 (18) 地表示：
$$
\begin{aligned}
& p_\theta\left(x_0 \mid x_1\right)=\prod_{i=1}^D p_\theta\left(x_0^i \mid x_1^i\right) \\
& =\prod_{i=1}^D \int_{\delta_{-}\left(x_0^i\right)}^{\delta_{+}\left(x_i^i\right)} \mathcal{N}\left(x ; \mu_\theta^i\left(x_1, 1\right), \sigma_1^2\right) d x
\end{aligned}
$$
其中约束有:
$$
\delta_{-}(x)= \begin{cases}-\infty & x=-1 \\ x-\frac{1}{255} & x>-1\end{cases}
$$
和:
$$
\delta_{+}(x)= \begin{cases}\infty & x=1 \\ x+\frac{1}{255} & x<1\end{cases}
$$

### 4.2.7 Loss Simplification 

Empirically, [Ho et al. (2020)](https://arxiv.org/abs/2006.11239) found that training the diffusion model works better with a simplified objective that ignores the weighting term:
$$
\begin{aligned}
L_t^\text{simple}
&= \mathbb{E}_{t \sim [1, T], \mathbf{x}_0, \boldsymbol{\epsilon}_t} \Big[\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2 \Big] \\
&= \mathbb{E}_{t \sim [1, T], \mathbf{x}_0, \boldsymbol{\epsilon}_t} \Big[\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t, t)\|^2 \Big]
\end{aligned}
$$
The final simple objective is:
$$
L_\text{simple} = L_t^\text{simple} + C
$$
where $C$ is a constant not depending on $\theta$.

<img src="/assets/BERTGPTDiffusion%20Research.assets/DDPM-algo.png" alt="img" style="zoom:50%;" />

Fig. 4. The training and sampling algorithms in DDPM(Image source: [Ho et al. 2020](https://arxiv.org/abs/2006.11239))



## 4.3 Reserving Approximate (overview)

VAE:
$$
\begin{aligned}
L_{\mathrm{VAE}}(\theta, \phi) & =-\log p_\theta(\mathbf{x})+D_{\mathrm{KL}}\left(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p_\theta(\mathbf{z} \mid \mathbf{x})\right) \\
& =-\mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})} \log p_\theta(\mathbf{x} \mid \mathbf{z})+D_{\mathrm{KL}}\left(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p_\theta(\mathbf{z})\right) \\
\theta^*, \phi^* & =\arg \underset{\theta, \phi}{\min } L_{\mathrm{VAE}} \\
-L_{\mathrm{VAE}} & =\log p_\theta(\mathbf{x})-D_{\mathrm{KL}}\left(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p_\theta(\mathbf{z} \mid \mathbf{x})\right) \leq \log p_\theta(\mathbf{x})
\end{aligned}
$$
GAN:
$$
\begin{gathered}
\min _G \max _D L(D, G)=\mathbb{E}_{x \sim p_r(x)}[\log D(x)]+\mathbb{E}_{z \sim p_z(z)}[\log (1-D(G(z)))] \\
=\mathbb{E}_{x \sim p_r(x)}[\log D(x)]+\mathbb{E}_{x \sim p_g(x)}[\log (1-D(x)] \\
L\left(G, D^*\right)=2 D_{J S}\left(p_r \| p_g\right)-2 \log 2
\end{gathered}
$$
Turns out that for small enough forward steps, i.e.
$$
\left\{\beta_t \in(0,1)\right\}_{t=1}^T
$$
the reverse process step $q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)$ can be estimate is a Gaussian distribution too (take a course of stochastic differential equations if you want learn more)! 
Therefore, we can parametrize the learned reverse process as
$$
p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)=\mathcal{N}\left(\mathbf{x}_{t-1} ; \boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right), \mathbf{\Sigma}_\theta\left(\mathbf{x}_t, t\right)\right)
$$
such that
$$
p_\theta\left(\mathbf{x}_{0: T}\right)=p\left(\mathbf{x}_T\right) \prod_{t=1}^T p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)
$$

### 4.3.1 A Preliminary objective 

The VAE (ELBO) loss is a bound on the true log likelihood (also called the variational lower bound)
$$
-L_{\mathrm{VAE}}=\log p_\theta(\mathbf{x})-D_{\mathrm{KL}}\left(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p_\theta(\mathbf{z} \mid \mathbf{x})\right) \leq \log p_\theta(\mathbf{x})
$$
Apply the same trick to diffusion:
$$
-\log p_\theta\left(\mathbf{x}_0\right) \leq \mathbb{E}_{q\left(\mathbf{x}_{0: T}\right)}\left[-\log \frac{p_\theta\left(\mathbf{x}_{0: T}\right)}{q\left(\mathbf{x}_{1: T} \mid \mathbf{x}_0\right)}\right]=L_{V L B}
$$
Expanding out,
$$
\begin{aligned}
L_{\mathrm{VLB}} & =L_T+L_{T-1}+\cdots+L_0 \\
\text { where } L_T & =D_{\mathrm{KL}}\left(q\left(\mathbf{x}_T \mid \mathbf{x}_0\right) \| p_\theta\left(\mathbf{x}_T\right)\right) \\
L_t & =D_{\mathrm{KL}}\left(q\left(\mathbf{x}_t \mid \mathbf{x}_{t+1}, \mathbf{x}_0\right) \| p_\theta\left(\mathbf{x}_t \mid \mathbf{x}_{t+1}\right)\right) \text { for } 1 \leq t \leq T-1 \\
L_0 & =-\log p_\theta\left(\mathbf{x}_0 \mid \mathbf{x}_1\right)
\end{aligned}
$$

$$
\begin{aligned}
L_{\mathrm{VLB}} & =L_T+L_{T-1}+\cdots+L_0 \\
\text { where } L_T & =D_{\mathrm{KL}}\left(q\left(\mathbf{x}_T \mid \mathbf{x}_0\right) \| p_\theta\left(\mathbf{x}_T\right)\right) \\
L_t & =D_{\mathrm{KL}}\left(q\left(\mathbf{x}_t \mid \mathbf{x}_{t+1}, \mathbf{x}_0\right) \| p_\theta\left(\mathbf{x}_t \mid \mathbf{x}_{t+1}\right)\right) \text { for } 1 \leq t \leq T-1 \\
L_0 & =-\log p_\theta\left(\mathbf{x}_0 \mid \mathbf{x}_1\right)
\end{aligned}
$$

### **4.3.2 A simplified object**

The reverse step conditioned on x_0 is a Gaussian:
$$
\begin{aligned}
q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right) & =\mathcal{N}\left(\mathbf{x}_{t-1} ; \tilde{\boldsymbol{\mu}}_t\left(\mathbf{x}_t, \mathbf{x}_0\right), \tilde{\beta}_t \mathbf{I}\right) \\
\text { where } \quad \tilde{\boldsymbol{\mu}}_t\left(\mathbf{x}_t, \mathbf{x}_0\right) & :=\frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} \mathbf{x}_0+\frac{\sqrt{\alpha_t}\left(1-\bar{\alpha}_{t-1}\right)}{1-\bar{\alpha}_t} \mathbf{x}_t \quad \text { and } \quad \tilde{\beta}_t:=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t
\end{aligned}
$$
After doing some algebra, **each loss term can be approximated by**
$$
\begin{aligned}
L_{t-1} & =\mathbb{E}_{\mathbf{x}_0, \epsilon}\left[\frac{1}{2\left\|\mathbf{\Sigma}_\theta\right\|_2^2}\left\|\tilde{\mu}\left(\mathbf{x}_t, \mathbf{x}_0\right)-\mu_\theta\left(\mathbf{x}_{t,}, t\right)\right\|_2^2\right] \\
& =\mathbb{E}_{\mathbf{x}_0, \epsilon}\left[\frac{1}{2\left\|\mathbf{\Sigma}_\theta\right\|_2^2}\left\|\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon\right)-\mu_\theta\left(\mathbf{x}_{t,}, t\right)\right\|_2^2\right]
\end{aligned}
$$
**Instead of predicting the mu, Ho et al. say that we should predict epsilon instead!**
$$
\tilde{\boldsymbol{\mu}}_t\left(\mathbf{x}_t, \mathbf{x}_0\right)=\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}\right) \Longrightarrow \boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right)=\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t, t\right)\right)
$$
Thus, our loss becomes
$$
\begin{aligned}
L_{t-1} & =\mathbb{E}_{\mathbf{x}_0, \epsilon}\left[\frac{1}{2\left\|\boldsymbol{\Sigma}_\theta\right\|_2^2}\left\|\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon\right)-\frac{1}{\sqrt{\alpha}_t}\left(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta\left(\mathbf{x}_t, t\right)\right)\right\|_2^2\right] \\
& =\mathbb{E}_{\mathbf{x}_0, \epsilon}\left[\frac{\beta_t^2}{2 \alpha_t\left(1-\bar{\alpha}_t\right)\left\|\boldsymbol{\Sigma}_\theta\right\|_2^2}\left\|\epsilon-\epsilon_\theta\left(\mathbf{x}_t, t\right)\right\|_2^2\right] \\
& =\mathbb{E}_{\mathbf{x}_0, \epsilon}\left[\frac{\beta_t^2}{2 \alpha_t\left(1-\bar{\alpha}_t\right)\left\|\boldsymbol{\Sigma}_\theta\right\|_2^2}\left\|\epsilon-\epsilon_\theta\left(\sqrt{\bar{\alpha}_t} \mathbf{x}_0+\sqrt{1-\bar{\alpha}_t} \epsilon, t\right)\right\|_2^2\right]
\end{aligned}
$$
The authors of DDPM say that it's fine to drop all that baggage in the front and instead just use
$$
L_{t-1}=\mathbb{E}_{\mathbf{x}_0, \epsilon}\left[\left\|\epsilon-\epsilon_\theta\left(\sqrt{\bar{\alpha}_t} \mathbf{x}_0+\sqrt{1-\bar{\alpha}_t} \epsilon, t\right)\right\|_2^2\right]
$$
Note that this is not a variational lower bound on the log-likelihood anymore: in fact, you can view it as a reweighted version of ELBO that emphasizes reconstruction quality!

## 4.4 Training

$$
\begin{aligned}
& \hline \text { Algorithm } 1 \text { Training } \\
& \hline \text { 1: repeat } \\
& \text { 2: } \quad \mathbf{x}_0 \sim q\left(\mathbf{x}_0\right) \\
& \text { 3: } \quad t \sim \text { Uniform }(\{1, \ldots, T\}) \\
& \text { 4: } \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\
& \text { 5: } \quad \text { Take gradient descent step on } \\
& \quad \nabla_\theta\left\|\boldsymbol{\epsilon}-\boldsymbol{\epsilon}_\theta\left(\sqrt{\bar{\alpha}_t} \mathbf{x}_0+\sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}, t\right)\right\|^2 \\
& \text { 6: until converged }
\end{aligned}
$$

**下面这段对训练过程的数据来源使用解释得比较清楚：**

由上式可以看出，训练目标函数 $L_t$ 项表达的意图是：在给定 $x_t$ 时，若要最终获得某个 $x_0$ ，**条 件高斯分布的均值应为 $\tilde{\mu}_t\left(x_t, x_0\right)$ (可理解为学习的标签)**，因此，模型在给定 $x_t$ 时，应当能 尽可能输出 $\mu_\theta\left(x_t, t\right)$ ，从而能提升 $x_0$ 的似然函数值 $\log p_\theta\left(x_0\right)$ 。
根据展开式 $\tilde{\mu}_t\left(x_t, x_0\right)=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} z_t\right)$ 可知，在给定 $x_t$ 时，若模型能够预测 $z_t$ (也就是$\epsilon _t$)，就 能正确计算出均值。自然地， $\mu_\theta\left(x_t, t\right)$ 的计算方式与 $\tilde{\mu}_t\left(x_t, x_0\right)$ 相同而仅将 $z_t$ 参数化为神经 网络 $z_\theta\left(x_t, t\right)$ ，即
$$
\mu_\theta\left(x_t, t\right)=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} z_\theta\left(x_t, t\right)\right)
$$
将 $\tilde{\mu}_t\left(x_t, x_0\right)$ 和 $\mu_\theta\left(x_t, t\right)$ 的表达式替换， $L_t$ 项进一步写为
$$
L_t=\mathbb{E}_{x_0, z} \frac{\beta_t^2\left\|z_t-z_\theta\left(x_t, t\right)\right\|_2^2}{2 \sigma_t^2 \alpha_t\left(1-\bar{\alpha}_t\right)}
$$
<font color=red>**由于 $z_t \sim N(0, I)$ ，实际训练时， $z_t$ 可直接采样获得**</font>，**而 $z_\theta\left(x_t, t\right)$ 的输入 $x_t$ 可在给定最终 生成样本 $x_0$ 的条件下给出**， $x_t=\sqrt{\bar{\alpha}_t} x_0+\sqrt{1-\bar{\alpha}_t} z_t$ ，**即每个 $x_t$ 均对应于一个 $z_t$** 。由 上式也可以看出，神经网络 $\theta$ 本质上也是在学习“降噪”幅度，即为了最后获得给定的 $x_0$ ，神经 网络应该学习逐层降噪，即应该在 $t$ 时间步对 $x_t$ 进行 $z_\theta\left(x_t, t\right)$ 幅度的橾声降低，
**当完成训练后，在推断过程中生成样本时，当生成 $x_t$ 时，首先计算 $p\left(X_{t-1} \mid x_t\right)$ 对应高斯分布的 均值 $z_\theta\left(x_t, t\right)$ ，然后在 $N\left(z_\theta\left(x_t, t\right), \sigma_t^2\right)$ 中随机采样即可得到 $x_{t-1}$ 。**

> ### Algorithm1：Training
>
> 从数据中抽取一个样本，
>
> 从1-T中随机选取一个时间t
>
> 将 $x_0$ 和t传给GaussionDiffusion，GaussionDiffusion采样一个随机噪声，加到 $x_0$ ，形成$x_t$  ，然后将$x_t$  和t放入Unet，Unet根据t生成正弦位置编码和 $x_t$  结合，Unet预测加的这个噪声，并返回噪声，GaussionDiffusion计算该噪声和随机噪声的损失
>
> 将神经网络Unet预测的噪声与之前GaussionDiffusion采样的随机噪声求L2损失，计算梯度，更新权重。
>
> 重复以上步骤，直到网络Unet训练完成。
>
> ![img](/assets/BERTGPTDiffusion%20Research.assets/v2-33081e7d50e65e1ed4e5d1f91e67728b_r.jpg)
>
> ### Algorithm2：Sampling
>
> - - 从标准正态分布采样出 $x_T$
> - - 从 $T, T-1, \ldots, 2,1$ 依次重复以下步骤:
> - - 从标准正态分布采样 $z$ ，为重参数化做准备
> - - 根据模型求出 $\epsilon_\theta$ ，结合 $x_t$ 和采样得到z利用重参数化技巧，得到 $x_{t-1}$
> - - 循环结束后返回 $x_0$
>   采样步骤中每个模块的交互如下图:
>
> ![img](/assets/BERTGPTDiffusion%20Research.assets/v2-3673b2795344783503286c32f05fc7b6_r.jpg)

- #### Model Architecture Used In DDPMs

![img](/assets/BERTGPTDiffusion%20Research.assets/denoising-diffusion-probabilistic-models_UNet_model_architecture.png)

**The architecture comprises 5 components:**

1. Encoder blocks
2. Bottleneck blocks
3. Decoder blocks
4. [Self attention modules](https://learnopencv.com/attention-mechanism-in-transformer-neural-networks/)
5. Sinusoidal time embeddings

## 4.5 Parameterization of reverse process variance $\boldsymbol{\Sigma}_\theta$

Ho et al. (2020) chose to fix $\beta_t$ as constants instead of making them learnable and set $\boldsymbol{\Sigma}_\theta\left(\mathbf{x}_t, t\right)=\sigma_t^2 \mathbf{I}$, where $\sigma_t$ is not learned but set to $\beta_t$ or $\tilde{\beta}_t=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha} t} \cdot \beta_t$. Because they found that learning a diagonal variance $\boldsymbol{\Sigma}_{\boldsymbol{\theta}}$ leads to unstable training and poorer sample quality.

[Nichol & Dhariwal (2021)](https://arxiv.org/abs/2102.09672) proposed to learn $\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)$ as an interpolation between $\beta_t$ and $\tilde{\beta}_t$ by model predicting a mixing vector V :
$$
\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t) = \exp(\mathbf{v} \log \beta_t + (1-\mathbf{v}) \log \tilde{\beta}_t)
$$

## 4.6 Sampling (inference)

$$
\begin{aligned}
& \text { Algorithm } 2 \text { Sampling } \\
& \text { 1: } \mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\
& \text { 2: for } t=T, \ldots, 1 \text { do } \\
& \text { 3: } \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \text { if } t>1, \text { else } \mathbf{z}=\mathbf{0} \\
& \text { 4: } \quad \mathbf{x}_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t, t\right)\right)+\sigma_t \mathbf{z} \\

&\text{; where we have } \mathbf{x}_{t-1} = \mathcal{N}(\mathbf{x}_{t-1}; \frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \Big), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)) \\

& \text { 5: end for } \\
& \text { 6: return } \mathbf{x}_0
\end{aligned}
$$

<img src="/assets/BERTGPTDiffusion%20Research.assets/image-20230517124126420.png" alt="image-20230517124126420" style="zoom: 50%;" />

## 4.7 Speed up Diffusion Model Sampling (DDIM)

also refer to [扩散模型之DDIM - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/565698027)

### Recall DDPM

It is very slow to generate a sample from DDPM by following the Markov chain of the reverse diffusion process, as T can be up to one or a few thousand steps. One data point from [Song et al. 2020](https://arxiv.org/abs/2010.02502): “For example, it takes around 20 hours to sample 50k images of size 32 × 32 from a DDPM, but less than a minute to do so from a GAN on an Nvidia 2080 Ti GPU.”

One simple way is to run a strided sampling schedule ([Nichol & Dhariwal, 2021](https://arxiv.org/abs/2102.09672)) by taking the sampling update every ⌈T/S⌉ steps to reduce the process from T to S steps. The new sampling schedule for generation is $\{\tau_1, \dots, \tau_S\}$ where $\tau_1 < \tau_2 < \dots <\tau_S \in [1, T]$ and S<T.

With DDIM, **it is possible to train the diffusion model up to any arbitrary number of forward steps but only sample from a subset of steps in the generative process**.

扩散过程的一个重要特性是可以直接用 $\mathbf{x}_0$ 来对任意的 $\mathbf{x}_t$ 进行采样:
$$
q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I}) \quad \\
q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = \prod^T_{t=1} q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) \\

q\left(\mathbf{x}_t \mid \mathbf{x}_0\right)=\mathcal{N}\left(\mathbf{x}_t ; \sqrt{\alpha_t} \mathbf{x}_0,\left(1-\alpha_t\right) \mathbf{I}\right)
$$
注意，在DDIM的论文中， $\alpha_t$ 其实是DDPM论文中的 $\bar{\alpha}_t$ ，那么DDPM论文中的前向过程 $\beta_t$ 就为:
$$
\beta_t=\left(1-\frac{\alpha_t}{\alpha_{t-1}}\right)
$$
而DDPM的反向过程也定义为一个马尔卡夫链:
$$
p_\theta\left(\mathbf{x}_{0: T}\right)=p\left(\mathbf{x}_T\right) \prod_{t=1}^T p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right) \quad p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)=\mathcal{N}\left(\mathbf{x}_{t-1} ; \boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right), \mathbf{\Sigma}_\theta\left(\mathbf{x}_t, t\right)\right)
$$
这里用神经网络 $p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)$ 来拟合真实的分布 $q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)$ 。DDPM的前向过程和反向过程如下 所示:

![img](/assets/BERTGPTDiffusion%20Research.assets/v2-071e3c9962f3f12239a8b005940e4616_720w.webp)

**DDPM的目标是要拟合出一个$q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)$**

1. 我们近一步发现后验分布 $q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right)$ 是一个可获取的高斯分布:

$$
q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right)=\mathcal{N}\left(\mathbf{x}_{t-1} ; \tilde{\boldsymbol{\mu}}\left(\mathbf{x}_t, \mathbf{x}_0\right), \tilde{\beta}_t \mathbf{I}\right)
$$
​		其中这个高斯分布的方差是定值，而均值是一个依赖 $\mathrm{x}_0$ 和 $\mathbf{x}_t$ 的组合函数:
$$
\tilde{\boldsymbol{\mu}}_t\left(\mathbf{x}_t, \mathbf{x}_0\right)=\frac{\sqrt{\alpha_t}\left(1-\alpha_{t-1}\right)}{\sqrt{\alpha_{t-1}}\left(1-\alpha_t\right)} \mathbf{x}_t+\frac{\sqrt{\alpha_{t-1}} \beta_t}{1-\alpha_t} \mathbf{x}_0
$$
​	2. 然后我们基于变分法得到如下的优化目标:
$$
\begin{aligned}
& L=\mathbb{E}_{q\left(\mathbf{x}_{1: T} \mid \mathbf{x}_0\right)}\left[\log \frac{q\left(\mathbf{x}_{1: T} \mid \mathbf{x}_0\right)}{p_\theta\left(\mathbf{x}_{0: T}\right)}\right] \\
& =\underbrace{D_{\mathrm{KL}}\left(q\left(\mathbf{x}_T \mid \mathbf{x}_0\right) \| p_\theta\left(\mathbf{x}_T\right)\right)}_{L_T}+\sum_{t=2}^T \underbrace{\mathbb{E}_{q\left(\mathbf{x}_t \mid \mathbf{x}_0\right)}\left[D_{\mathrm{KL}}\left(q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right) \| p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)\right)\right]}_{L_{t-1}} \\
& -\underbrace{\mathbb{E}_{q\left(\mathbf{x}_1 \mid \mathbf{x}_0\right)} \log p_\theta\left(\mathbf{x}_0 \mid \mathbf{x}_1\right)}_{L_0} \\
&
\end{aligned}
$$
3. 根据两个高斯分布的KL公式，我们近一步得到:

$$
L_{t-1}=\mathbb{E}_{q\left(\mathbf{x}_t \mid \mathbf{x}_0\right)}\left[\frac{1}{2 \sigma_t^2}\left\|\tilde{\boldsymbol{\mu}}_t\left(\mathbf{x}_t, \mathbf{x}_0\right)-\boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right)\right\|^2\right]
$$
​		根据扩散过程的特性，我们通过重参数化可以近一步简化上述目标:
$$
L_{t-1}=\mathbb{E}_{\mathbf{x}_0, \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})}\left[\frac{\beta_t^2}{2 \sigma_t^2 \alpha_t\left(1-\bar{\alpha}_t\right)}\left\|\epsilon-\epsilon_\theta\left(\sqrt{\bar{\alpha}_t} \mathbf{x}_0+\sqrt{1-\bar{\alpha}_t} \epsilon, t\right)\right\|^2\right]
$$
​		如果去掉系数，那么就能得到更简化的优化目标:
$$
L_{t-1}^{\text {simple }}=\mathbb{E}_{\mathbf{x}_0, \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})}\left[\left\|\epsilon-\epsilon_\theta\left(\sqrt{\bar{\alpha}_t} \mathbf{x}_0+\sqrt{1-\bar{\alpha}_t} \epsilon, t\right)\right\|^2\right]
$$
仔细分析DDPM的优化目标会发现，DDPM其实仅仅依赖边缘分布 $q\left(\mathbf{x}_t \mid \mathbf{x}_0\right)$ ，而并不是直接作用 在联合分布 $q\left(\mathbf{x}_{1: T} \mid \mathbf{x}_0\right)$ 。这带来的一个启示是: **DDPM这个隐变量模型可以有很多推理分布来选 择，只要推理分布满足边缘分布条件（扩散过程的特性）即可**，而且这些推理过程并不一定要是马 尔卡夫链。但值得注意的一个点是，我们要得到DDPM的优化目标，还需要知道分布 $q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right)$ ，之前我们在根据贝叶斯公式推导这个分布时是知道分布 $q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right)$ （forward process know it ) 的，而且依 赖了前向过程的马尔卡夫链特性.

$$q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) 
= q(\mathbf{x}_t \vert \mathbf{x}_{t-1}, \mathbf{x}_0) \frac{ q(\mathbf{x}_{t-1} \vert \mathbf{x}_0) }{ q(\mathbf{x}_t \vert \mathbf{x}_0) }$$ ， zphilip48： **DDIM的目的就是去除依赖(公式推导上的) $q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right)$** 

### 解除对前向过程的依赖

现在我们在只给定 $p\left(x_t \mid \boldsymbol{x}_0\right) 、 p\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_0\right)$ 的情况下，通过待定系数法求解了 $p\left(x_{t-1} \mid x_t, x_0\right)$ 的一簇 解，它带有一个自由参数 $\sigma_t$ 。用 “拆楼-建 楼” 类比来说，就是我们知道楼会被拆成什么样 $【 p\left(\boldsymbol{x}_t \mid \boldsymbol{x}_0\right) 、 p\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_0\right) 】$ ，但是不知道每一步怎么 拆【 $p\left(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1}\right)$ 】 ，然后希望能够从中学会每一步怎么建$【 \left[p\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t\right) 】\right.$ 。当然，如果我们想看看每 步怎么拆的话，也可以反过来用贝叶斯公式
$$
p\left(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1}, \boldsymbol{x}_0\right)=\frac{p\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{x}_0\right) p\left(\boldsymbol{x}_t \mid \boldsymbol{x}_0\right)}{p\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_0\right)}
$$
**如果要解除对前向过程的依赖**，那么我们就需要直接定义这个分 布 $q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right)$ 。 基于上述分析，DDIM论文中将推理分布定义为：
$$
q_\sigma\left(\mathbf{x}_{1: T} \mid \mathbf{x}_0\right)=q_\sigma\left(\mathbf{x}_T \mid \mathbf{x}_0\right) \prod_{t=2}^T q_\sigma\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right)
$$
理解定义推导过程（Bayesian rule)： 
$$
\begin{aligned}
q_\sigma\left(\mathbf{x}_{1: T} \mid \mathbf{x}_0\right) &= \prod_{t=1}^T q_\sigma\left(\mathbf{x}_{t} \mid \mathbf{x}_{t-1}\right) \\
& = q_\sigma(\mathbf{x}_1|\mathbf{x}_0)\prod_{t=2}^T \frac{q_\sigma\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}, \mathbf{x}_0\right) q_\sigma\left(\boldsymbol{x}_t \mid \boldsymbol{x}_0\right)}{q_\sigma\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_0\right) }\\
& = q_\sigma\left(\mathbf{x}_T \mid \mathbf{x}_0\right) \prod_{t=2}^T q_\sigma\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right)
\end{aligned}
$$
对比DDPM前向定义
$$
q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = \prod^T_{t=1} q(\mathbf{x}_t \vert \mathbf{x}_{t-1})
$$
DDPM后向定义
$$
p_\theta\left(\mathbf{x}_{0: T}\right)=p\left(\mathbf{x}_T\right) \prod_{t=1}^T p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)
$$
**可以理解为，DDIM 把前向过程定义为得到了$X_{T}$以后的一个后向过程...**

DDIM论文中将推理分布定义为：
$$
q_\sigma\left(\mathbf{x}_{1: T} \mid \mathbf{x}_0\right)=q_\sigma\left(\mathbf{x}_T \mid \mathbf{x}_0\right) \prod_{t=2}^T q_\sigma\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right)
$$
这里要同时满足 $q_\sigma\left(\mathbf{x}_T \mid \mathbf{x}_0\right)=\mathcal{N}\left(\sqrt{\alpha_T} \mathbf{x}_0,\left(1-\alpha_T\right) \mathbf{I}\right)$ 以及对于所有的 $t \geq 2$ 有:
$$
q_\sigma\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right)=\mathcal{N}\left(\mathbf{x}_{t-1} ; \sqrt{\alpha_{t-1}} \mathbf{x}_0+\sqrt{1-\alpha_{t-1}-\sigma_t^2} \frac{\mathbf{x}_t-\sqrt{\alpha_t} \mathbf{x}_0}{\sqrt{1-\alpha_t}}, \sigma_t^2 \mathbf{I}\right)
$$
这里的方差 $\sigma_t^2$ 是一个实数，不同的设置就是不一样的分布，所以 $q_\sigma\left(\mathbf{x}_{1: T} \mid \mathbf{x}_0\right)$ 其实是一系列的推 理分布。可以看到这里分布 $q_\sigma\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right)$ 的均值也定义为一个依赖 $\mathbf{x}_0$ 和 $\mathbf{x}_t$ 的组合函数，之所 以定义为这样的形式，是因为根据 $q_\sigma\left(\mathbf{x}_T \mid \mathbf{x}_0\right)$ ，我们可以通过数学归纳法证明，对于所有的 $t$ 均满 足:
$$
q_\sigma\left(\mathbf{x}_t \mid \mathbf{x}_0\right)=\mathcal{N}\left(\mathbf{x}_t ; \sqrt{\alpha_t} \mathbf{x}_0,\left(1-\alpha_t\right) \mathbf{I}\right)
$$
可以看到这里定义的推理 分布 $q_\sigma\left(\mathbf{x}_{1: T} \mid \mathbf{x}_0\right)$ **并没有直接定义前向过程，**但这里满足了我们前面要讨论的两个条件: 边缘分布 $q_\sigma\left(\mathbf{x}_t \mid \mathbf{x}_0\right)=\mathcal{N}\left(\mathbf{x}_t ; \sqrt{\alpha_t} \mathbf{x}_0,\left(1-\alpha_t\right) \mathbf{I}\right)$ ，同时已知后验分布 $q_\sigma\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right)$ 。

同样地，我 们可以按照和DDPM的一样的方式去推导优化目标，最终也会得到同样的 $L^{\text {simple }}$ (虽然VLB的系 数不同，论文3.2部分也证明了这个结论）。论文也给出了一个前向过程是非马尔可夫链的示例， 如下图所示，这里前向过程是 $q_\sigma\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}, \mathbf{x}_0\right)$ ，由于生成 $\mathbf{x}_t$ 不仅依赖 $\mathbf{x}_{t-1}$ ，而且依赖 $\mathbf{x}_0$ ，所以 是一个非马尔可夫链:

![image-20230608101549213](/assets/BERTGPTDiffusion%20Research.assets/image-20230608101549213.png)

###  $q_\sigma\left(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0\right)$  推导-1

贝叶斯定理   
$$
p\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{x}_0\right)=\frac{p\left(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1}\right) p\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_0\right)}{p\left(\boldsymbol{x}_t \mid \boldsymbol{x}_0\right)} \tag {2}
$$
设有给定 $p\left(x_t \mid x_{t-1}\right)$ 怎么能得到 $p\left(x_{t-1} \mid x_t, x_0\right)$ ? 这其实是思维过于定式了，理论上在没有给定 $p\left(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1}\right)$ 的情况下， $p\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{x}_0\right)$ 的解空间更大，某种意义上来说是更加容易推导，此时它**只需 要满足边际分布条件**:
$$
\int p\left(x_{t-1} \mid x_t, x_0\right) p\left(x_t \mid x_0\right) d x_t=p\left(x_{t-1} \mid x_0\right) \tag {3}
$$
一般性假设:
$$
p\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{x}_0\right)=\mathcal{N}\left(\boldsymbol{x}_{t-1} ; \kappa_t \boldsymbol{x}_t+\lambda_t \boldsymbol{x}_0, \sigma_t^2 \boldsymbol{I}\right) \tag{4}
$$
其中 $\kappa_t, \lambda_t, \sigma_t$ 都是待定系数，而为了不重新训练模型，我们不改变 $p\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_0\right)$ 和 $p\left(\boldsymbol{x}_t \mid \boldsymbol{x}_0\right)$ ，于是我们 叮以列出

| question                                                     | Distribution                                                 | Sampling                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| $p\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_0\right)$ forward defined | $\mathcal{N}\left(\boldsymbol{x}_{t-1} ; \bar{\alpha}_{t-1} \boldsymbol{x}_0, \bar{\beta}_{t-1}^2 \boldsymbol{I}\right)$ | $\boldsymbol{x}_{t-1}=\bar{\alpha}_{t-1} \boldsymbol{x}_0+\bar{\beta}_{t-1} \varepsilon$ |
| $p\left(\boldsymbol{x}_t \mid \boldsymbol{x}_0\right)$ forward defined | $\mathcal{N}\left(\boldsymbol{x}_t ; \bar{\alpha}_t \boldsymbol{x}_0, \bar{\beta}_t^2 \boldsymbol{I}\right)$ | $\boldsymbol{x}_t=\bar{\alpha}_t \boldsymbol{x}_0+\bar{\beta}_t \varepsilon_1$ |
| $p\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{x}_0\right)$ reversing | $\mathcal{N}\left(\boldsymbol{x}_{t-1} ; \kappa_t \boldsymbol{x}_t+\lambda_t \boldsymbol{x}_0, \sigma_t^2 \boldsymbol{I}\right)$ | $\boldsymbol{x}_{t-1}=\kappa_t \boldsymbol{x}_t+\lambda_t \boldsymbol{x}_0+\sigma_t \varepsilon_2$ |
| $\int p\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{x}_0\right) p\left(\boldsymbol{x}_t \mid \boldsymbol{x}_0\right) d \boldsymbol{x}_t$ reversing |                                                              | $$\begin{aligned} \boldsymbol{x}_{t-1} &=\kappa_t \boldsymbol{x}_t+\lambda_t \boldsymbol{x}_0+\sigma_t \varepsilon_2 \\ &=\kappa_t\left(\bar{\alpha}_t \boldsymbol{x}_0+\bar{\beta}_t \varepsilon_1\right)+\lambda_t \boldsymbol{x}_0+\sigma_t \varepsilon_2 \\ &=\left(\kappa_t \bar{\alpha}_t+\lambda_t\right) \boldsymbol{x}_0+\left(\kappa_t \bar{\beta}_t \varepsilon_1+\sigma_t \varepsilon_2\right) \end{aligned}$$ |

其中 $\varepsilon, \varepsilon_1, \varepsilon_2 \sim \mathcal{N}(\mathbf{0}, \boldsymbol{I})$ ，并且由正态分布的叠加性我们知道 $\kappa_t \bar{\beta}_t \varepsilon_1+\sigma_t \varepsilon_2 \sim \sqrt{\kappa_t^2 \bar{\beta}_t^2+\sigma_t^2} \varepsilon_{\text {。 }}$ 对 比 $x_{t-1}$ 的两个采样形式，我们发现要想 $(3)$ 成立，只需要满足两个方程:
$$
\bar{\alpha}_{t-1}=\kappa_t \bar{\alpha}_t+\lambda_t, \quad \bar{\beta}_{t-1}=\sqrt{\kappa_t^2 \bar{\beta}_t^2+\sigma_t^2} \tag {5}
$$
可以看到有三个末知数，但只有两个方程，这就是为什么说没有给定 $p\left(x_t \mid x_{t-1}\right)$ 时解空间反而更大 了。将 $\sigma_t$ 视为可变参数，可以解出
$$
\kappa_t=\frac{\sqrt{\bar{\beta}_{t-1}^2-\sigma_t^2}}{\bar{\beta}_t}, \quad \lambda_t=\bar{\alpha}_{t-1}-\frac{\bar{\alpha}_t \sqrt{\bar{\beta}_{t-1}^2-\sigma_t^2}}{\bar{\beta}_t} \tag{6}
$$
或者写成
$$
p\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{x}_0\right)=\mathcal{N}\left(\boldsymbol{x}_{t-1} ; \frac{\sqrt{\bar{\beta}_{t-1}^2-\sigma_t^2}}{\bar{\beta}_t} \boldsymbol{x}_t+\left(\bar{\alpha}_{t-1}-\frac{\bar{\alpha}_t \sqrt{\bar{\beta}_{t-1}^2-\sigma_t^2}}{\bar{\beta}_t}\right) \boldsymbol{x}_0, \sigma_t^2 \boldsymbol{I}\right) \tag {7}
$$
方便起见，我们约定 $\bar{\alpha}_0=1, \bar{\beta}_0=0$ 。特别地，这个结果并不需要限定 $\bar{\alpha}_t^2+\bar{\beta}_t^2=1$ ，不过为了简化 参数设置，同时也为了跟以往的结果对齐，这里还是约定 $\bar{\alpha}_t^2+\bar{\beta}_t^2=1$ 。

我们最终想要 $p\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t\right)$ 而不是 $p\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{x}_0\right)$ ，所以 我们希望用
$$
\overline{\boldsymbol{\mu}}\left(\boldsymbol{x}_t\right)=\frac{1}{\bar{\alpha}_t}\left(\boldsymbol{x}_t-\bar{\beta}_t \boldsymbol{\epsilon} _\boldsymbol{\theta}\left(\boldsymbol{x}_t, t\right)\right)
$$
来估计 $\boldsymbol{x}_0$ （**可以看后面关于$f_\theta^{(t)}\left(\boldsymbol{x}_t\right)$的论文解释**），由于没有改动 $p\left(\boldsymbol{x}_t \mid \boldsymbol{x}_0\right)$ ，所以训练所用的目标函数依然是 $\left\|\varepsilon-\boldsymbol{\epsilon}_{\boldsymbol{\theta}}\left(\bar{\alpha}_t \boldsymbol{x}_0+\bar{\beta}_t \varepsilon, t\right)\right\|^2$ (除 去权重系数），也就是说训练过程没有改变，我们可以用回DDPM训练好的模型。而用 $\overline{\boldsymbol{\mu}}(\boldsymbol{x} t)$ 替换掉 式(7)中的 $\boldsymbol{x}_0$ 后，得到
$$
\begin{aligned}
p\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t\right) & \approx p\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{x}_0=\overline{\boldsymbol{\mu}}\left(\boldsymbol{x}_t\right)\right) \\
& =\mathcal{N}\left(\boldsymbol{x}_{t-1} ; \frac{1}{\alpha_t}\left(\boldsymbol{x}_t-\left(\bar{\beta}_t-\alpha_t \sqrt{\bar{\beta}_{t-1}^2-\sigma_t^2}\right) \boldsymbol{\epsilon}_{\boldsymbol{\theta}}\left(\boldsymbol{x}_t, t\right)\right), \sigma_t^2 \boldsymbol{I}\right)
\end{aligned}
$$
这就求出了生成过程所需要的 $p\left(x_{t-1} \mid x_t\right)$ ，其中 $\alpha_t=\frac{\bar{\alpha} t}{\bar{\alpha} t-1}$ 。它的特点是训练过程没有变化（也就是 说最终保存下来的模型没有变化)，**但生成过程却有一个可变动的参数 $\sigma_t$ ，就是这个参数给DDPM带 来了新鲜的结果。**

### 合并推导参数

以上的参数 $\alpha = \sqrt{\bar{\alpha}_{ddpm}} = \sqrt{\alpha_{ddim}}$ , ${\beta}^2 = {1- \alpha^2} = {1- \bar\alpha_{ddpm}}$and $\bar{\beta} = \sqrt{1-\bar{\alpha}_{ddpm}} = \sqrt{1-\alpha_{ddim}}$ , replace the parameters,下面是ddpm or ddim 的参数: 
$$
\begin {aligned}
\tilde{\boldsymbol{\mu}}\left(\mathbf{x}_t, \mathbf{x}_0\right) &=\frac{\sqrt{\beta_{t-1}-\sigma_t^2}}{\sqrt{1-\alpha}_t} \boldsymbol{x}_t+\left(\sqrt{\alpha}_{t-1}-\frac{\sqrt{\alpha}_t \sqrt{\beta_{t-1}-\sigma_t^2}}{\sqrt{1-\alpha}_t}\right) \boldsymbol{x}_0 \\
& = \frac{\sqrt{(1-\alpha)_{t-1}-\sigma_t^2}}{\sqrt{1-\alpha}_t} \boldsymbol{x}_t+\left(\sqrt{\alpha}_{t-1}-\frac{\sqrt{\alpha}_t \sqrt{(1-\alpha)_{t-1}-\sigma_t^2}}{\sqrt{1-\alpha}_t}\right) \boldsymbol{x}_0 \\ 
& = \sqrt{\alpha_{t-1}} \mathbf{x}_0+\sqrt{1-\alpha_{t-1}-\sigma_t^2} \frac{\mathbf{x}_t-\sqrt{\alpha_t} \mathbf{x}_0}{\sqrt{1-\alpha_t}}
\end {aligned}
$$

$$
\begin{aligned}
\tilde{\boldsymbol{\mu}}\left(\mathbf{x}_t \right) &=\frac{1}{\alpha_t}\left(\boldsymbol{x}_t-\left(\bar{\beta}_t-\alpha_t \sqrt{\bar{\beta}_{t-1}^2-\sigma_t^2}\right) \boldsymbol{\epsilon}_{\boldsymbol{\theta}}\left(\boldsymbol{x}_t, t\right)\right) \\
& \text { 其中 } \alpha_t=\frac{\bar{\alpha} t}{\bar{\alpha}_{t-1}} 下面参数是ddim \\ 
& = \sqrt{\alpha_{t-1}} \underbrace{\left(\frac{\boldsymbol{x}_t-\sqrt{1-\alpha_t} \epsilon_\theta^{(t)}\left(\boldsymbol{x}_t\right)}{\sqrt{\alpha_t}}\right)}_{\text {"predicted } \boldsymbol{x}_0 \text { " }}+\underbrace{\sqrt{1-\alpha_{t-1}-\sigma_t^2} \cdot \epsilon_\theta^{(t)}\left(\boldsymbol{x}_t\right)}_{\text {“direction pointing to } \boldsymbol{x}_t \text { " }}
\end{aligned}
$$

和以下的公式比对....

### $q_\sigma(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)$   推导-2 

rewrite **<font color=red>$q_\sigma(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)$ to be parameterized by a desired standard deviation $\sigma_t$</font>**according to the [nice property](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#nice):
$$
\begin{eqnarray}
\mathbf{x}_{t-1} 
&&= \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 +  \sqrt{1 - \bar{\alpha}_{t-1}}\boldsymbol{\epsilon}_{t-1} \\
&&= \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \boldsymbol{\epsilon}_t + \sigma_t\boldsymbol{\epsilon} \\

&& \color{red} \text {; don't clear what this come??  } \sqrt{1 - \bar{\alpha}_{t-1}}\boldsymbol{\epsilon}_{t-1} = \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \boldsymbol{\epsilon}_t + \sigma_t\boldsymbol{\epsilon} \\
&& \color{red} \text {; we have  }   \mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon} ==>  \epsilon_t = \frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0}{\sqrt{1 - \bar{\alpha}_t}}\\

&&= \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0}{\sqrt{1 - \bar{\alpha}_t}} + \sigma_t\boldsymbol{\epsilon} \\
q_\sigma(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)
&&= \mathcal{N}(\mathbf{x}_{t-1}; \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0}{\sqrt{1 - \bar{\alpha}_t}}, \sigma_t^2 \mathbf{I}) \tag{7-DDIM} 
\end{eqnarray}
$$
During generation, we only sample a subset of $S$ diffusion steps $\{\tau_1, \dots, \tau_S\}$ and the inference process becomes:
$$
q_{\sigma, \tau}(\mathbf{x}_{\tau_{i-1}} \vert \mathbf{x}_{\tau_t}, \mathbf{x}_0)
= \mathcal{N}(\mathbf{x}_{\tau_{i-1}}; \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \frac{\mathbf{x}_{\tau_i} - \sqrt{\bar{\alpha}_t}\mathbf{x}_0}{\sqrt{1 - \bar{\alpha}_t}}, \sigma_t^2 \mathbf{I}) \tag{7-DDIM}
$$
While all the models are trained with $T=1000$ diffusion steps in the experiments, they observed that DDIM $\eta=0$ can produce the best quality samples when S is small, while DDPM $\eta=1$ performs much worse on small S. DDPM does perform better when we can afford to run the full reverse Markov diffusion steps $S=T=1000$. With DDIM, it is possible to train the diffusion model up to any arbitrary number of forward steps but only sample from a subset of steps in the generative process.

**From forward process we know following**
$$
\boldsymbol{x}_t=\sqrt{\alpha_t} \boldsymbol{x}_0+\sqrt{1-\alpha_t} \epsilon, \quad \text { where } \epsilon \sim \mathcal{N}(\mathbf{0}, \boldsymbol{I}) \text {. } \tag{4}
$$
For some $\boldsymbol{x}_0 \sim q\left(\boldsymbol{x}_0\right)$ and $\epsilon_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{I}), \boldsymbol{x}_t$ can be obtained using Eq. (4). The model $\epsilon_\theta^{(t)}\left(\boldsymbol{x}_t\right)$ then attempts to predict $\epsilon_t$ from $\boldsymbol{x}_t$, **without knowledge of $\boldsymbol{x}_0$**. By rewriting Eq. (4), one can then predict the denoised observation, **which is a prediction of $\boldsymbol{x}_0$ given $\boldsymbol{x}_t$** :
$$
f_\theta^{(t)}\left(\boldsymbol{x}_t\right):=\left(\boldsymbol{x}_t-\sqrt{1-\alpha_t} \cdot \epsilon_\theta^{(t)}\left(\boldsymbol{x}_t\right)\right) / \sqrt{\alpha_t} \tag {9-DDIM}
$$
We can then define the generative process with a fixed prior $p_\theta\left(\boldsymbol{x}_T\right)=\mathcal{N}(\mathbf{0}, \boldsymbol{I})$ and
$$
p_\theta^{(t)}\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t\right)= \begin{cases}\mathcal{N}\left(f_\theta^{(1)}\left(\boldsymbol{x}_1\right), \sigma_1^2 \boldsymbol{I}\right) & \text { if } t=1 \\
\text{;zphlip48, where from x1->x0 special handling} \\
\text{;check L0 } \\ 
q_\sigma\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, f_\theta^{(t)}\left(\boldsymbol{x}_t\right)\right) & \text { otherwise, }\end{cases}
\tag {10-DDIM}
$$
where $q_\sigma\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, f_\theta^{(t)}\left(\boldsymbol{x}_t\right)\right)$ is defined as in Eq. (7) **with $\boldsymbol{x}_0$ replaced by $f_\theta^{(t)}\left(\boldsymbol{x}_t\right)$**. We add some Gaussian noise (with covariance $\sigma_1^2 \boldsymbol{I}$ ) for the case of $t=1$ to ensure that the generative process is supported everywhere.

From $p_θ(x_{1:T} )$ in Eq. (10), one can generate a sample $x_{t−1}$ from a sample $x_t$ via:
$$
\text {DENOISING DIFFUSION IMPLICIT MODELS }\\
\boldsymbol{x}_{t-1}=\sqrt{\alpha_{t-1}} \underbrace{\left(\frac{\boldsymbol{x}_t-\sqrt{1-\alpha_t} \epsilon_\theta^{(t)}\left(\boldsymbol{x}_t\right)}{\sqrt{\alpha_t}}\right)}_{\text {"predicted } \boldsymbol{x}_0 \text { " }}+\underbrace{\sqrt{1-\alpha_{t-1}-\sigma_t^2} \cdot \epsilon_\theta^{(t)}\left(\boldsymbol{x}_t\right)}_{\text {“direction pointing to } \boldsymbol{x}_t \text { " }}+\underbrace{\sigma_t \epsilon_t}_{\text {random noise }} \tag {12-DDIM}
$$
where $\epsilon_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{I})$ is standard Gaussian noise independent of $\boldsymbol{x}_t$, and we define $\alpha_0:=1$. Different choices of $\sigma$ values results in different generative processes, all while using the same model $\epsilon_\theta$, so re-training the model is unnecessary. 

其中, $\sigma$ 可以参考 DDIM 论文的公式 (16) :
$$
\sigma_t=\eta \sqrt{\left(1-\bar{\alpha}_{t-1}\right) /\left(1-\bar{\alpha}_t\right)} \sqrt{1-\bar{\alpha}_t / \bar{\alpha}_{t-1}} \tag{16-DDIM}
$$
如果 $\eta=0$ ，那么生成过程就是确定的，这种情况下为 DDIM。
论文中指出, 当 $\eta=1$, 该 forward process 变成了马尔科大链, 该生成过程等价于 DDPM 的生成过 程。也就是说当 $\eta=1$ 时, 公式 (12) 等于 DDPM 的采样公式, 即公式 (7) :
$$
\begin{aligned}
& \hat{x}_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta\left(x_t, t\right)\right)+\sigma_t z \\
& \quad \text { where } z=N(0, I)
\end{aligned}
$$
将 (16) 式带入到 (1) 式中得到 DDPM 分布公式（本文章标记依照 DDPM 论文, 因此有 $\bar{\alpha}_t=\Pi_T \alpha_t$ ) :
$$
\sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}=\frac{1-\bar{\alpha}_{t-1}}{\sqrt{1-\bar{\alpha}_t}} \sqrt{\alpha_t}
$$
上式的推导过程:
$$
\begin{aligned}
\frac{\sqrt{1-\bar{\alpha}_t}}{\sqrt{1-\bar{\alpha}_t}} \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2} & =\frac{\sqrt{\left[\left(1-\bar{\alpha}_{t-1}-\left(\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\right)\left(1-\alpha_t\right)\right]\left(1-\bar{\alpha}_t\right)\right.}}{\sqrt{1-\bar{\alpha}_t}} \\
& =\frac{\sqrt{\left(1-\bar{\alpha}_{t-1}\right)\left(1-\left(\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\right)\left(1-\alpha_t\right)\right)\left(1-\bar{\alpha}_{t-1}\right)}}{\sqrt{1-\bar{\alpha}_t}} \\
& =\frac{\sqrt{\left(1-\bar{\alpha}_{t-1}\right)\left(1-\bar{\alpha}_t-1+\frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}\right)}}{\sqrt{1-\bar{\alpha}_t}} \\
& =\frac{\sqrt{\left(1-\bar{\alpha}_{t-1}\right)\left(1-\bar{\alpha}_{t-1}\right) \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}}}{\sqrt{1-\bar{\alpha}_t}} \\
& =\frac{1-\bar{\alpha}_{t-1}}{\sqrt{1-\bar{\alpha}_t} \sqrt{\alpha_t}}
\end{aligned}
$$
因此
$$
\begin{aligned}
& x_{t-1}=\sqrt{\bar{\alpha}_{t-1}} \underbrace{\left(\frac{x_t-\sqrt{1-\bar{\alpha}_t} \epsilon_\theta^{(t)}\left(x_t\right)}{\sqrt{\bar{\alpha}_t}}\right)}_{\text {" predicted } x_0 "}+\underbrace{\sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2} \cdot \epsilon_\theta^{(t)}\left(x_t\right)}_{\text {"direction pointing to } \boldsymbol{x}_t \text { " }}+\underbrace{\sigma_t \epsilon_t}_{\text {random noise }} \\
& =\sqrt{\frac{\bar{\alpha}_{t-1}}{\bar{\alpha}_t}} x_t-\sqrt{\frac{\bar{\alpha}_{t-1}}{\bar{\alpha}_t}} \sqrt{1-\bar{\alpha}_t} \epsilon_\theta^{(t)}+\frac{1-\bar{\alpha}_{t-1}}{\sqrt{1-\bar{\alpha}_t}} \sqrt{\alpha_t} \epsilon_\theta^{(t)}+\sigma_t \epsilon_t \\
& =\frac{1}{\sqrt{\alpha}_t} x_t-\frac{1}{\sqrt{\alpha_t} \sqrt{1-\bar{\alpha}_t}}\left(1-\bar{\alpha}_t+\left(1-\bar{\alpha}_{t-1}\right) \alpha_t\right) \epsilon_\theta^{(t)}+\sigma_t \epsilon_t \\
& =\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta^{(t)}\right)+\sigma_t \epsilon_t \\
& =\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta^{(t)}\right)+\sigma_t \epsilon_t \\
&
\end{aligned}
$$
**comparing to DDPM** 
$$
\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \color{cyan}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \Big)}
$$


因此，根据推导，*η*=1 时候的 Forward Processes 等价于 DDPM，我们将在 notebook 后半部分，通过代码的方式验证当 *η*=1 DDIM 的结果与 DDPM 基本相同。

### After defined the DDIM model, the loss function also defined for $q_{\sigma}$* 

We optimize $\theta$ via the following variational inference objective (which is a functional over $\epsilon_\theta$ ):
$$
\begin{aligned}
& J_\sigma\left(\epsilon_\theta\right):=\mathbb{E}_{\boldsymbol{x}_{0: T} \sim q_\sigma\left(\boldsymbol{x}_{0: T}\right)}\left[\log q_\sigma\left(\boldsymbol{x}_{1: T} \mid \boldsymbol{x}_0\right)-\log p_\theta\left(\boldsymbol{x}_{0: T}\right)\right] \\
= & \mathbb{E}_{\boldsymbol{x}_{0: T} \sim q_\sigma\left(\boldsymbol{x}_{0: T}\right)}\left[\log q_\sigma\left(\boldsymbol{x}_T \mid \boldsymbol{x}_0\right)+\sum_{t=2}^T \log q_\sigma\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{x}_0\right)-\sum_{t=1}^T \log p_\theta^{(t)}\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t\right)-\log p_\theta\left(\boldsymbol{x}_T\right)\right]
\end{aligned}
$$
where we factorize $q_\sigma\left(\boldsymbol{x}_{1: T} \mid \boldsymbol{x}_0\right)$ according to Eq. (6) and $p_\theta\left(\boldsymbol{x}_{0: T}\right)$ according to Eq. (1).
From the definition of $J_\sigma$, it would appear that a different model has to be trained for every choice of $\sigma$, since it corresponds to a different variational objective (and a different generative process). However, $J_\sigma$ is equivalent to $L_\gamma$ for certain weights $\gamma$, as we show below.
Theorem 1. For all $\sigma>\mathbf{0}$, there exists $\gamma \in \mathbb{R}_{>0}^T$ and $C \in \mathbb{R}$, such that $J_\sigma=L_\gamma+C$.

### Comparing to DDPM

Recall that in **(DDPM reversing)** $q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I})$ so $\tilde{\beta}_t = \sigma_t^2 = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t$

Let $\sigma_t^2 = \eta \cdot \tilde{\beta}_t$ such that we can adjust $\eta \in \mathbb{R}^+$ as a hyperparameter to control the sampling stochasticity. **The special case of $\eta=0$ makes the sampling process *deterministic***. Such a model is named the *denoising diffusion implicit model* (**DDIM**; [Song et al., 2020](https://arxiv.org/abs/2010.02502)). DDIM has the same marginal noise distribution but deterministically maps noise back to the original data samples.
$$
\tilde{\beta}_t=\sigma_t^2=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \cdot \beta_t \\
\sigma_t=\eta \cdot \sqrt{\tilde{\beta}_t}=\eta \cdot \sqrt{\left(1-\alpha_{t-1}\right) /\left(1-\alpha_t\right)} \sqrt{\left(1-\alpha_t / \alpha_{t-1}\right)}
$$
When $\sigma_t=\sqrt{\left(1-\alpha_{t-1}\right) /\left(1-\alpha_t\right)} \sqrt{1-\alpha_t / \alpha_{t-1}}$ for all $t$, the forward process becomes Markovian, and the generative process becomes a DDPM.
$$
p\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t\right) \approx p\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{x}_0=\overline{\boldsymbol{\mu}}\left(\boldsymbol{x}_t\right)\right)=\mathcal{N}\left(\boldsymbol{x}_{t-1} ; \frac{1}{\alpha_t}\left(\boldsymbol{x}_t-\frac{\beta_t^2}{\bar{\beta}_t} \boldsymbol{\epsilon}_{\boldsymbol{\theta}}\left(\boldsymbol{x}_t, t\right)\right), \frac{\bar{\beta}_{t-1}^2 \beta_t^2}{\bar{\beta}_t^2} \boldsymbol{I}\right) \\
\text { where }
\sigma_t=\frac{\bar{\beta}_{t-1} \beta_t}{\bar{\beta}_t} \\
\beta_t=\sqrt{1-\alpha_t^2} \\
\alpha_t=\frac{\bar{\alpha} t}{\bar{\alpha}_{t-1}} \\
$$
这就是DDPM 。特别是，DDIM论文中还对 $\sigma t=\eta \frac{\bar{\beta}_{t-1} \beta_t}{\bar{\beta}_t}$ 做了对比实验，其中 $\eta \in[0,1]$
如果取 $\sigma_t=\beta_t$ ，这也是前两篇文章所指出的 $\sigma_t$ 的两个选择之一，在此选择下式(10)末能 做进一步的化简，但DDIM的实验结果显示此选择在DDPM的标准参数设置下表现还是很好的。
最特殊的一个例子是取 $\sigma_t=0$ ，此时从 $x_t$ 到 $x_{t-1}$ 是一个确定性变换
$$
\boldsymbol{x}_{t-1}=\frac{1}{\alpha_t}\left(\boldsymbol{x}_t-\left(\bar{\beta}_t-\alpha_t \bar{\beta}_{t-1}\right) \boldsymbol{\epsilon}_{\boldsymbol{\theta}}\left(\boldsymbol{x}_t, t\right)\right)
$$
**这也是DDIM论文中特别关心的一个例子，准确来说，原论文的DDIM就是特指 $\sigma_t=0$ 的情形，其中 “T”的含义就是“Implicit”，意思这是一个隐式的概率模型，因为跟其他选择所不同的是，此时从给定 的 $\boldsymbol{x}_T=\boldsymbol{z}$ 出发，得到的生成结果 $\boldsymbol{x}_0$ 是不带随机性的。后面我们将会看到，这在理论上和实用上都带 来了一些好处。**

### ACCELERATED SAMPLING PROCESSES:  $q\left(\mathbf{x}_{\tau_i} \mid \mathbf{x}_0\right)$  and $q_{\sigma, \tau}(\mathbf{x}_{\tau_{i-1}} \vert \mathbf{x}_{\tau_t}, \mathbf{x}_0)$    推导

DDIM并没有明确前向过程，这意味着我们可以**定义一个更短的步数的前向过程**. 具体地，这里我们从原始的序列 $[1, \ldots, T]$ 采样一个长度为 $S$ 的子序列 $\left[\tau_1, \ldots, \tau_S\right]$ ，**我们将 $\mathbf{x}_{\tau_1}, \ldots, \mathbf{x}_{\tau_S}$ 的前向过程定义为一个马尔卡夫链**，并且它们满足: $q\left(\mathbf{x}_{\tau_i} \mid \mathbf{x}_0\right)=\mathcal{N}\left(\mathbf{x}_t ; \sqrt{\alpha_{\tau_i}} \mathbf{x}_0,\left(1-\alpha_{\tau_i}\right) \mathbf{I}\right)$ 。下图展示了一个具体的示例:

During generation, we only sample a subset of S diffusion steps $\{\tau_1, \dots, \tau_S\}$ and the inference process becomes:
$$
q_{\sigma, \tau}(\mathbf{x}_{\tau_{i-1}} \vert \mathbf{x}_{\tau_t}, \mathbf{x}_0)
= \mathcal{N}(\mathbf{x}_{\tau_{i-1}}; \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \frac{\mathbf{x}_{\tau_i} - \sqrt{\bar{\alpha}_t}\mathbf{x}_0}{\sqrt{1 - \bar{\alpha}_t}}, \sigma_t^2 \mathbf{I})
$$

![image-20230531135247420](/assets/BERTGPTDiffusion%20Research.assets/image-20230531135247420.png)

那么生成过程也可以用这个子序列的反向马尔卡夫链来替代，由于 $S$ 可以设置比原来的步数 $L$ 要 小，那么就可以加速生成过程。这里的生成过程变成:
$$
\mathbf{x}_{\tau_{i-1}}=\sqrt{\alpha_{\tau_{i-1}}}\left(\frac{\mathbf{x}_{\tau_i}-\sqrt{1-\alpha_{\tau_i}} \epsilon_\theta\left(\mathbf{x}_{\tau_i}, \tau_i\right)}{\sqrt{\alpha_{\tau_i}}}\right)+\sqrt{1-\alpha_{\tau_{i-1}}-\sigma_{\tau_i}^2} \cdot \epsilon_\theta\left(\mathbf{x}_{\tau_i}, \tau_i\right)+\sigma_{\tau_i} \epsilon
$$
其实上述的加速，我们是**将前向过程**按如下方式进行了分解: （why??)
$$
q_{\sigma, \tau}\left(\mathbf{x}_{1: T} \mid \mathbf{x}_0\right)=q_{\sigma, \tau}\left(\mathbf{x}_T \mid \mathbf{x}_0\right) \prod_{i=1}^S q_\sigma\left(\mathbf{x}_{\tau_{i-1}} \mid \mathbf{x}_{\tau_i}, \mathbf{x}_0\right) \prod_{t \in \bar{\tau}} q_{\sigma, \tau}\left(\mathbf{x}_t \mid \mathbf{x}_0\right)
$$
where $\tau$ is a sub-sequence of $[1, \ldots, T]$ of length $S$ with $\tau_S=T$, and let $\bar{\tau}:=\{1, \ldots, T\} \backslash \tau$ be its complement. Intuitively, the graphical model of $\left\{\boldsymbol{x}_{\tau_i}\right\}_{i=1}^S$ and $\boldsymbol{x}_0$ form a chain, whereas the graphical model of $\left\{\boldsymbol{x}_t\right\}_{t \in \bar{\tau}}$ and $\boldsymbol{x}_0$ forms a star graph. We define:
$$
\begin{gathered}
q_{\sigma, \tau}\left(\boldsymbol{x}_t \mid \boldsymbol{x}_0\right)=\mathcal{N}\left(\sqrt{\alpha_t} \boldsymbol{x}_0,\left(1-\alpha_t\right) \boldsymbol{I}\right) \quad \forall t \in \bar{\tau} \cup\{T\} \\
q_{\sigma, \tau}\left(\boldsymbol{x}_{\tau_{i-1}} \mid \boldsymbol{x}_{\tau_i}, \boldsymbol{x}_0\right)=\mathcal{N}\left(\sqrt{\alpha_{\tau_{i-1}}} \boldsymbol{x}_0+\sqrt{1-\alpha_{\tau_{i-1}}-\sigma_{\tau_i}^2} \cdot \frac{\boldsymbol{x}_{\tau_i}-\sqrt{\alpha_{\tau_i}} \boldsymbol{x}_0}{\sqrt{1-\alpha_{\tau_i}}}, \sigma_{\tau_i}^2 \boldsymbol{I}\right) \forall i \in[S]
\end{gathered}
$$
where the coefficients are chosen such that:
$$
q_{\sigma, \tau}\left(\boldsymbol{x}_{\tau_i} \mid \boldsymbol{x}_0\right)=\mathcal{N}\left(\sqrt{\alpha_{\tau_i}} \boldsymbol{x}_0,\left(1-\alpha_{\tau_i}\right) \boldsymbol{I}\right) \quad \forall i \in[S]
$$
i.e., the "marginals" match.
**The corresponding "generative process" is defined as:**
$$
p_\theta\left(\boldsymbol{x}_{0: T}\right):=\underbrace{p_\theta\left(\boldsymbol{x}_T\right) \prod_{i=1}^S p_\theta^{\left(\tau_i\right)}\left(\boldsymbol{x}_{\tau_{i-1}} \mid \boldsymbol{x}_{\tau_i}\right)}_{\text {use to produce samples }} \times \underbrace{\prod_{t \in \bar{\tau}} p_\theta^{(t)}\left(\boldsymbol{x}_0 \mid \boldsymbol{x}_t\right)}_{\text {in variational objective }}
$$
where only part of the models are actually being used to produce samples. The conditionals are:
$$
\begin{gathered}
p_\theta^{\left(\tau_i\right)}\left(\boldsymbol{x}_{\tau_{i-1}} \mid \boldsymbol{x}_{\tau_i}\right)=q_{\sigma, \tau}\left(\boldsymbol{x}_{\tau_{i-1}} \mid \boldsymbol{x}_{\tau_i}, f_\theta^{\left(\tau_i\right)}\left(\boldsymbol{x}_{\tau_{i-1}}\right)\right) \quad \text { if } i \in[S], i>1 \\
p_\theta^{(t)}\left(\boldsymbol{x}_0 \mid \boldsymbol{x}_t\right)=\mathcal{N}\left(f_\theta^{(t)}\left(\boldsymbol{x}_t\right), \sigma_t^2 \boldsymbol{I}\right) \quad \text { otherwise, }
\end{gathered}
$$
**所以这里也得到了加速子集的逆向推导方程，应该和前面的一样--论文里没给出形式**

where we leverage $q_{\sigma, \tau}\left(\boldsymbol{x}_{\tau_{i-1}} \mid \boldsymbol{x}_{\tau_i}, \boldsymbol{x}_0\right)$ as part of the inference process (similar to what we have done in Section 3). The resulting variational objective becomes (define $\boldsymbol{x}_{\tau_{L+1}}=\varnothing$ for conciseness):
$$
\begin{aligned}
& J\left(\epsilon_\theta\right)=\mathbb{E}_{\boldsymbol{x}_{0: T} \sim q_{\sigma, \tau}\left(\boldsymbol{x}_{0: T}\right)}\left[\log q_{\sigma, \tau}\left(\boldsymbol{x}_{1: T} \mid \boldsymbol{x}_0\right)-\log p_\theta\left(\boldsymbol{x}_{0: T}\right)\right] \\
&=\mathbb{E}_{\boldsymbol{x}_{0: T} \sim q_{\sigma, \tau}\left(\boldsymbol{x}_{0: T}\right)}[ {\left[\sum _ { t \in \overline { \tau } } D _ { \mathrm { KL } } \left(q_{\sigma, \tau}\left(\boldsymbol{x}_t \mid \boldsymbol{x}_0\right) \| p_\theta^{(t)}\left(\boldsymbol{x}_0 \mid \boldsymbol{x}_t\right)\right.\right.} \\
&\left.\left.+\sum_{i=1}^L D_{\mathrm{KL}}\left(q_{\sigma, \tau}\left(\boldsymbol{x}_{\tau_{i-1}} \mid \boldsymbol{x}_{\tau_i}, \boldsymbol{x}_0\right) \| p_\theta^{\left(\tau_i\right)}\left(\boldsymbol{x}_{\tau_{i-1}} \mid \boldsymbol{x}_{\tau_i}\right)\right)\right)\right]
\end{aligned}
$$
where each KL divergence is between two Gaussians with variance independent of $\theta$. A similar argument to the proof used in Theorem 1 can show that the variational objective $J$ can also be converted to an objective of the form $L_\gamma$.



论文共设计了两种方法来采样子序列，分别是:

- Linear: 采用线性的序列 $\tau_i=\lfloor c i\rfloor$;
- Quadratic: 采样二次方的序列 $\tau_i=\left\lfloor c i^2\right\rfloor$; 列，其它数据集均采用Linear序列。

### Sampling Speedup(子序列Sampling)

从损失函数 $\left\|\varepsilon-\boldsymbol{\epsilon} \boldsymbol{\theta}\left(\bar{\alpha}_t \boldsymbol{x}_0+\bar{\beta}_t \varepsilon, t\right)\right\|^2$  or （以DDPM or DDIM 参数） $\left\|\varepsilon-\boldsymbol{\epsilon} \boldsymbol{\theta}\left(\sqrt{\bar{\alpha}_t }\boldsymbol{x}_0+\sqrt{1-\bar\alpha_t} \varepsilon, t\right)\right\|^2$ 可以看出，给定了各个 $\overline{\boldsymbol{\alpha}}_t$ ，训练过程也就确定了。从这个过程中，DDIM进步留意到了如下事实:

**DDPM的训练结果实质上包含了它的任意子序列参数的训练结果**

具体来说，设 $\tau=\left[\tau_1, \tau_2, \ldots, \tau_{\operatorname{dim}(\tau)}\right]$ 是 $[1,2, \cdots, T]$ 的任意子序列，那么我们以 $\bar{\alpha}_{\tau 1}, \bar{\alpha}_{\tau 2}, \cdots, \bar{\alpha}_{\operatorname{dim}(\tau)}$ 为参数训练一个扩散步数为 $\operatorname{dim}(\tau)$ 步的DDPM，其目标函数实际上是原来以 $\bar{\alpha}_1, \bar{\alpha}_2, \cdots, \bar{\alpha}_T$ 的 $T$ 步DDPM的目标函数的一个子集! 所以在模型拟合能力足够好的情况下，它其实 包含了任意子序列参数的训练结果。

**也就是说 $q\left(\mathbf{x}_{\tau_i} \mid \mathbf{x}_0\right)$ 是 $q\left(\mathbf{x}_{\tau} \mid \mathbf{x}_0\right)$ 的子集**

那么反过来想，如果有一个训练好的 $T$ 步DDPM模型，我们也可以将它当成是以 $\bar{\alpha}_{\tau 1}, \bar{\alpha}_{\tau 2}, \cdots, \bar{\alpha}_{\operatorname{dim}(\tau)}$ 为参数训练出来的 $\operatorname{dim}(\boldsymbol{\tau})$ 步模型，而既然是 $\operatorname{dim}(\tau)$ 步的模型，生成过程也就 只需要 $\operatorname{dim}(\tau)$ 步了, 根据式 $(10)$ 有:
$$
p\left(\boldsymbol{x}_{\tau i-1} \mid \boldsymbol{x}_{\tau i}\right) \approx \mathcal{N}\left(\boldsymbol{x}_{\tau i-1} ; \frac{\bar{\alpha}_{\tau i-1}}{\bar{\alpha}_{\tau_i}}\left(\boldsymbol{x}_{\tau i}-\left(\bar{\beta}_{\tau_i}-\frac{\bar{\alpha}_{\tau_i}}{\bar{\alpha}_{\tau_{i-1}}} \sqrt{\bar{\beta}_{\tau i-1}^2-\tilde{\sigma}_{\tau_i}^2}\right) \boldsymbol{\epsilon}\left(\boldsymbol{x}_{\tau i}, \tau_i\right)\right), \bar{\sigma}_{\tau_i}^2 \boldsymbol{I}\right)
$$
这就是加速采样的生成过程了，从原来的 $T$ 步扩散生成变成了 $\operatorname{dim}(\boldsymbol{\tau})$ 步。要注意不能直接将式(10)的 $\alpha_t$ 换成 $\alpha_{\tau_i}$ ，因为我们说过 $\alpha_t$ 是派生记号而已，它实际上等于 $\frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}$ ，因此 $\alpha_t$ 要换成 $\frac{\bar{\alpha}_{\tau_i}}{\bar{\alpha}_{\tau_{i-1}}}$ 才对。同 理， $\tilde{\sigma}_{\tau i}$ 也不是直接取 $\sigma_{\tau i}$ ，而是在将其定义全部转化为 $\bar{\alpha}, \bar{\beta}$ 符号后，将 $t$ 替换为 $\tau_i 、 t-1$ 替换为 $\tau_{i-1}$, 比如式 $(11)$ 对应的 $\tilde{\sigma}_{\tau i}$ 为
$$
\sigma_t=\frac{\bar{\beta}_{t-1} \beta_t}{\bar{\beta}_t}=\frac{\bar{\beta}_{t-1}}{\bar{\beta}_t} \sqrt{1-\frac{\bar{\alpha}_t^2}{\bar{\alpha}_{t-1}^2}} \rightarrow \frac{\bar{\beta}_{\tau_{i-1}}}{\bar{\beta}_{\tau i}} \sqrt{1-\frac{\bar{\alpha}_{\tau i}^2}{\bar{\alpha}_{\tau i-1}^2}}=\tilde{\sigma}_{\tau i}
$$
那我们为什么干脆不直接训练一个 $\operatorname{dim}(\boldsymbol{\tau})$ 步的扩散模型，而是要先训练 $T>\operatorname{dim}(\boldsymbol{\tau})$ 步然后去做子序列采样? 笔者认为可能有两方面的考虑: 一方面从 $\operatorname{dim}(\tau)$ 步生成来说，训练更多步 数的模型也许能增强泛化能力；另一方面，通过子序列 $\tau$ 进行加速只是其中一种加速手段，训练更充 分的 $T$ 步允许我们尝试更多的其他加速手段，但并不会显著增加训练成本。

### DDIM ODES/VE-SDE

 we can rewrite the DDIM iterate according to Eq. (12), and its similarity to Euler integration for solving ordinary differential equations (ODEs) becomes more apparent:
$$
\frac{\boldsymbol{x}_{t-\Delta t}}{\sqrt{\alpha_{t-\Delta t}}}=\frac{\boldsymbol{x}_t}{\sqrt{\alpha_t}}+\left(\sqrt{\frac{1-\alpha_{t-\Delta t}}{\alpha_{t-\Delta t}}}-\sqrt{\frac{1-\alpha_t}{\alpha_t}}\right) \epsilon_\theta^{(t)}\left(\boldsymbol{x}_t\right) \tag {13}
$$
To derive the corresponding ODE, we can reparametrize $(\sqrt{1-\alpha} / \sqrt{\alpha})$ with $\sigma$ and $(x / \sqrt{\alpha})$ with $\overline{\boldsymbol{x}}$. 

> Proof. In the context of the proof, we consider $t$ as a continous, independent "time" variable and $\boldsymbol{x}$ and $\alpha$ as functions of $t$. First, let us consider a reparametrization between DDIM and the VE-SDE ${ }^8$ by introducing the variables $\overline{\boldsymbol{x}}$ and $\sigma$ :
> $$
> \overline{\boldsymbol{x}}(t)=\overline{\boldsymbol{x}}(0)+\sigma(t) \epsilon, \quad \epsilon \sim \mathcal{N}(0, \boldsymbol{I})
> $$
> We can then define $\alpha(t)$ and $\boldsymbol{x}(t)$ corresponding to DDIM case as:
> $$
> \begin{gathered}
> \bar{x}(t)=\frac{\boldsymbol{x}(t)}{\sqrt{\alpha(t)}} \\
> \sigma(t)=\sqrt{\frac{1-\alpha(t)}{\alpha(t)}} .
> \end{gathered}
> $$
> This also means that:
> $$
> \begin{aligned}
> x(t) & =\frac{\bar{x}(t)}{\sqrt{\sigma^2(t)+1}} \\
> \alpha(t) & =\frac{1}{1+\sigma^2(t)},
> \end{aligned}
> $$
> which establishes an bijection between $(\boldsymbol{x}, \alpha)$ and $(\overline{\boldsymbol{x}}, \sigma)$. From Equation (4) we have (note that $\alpha(0)=1)$ :
> $$
> \frac{\boldsymbol{x}(t)}{\sqrt{\alpha(t)}}=\frac{\boldsymbol{x}(0)}{\sqrt{\alpha(0)}}+\sqrt{\frac{1-\alpha(t)}{\alpha(t)}} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \boldsymbol{I})
> $$
> which can be reparametrized into a form that is consistent with VE-SDE:
> $$
> \overline{\boldsymbol{x}}(t)=\overline{\boldsymbol{x}}(0)+\sigma(t) \epsilon .
> $$
> Now, we derive the ODE forms for both DDIM and VE-SDE and show that they are equivalent.

In the continuous case, $\sigma$ and $\boldsymbol{x}$ are functions of $t$, where $\sigma: \mathbb{R}_{\geq 0} \rightarrow \mathbb{R}_{\geq 0}$ is continuous, increasing with $\sigma(0)=0$. Equation (13) with can be treated as a Euler method over the following ODE:
$$
\mathrm{d} \overline{\boldsymbol{x}}(t)=\epsilon_\theta^{(t)}\left(\frac{\overline{\boldsymbol{x}}(t)}{\sqrt{\sigma^2+1}}\right) \mathrm{d} \sigma(t) \tag {14}
$$
where the initial conditions is $\boldsymbol{x}(T) \sim \mathcal{N}(0, \sigma(T))$ for a very large $\sigma(T)$ (which corresponds to the case of $\alpha \approx 0)$. This suggests that with enough discretization steps, the we can also reverse the generation process (going from $t=0$ to $T$ ), **which encodes $\boldsymbol{x}_0$ to $\boldsymbol{x}_T$ and simulates the reverse of the ODE in Eq. (14).** This suggests that unlike DDPM, we can use DDIM to obtain encodings of the observations (as the form of $\boldsymbol{x}_T$ ), which might be useful for other downstream applications that requires latent representations of a model.

Equation (13)  which is equivalent to:
$$
\overline{\boldsymbol{x}}(t-\Delta t)=\overline{\boldsymbol{x}}(t)+(\sigma(t-\Delta t)-\sigma(t)) \cdot \epsilon_\theta^{(t)}(\boldsymbol{x}(t))
$$
Divide both sides by $(-\Delta t)$ and as $\Delta t \rightarrow 0$, we have:
$$
\frac{\mathrm{d} \bar{x}(t)}{\mathrm{d} t}=\frac{\mathrm{d} \sigma(t)}{\mathrm{d} t} \epsilon_\theta^{(t)}\left(\frac{\bar{x}(t)}{\sqrt{\sigma^2(t)+1}}\right) \tag {45}
$$
which is exactly what we have in Equation (14).

We note that for the optimal model, $\epsilon_\theta^{(t)}$ is a minimizer: ==这个是我们前面训练的目标，请记住$J = \mathbb{E}_{t \sim [1, T], \mathbf{x}_0, \boldsymbol{\epsilon}_t} \Big[\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t, t)\|^2 \Big]$==
$$
\epsilon_\theta^{(t)}=\underset{f_t}{\arg \min } \mathbb{E}_{\boldsymbol{x}(0) \sim q(\boldsymbol{x}), \epsilon \sim \mathcal{N}(0, \boldsymbol{I})}\left[\left\|f_t(\boldsymbol{x}(t))-\epsilon\right\|_2^2\right]
$$
where $x(t)=\sqrt{\alpha(t)} x(t)+\sqrt{1-\alpha(t)} \epsilon$.

##### **ODE form for VE-SDE Define** 

$p_t(\overline{\boldsymbol{x}})$ as the data distribution perturbed with $\sigma^2(t)$ variance Gaussian noise. **The probability flow for VE-SDE** is defined as Song et al. (2020) (*Score-based generative modeling through stochastic differential equations)*:
$$
\mathrm{d} \overline{\boldsymbol{x}}=-\frac{1}{2} g(t)^2 \nabla_{\overline{\boldsymbol{x}}} \log p_t(\overline{\boldsymbol{x}}) \mathrm{d} t
$$
==where $g(t)=\sqrt{\frac{\mathrm{d} \sigma^2(t)}{\mathrm{d} t}}$ is the diffusion coefficient, and $\nabla_{\overline{\boldsymbol{x}}} \log p_t(\overline{\boldsymbol{x}})$ is the score of $p_t$.==

The $\sigma(t)$-perturbed score function $\nabla_{\bar{x}} \log p_t(\overline{\boldsymbol{x}})$ is also a minimizer (from denoising score matching (Vincent, 2011)): ==不知道如何推导出这个，也需要看下这里的这篇论文？==
$$
\nabla_{\overline{\boldsymbol{x}}} \log p_t=\underset{g_t}{\arg \min } \mathbb{E}_{\boldsymbol{x}(0) \sim q(\boldsymbol{x}), \epsilon \sim \mathcal{N}(0, \boldsymbol{I})}\left[\left\|g_t(\overline{\boldsymbol{x}})+\epsilon / \sigma(t)\right\|_2^2\right] \tag {48}
$$
where $\overline{\boldsymbol{x}}(t)=\overline{\boldsymbol{x}}(t)+\sigma(t) \epsilon$.
Since there is an equivalence between $x(t)$ and $\overline{\boldsymbol{x}}(t)$, we have the following relationship:
$$
\nabla_{\overline{\boldsymbol{x}}} \log p_t(\overline{\boldsymbol{x}})=-\frac{\epsilon_\theta^{(t)}\left(\frac{\overline{\boldsymbol{x}}(t)}{\sqrt{\sigma^2(t)+1}}\right)}{\sigma(t)} \tag{49}
$$
==不过因为我们知道:== 
$$
\mathbf{s}_\theta\left(\mathbf{x}_t, t\right) \approx \nabla_{\mathbf{x}_t} \log q\left(\mathbf{x}_t\right)=\mathbb{E}_{q\left(\mathbf{x}_0\right)}\left[\nabla_{\mathbf{x}_t} q\left(\mathbf{x}_t \mid \mathbf{x}_0\right)\right]=\mathbb{E}_{q\left(\mathbf{x}_0\right)}\left[-\frac{\epsilon_\theta\left(\mathbf{x}_t, t\right)}{\sqrt{1-\bar{\alpha}_t}}\right]=-\frac{\epsilon_\theta\left(\mathbf{x}_t, t\right)}{\sqrt{1-\bar{\alpha}_t}}
$$
from Equation (46) and Equation (48). Plug Equation (49) and definition of $g(t)$ in Equation (47), we have:
$$
\mathrm{d} \overline{\boldsymbol{x}}(t)=\frac{1}{2} \frac{\mathrm{d} \sigma^2(t)}{\mathrm{d} t} \frac{\epsilon_\theta^{(t)}\left(\frac{\overline{\boldsymbol{x}}(t)}{\sqrt{\sigma^2(t)+1}}\right)}{\sigma(t)} \mathrm{d} t,
$$
and we have the following by rearranging terms:
$$
\frac{\mathrm{d} \overline{\boldsymbol{x}}(t)}{\mathrm{d} t}=\frac{\mathrm{d} \sigma(t)}{\mathrm{d} t} \epsilon_\theta^{(t)}\left(\frac{\overline{\boldsymbol{x}}(t)}{\sqrt{\sigma^2(t)+1}}\right)
$$
which is equivalent to Equation (45). In both cases the initial conditions are $\overline{\boldsymbol{x}}(T) \sim \mathcal{N}\left(\mathbf{0}, \sigma^2(T) \boldsymbol{I}\right)$, 

so the resulting ODEs are identical. ==也就是说从ODE推导的和从VE-SDE推导的结果是一致的，也即是说DDIM也符合统一框架==

##### **ddim reverse sample用于反向 ODE 加噪**

$$
\frac{x_{t-1}}{\sqrt{\bar{\alpha}_{t-1}}}=\frac{x_t}{\sqrt{\bar{\alpha}_t}}-\frac{\sqrt{1-\bar{\alpha}_t}}{\sqrt{\bar{\alpha}_t}} \epsilon_\theta^{(t)}+\frac{\sqrt{1-\bar{\alpha}_{t-1}}}{\sqrt{\bar{\alpha}_{t-1}}} \epsilon_\theta^{(t)}\left(\boldsymbol{x}_t\right)
$$

当 $\mathrm{t}$ 足够大时可以看做
$$
\frac{\boldsymbol{x}_{t-\Delta t}}{\sqrt{\bar{\alpha}_{t-\Delta t}}}=\frac{x_t}{\sqrt{\bar{\alpha}_t}}+\left(\sqrt{\frac{1-\bar{\alpha}_{t-\Delta t}}{\bar{\alpha}_{t-\Delta t}}}-\sqrt{\frac{1-\bar{\alpha}_t}{\bar{\alpha}_t}}\right) \epsilon_\theta^{(t)}\left(\boldsymbol{x}_t\right)
$$
而后进行换元, 令 $\sigma=(\sqrt{1-\bar{\alpha}} / \sqrt{\bar{\alpha}}), \bar{x}=x / \sqrt{\bar{\alpha}}$, 带入得到:
$$
\mathrm{d} \overline{\boldsymbol{x}}(t)=\epsilon_\theta^{(t)}\left(\frac{\bar{x}(t)}{\sqrt{\sigma^2+1}}\right) \mathrm{d} \sigma(t)
$$
于是, 基于这个 ODE 结果, 能通过 $\bar{x}(t)+d \bar{x}(t)$ 计算得到 $\bar{x}(t+1)$ 与 $x_{t+1}$

根据 github - openai/improved-diffusion๔, 其实现根据 ODE 反向采样的方式为：直接根据公式 (5) 进行变换, 把 $t-1$ 换成 $t+1$ :


$$
\boldsymbol{x}_{t+1}=\sqrt{\bar{\alpha}_{t+1}} \underbrace{\left(\frac{\boldsymbol{x}_t-\sqrt{1-\bar{\alpha}_t} \epsilon_\theta^{(t)}\left(\boldsymbol{x}_t\right)}{\sqrt{\bar{\alpha}_t}}\right)}_{\text {"predicted } \boldsymbol{x}_0 "}+\underbrace{\sqrt{1-\bar{\alpha}_{t+1}} \cdot \epsilon_\theta^{(t)}\left(\boldsymbol{x}_t\right)}_{\text {"direction pointing to } x_t "}+\underbrace{\sigma_t \epsilon_t}_{\text {random noise }}
$$
而参考公式 (11) 的推导过程, (12) 可以看成下面这种形式:
$$
\frac{x_{t+\Delta t}}{\sqrt{\bar{\alpha}_{t+\Delta t}}}=\frac{x_t}{\sqrt{\bar{\alpha}_t}}+\left(\sqrt{\frac{1-\bar{\alpha}_{t+\Delta t}}{\bar{\alpha}_{t+\Delta t}}}-\sqrt{\frac{1-\bar{\alpha}_t}{\bar{\alpha}_t}}\right) \epsilon_\theta^{(t)}\left(\boldsymbol{x}_t\right)
$$

## 4.8 Learning a Covariance matrix (Parameterization of reverse process variance )

- DDPM authors said that it's better to use a fixed covariance matrix $\boldsymbol{\Sigma}_\theta\left(\mathbf{x}_t, t\right)=\sigma_t^2 \mathbf{I}$ where $\sigma_t^2=\beta_t$ or $\sigma_t^2=\tilde{\beta}_t=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t$.  
- The intuition is that covariance does not contribute as significantly as the mean does to the learned conditional distributions during the reverse process
- However, it can still help us improve log-likelihood!
-  So, [Nichol & Dhariwal (2021)](https://arxiv.org/abs/2102.09672) propose to learn $\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)$ as an interpolation between $β_t$ and $\tilde{\beta}_t$ by model predicting a mixing vector v :

$$
\Sigma_\theta\left(x_t, t\right)=\exp \left(v \log \beta_t+(1-v) \log \tilde{\beta}_t\right)
$$
This modification leads to better likelihood estimates while maintaining image quality!

<img src="/assets/BERTGPTDiffusion%20Research.assets/improved-DDPM-nll.png" alt="img" style="zoom:33%;" />

## 4.9 DDPM vs DDIM



|                                                          | DDPM                                                         | DDIM |
| :----------------------------------------------------------- | ---- | ------------------------------------------------------------ |
| forward | $$q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I}) \quad \\q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = \prod^T_{t=1} q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) \\q\left(\mathbf{x}_t \mid \mathbf{x}_0\right)=\mathcal{N}\left(\mathbf{x}_t ; \sqrt{\alpha_t} \mathbf{x}_0,\left(1-\alpha_t\right) \mathbf{I}\right)$$ | $\int p\left(x_{t-1} \mid x_t, x_0\right) p\left(x_t \mid x_0\right) d x_t=p\left(x_{t-1} \mid x_0\right) \tag {3}$ and<br /> $$ q\left(\mathbf{x}_t \mid \mathbf{x}_0\right)=\mathcal{N}\left(\mathbf{x}_t ; \sqrt{\alpha_t} \mathbf{x}_0,\left(1-\alpha_t\right) \mathbf{I}\right) \\ q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_0\right)=\mathcal{N}\left(\mathbf{x}_{t-1} ; \sqrt{\alpha_{t-1}} \mathbf{x}_0,\left(1-\alpha_{t-1}\right) \mathbf{I}\right)$$ |
| reverse | $q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \color{blue}{\tilde{\boldsymbol{\mu}}}(\mathbf{x}_t, \mathbf{x}_0), \color{red}{\tilde{\beta}_t} \mathbf{I})$  <br />$$ \begin{aligned} \tilde{\boldsymbol{\mu}}(\mathbf{x}_t, \mathbf{x}_0) &= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0\\ \tilde{\beta}_t &= \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t \end{aligned}$$ | $$q_{\sigma, \tau}(\mathbf{x}_{\tau_{i-1}} \vert \mathbf{x}_{\tau_t}, \mathbf{x}_0)= \mathcal{N}(\mathbf{x}_{\tau_{i-1}}; \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \frac{\mathbf{x}_{\tau_i} - \sqrt{\bar{\alpha}_t}\mathbf{x}_0}{\sqrt{1 - \bar{\alpha}_t}}, \sigma_t^2 \mathbf{I}) \tag{7-DDIM}$$ |
| sampling | 因为 $$\mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t)$$ 所以 <br /> $$ \begin{aligned} \boldsymbol{\mu}_\theta(\mathbf{x}_t, t) &= \color{cyan}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \Big)} \\ \text{Thus }\mathbf{x}_{t-1} &= \mathcal{N}(\mathbf{x}_{t-1}; \frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \Big), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)) \end{aligned} $$ | 因为 $f_\theta^{(t)}\left(\boldsymbol{x}_t\right):=\left(\boldsymbol{x}_t-\sqrt{1-\alpha_t} \cdot \epsilon_\theta^{(t)}\left(\boldsymbol{x}_t\right)\right) / \sqrt{\alpha_t} \tag {9-DDIM}$  <br />where we have $$q_\sigma\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t \right) \approx q_\sigma\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, f_\theta^{(t)}\left(\boldsymbol{x}_t\right)\right)$$ 所以 $$ \text {DENOISING DIFFUSION IMPLICIT MODELS }\\ \boldsymbol{x}_{t-1}=\sqrt{\alpha_{t-1}} \underbrace{\left(\frac{\boldsymbol{x}_t-\sqrt{1-\alpha_t} \epsilon_\theta^{(t)}\left(\boldsymbol{x}_t\right)}{\sqrt{\alpha_t}}\right)}_{\text {"predicted } \boldsymbol{x}_0 \text { " }}+\underbrace{\sqrt{1-\alpha_{t-1}-\sigma_t^2} \cdot \epsilon_\theta^{(t)}\left(\boldsymbol{x}_t\right)}_{\text {“direction pointing to } \boldsymbol{x}_t \text { " }}+\underbrace{\sigma_t \epsilon_t}_{\text {random noise }} \tag {12}$$<br />或者 <br />$$\tilde{\boldsymbol{\mu}}\left(\mathbf{x}_t \right) &=\frac{1}{\alpha_t}\left(\boldsymbol{x}_t-\left(\bar{\beta}_t-\alpha_t \sqrt{\bar{\beta}_{t-1}^2-\sigma_t^2}\right) \boldsymbol{\epsilon}_{\boldsymbol{\theta}}\left(\boldsymbol{x}_t, t\right)\right) \\ & \text { 其中 } \alpha_t=\frac{\bar{\alpha} t}{\bar{\alpha}_{t-1}}$$ |
| training loss | $$\begin{aligned} loss &= - \log p_\theta(\mathbf{x}_0) = ELBO  \\ &= \|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t, t)\|^2 \end{aligned}$$ | $$\begin{aligned} J_\sigma\left(\epsilon_\theta\right)&:=\mathbb{E}_{\boldsymbol{x}_{0: T} \sim q_\sigma\left(\boldsymbol{x}_{0: T}\right)}\left[\log q_\sigma\left(\boldsymbol{x}_{1: T} \mid \boldsymbol{x}_0\right)-\log p_\theta\left(\boldsymbol{x}_{0: T}\right)\right] \\ &= \mathbb{E}_{\boldsymbol{x}_{0: T} \sim q_\sigma\left(\boldsymbol{x}_{0: T}\right)}\left[\log q_\sigma\left(\boldsymbol{x}_T \mid \boldsymbol{x}_0\right)+\sum_{t=2}^T \log  q_\sigma\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{x}_0\right)-\sum_{t=1}^T \log p_\theta^{(t)}\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t\right)-\log p_\theta\left(\boldsymbol{x}_T\right)\right] \end{aligned}$$ |
| score | $\mathbf{s}_\theta\left(\mathbf{x}_t, t\right) \approx \nabla_{\mathbf{x}_t} \log q\left(\mathbf{x}_t\right)=\mathbb{E}_{q\left(\mathbf{x}_0\right)}\left[\nabla_{\mathbf{x}_t} q\left(\mathbf{x}_t \mid \mathbf{x}_0\right)\right] \\ =\mathbb{E}_{q\left(\mathbf{x}_0\right)}\left[-\frac{\epsilon_\theta\left(\mathbf{x}_t, t\right)}{\sqrt{1-\bar{\alpha}_t}}\right]=-\frac{\epsilon_\theta\left(\mathbf{x}_t, t\right)}{\sqrt{1-\bar{\alpha}_t}}$ <br /> where $q\left(\mathbf{x}_t \mid \mathbf{x}_0\right) \sim \mathcal{N}\left(\sqrt{\bar{\alpha}_t} \mathbf{x}_0,\left(1-\bar{\alpha}_t\right) \mathbf{I}\right)$ | $\nabla_{\overline{\boldsymbol{x}}} \log p_t(\overline{\boldsymbol{x}})=-\frac{\epsilon_\theta^{(t)}\left(\frac{\overline{\boldsymbol{x}}(t)}{\sqrt{\sigma^2(t)+1}}\right)}{\sigma(t)} \tag{49}$ , where $\sigma(t)=\sqrt{\frac{1-\alpha(t)}{\alpha(t)}}$ |
|  | $X_{t+1}=X_t+d t \nabla \log p\left(X_t\right)+\mathcal{N}(0,2 d t)$ <br />$\begin{equation}   x_{n+1} = x_n + \nabla \log p(x_n) \epsilon + \sigma \sqrt{2 \epsilon}\ z\end{equation}$ |  |
| Classifier Guided | $\nabla_x \log p_\gamma(x \mid y) = \nabla_x \log p(x) + \gamma \nabla_x \log p(y \mid x) .$ | $$ \begin{aligned} \nabla_{\mathbf{x}_t} \log q\left(\mathbf{x}_t, y\right) & =\nabla_{\mathbf{x}_t} \log q\left(\mathbf{x}_t\right)+\nabla_{\mathbf{x}_t} \log q\left(y \mid \mathbf{x}_t\right) \\& \approx-\frac{1}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta\left(\mathbf{x}_t, t\right)+\nabla_{\mathbf{x}_t} \log f_\phi\left(y \mid \mathbf{x}_t\right) \\ & =-\frac{1}{\sqrt{1-\bar{\alpha}_t}}\left(\epsilon_\theta\left(\mathbf{x}_t, t\right)-\sqrt{1-\bar{\alpha}_t} \nabla_{\mathbf{x}_t} \log f_\phi\left(y \mid \mathbf{x}_t\right)\right) \\ &= -\frac{1}{\sqrt{1-\bar{\alpha}_t}} \bar{\epsilon}_\theta\left(\mathbf{x}_t, t\right) \end{aligned}  $$<br />where $\bar{\epsilon}_\theta\left(\mathbf{x}_t, t\right)=\epsilon_\theta\left(x_t, t\right)-\sqrt{1-\bar{\alpha}_t} \nabla_{\mathbf{x}_t} \log f_\phi\left(y \mid \mathbf{x}_t\right)$ |


## 4.10 Latent diffusion models (**LDM**; [Rombach & Blattmann, et al. 2022](https://arxiv.org/abs/2112.10752))

**[Stable Diffusion (2021)](https://arxiv.org/abs/2112.10752) differs from the previous diffusion models by working in the latent space instead of pixel space.** It first compresses images via a variational autoencoder (VAE) into a more efficient and lower dimensional latent embedding. Next, the diffusion model learns to generate latent (i.e., compressed) representations of images which are then decoded into images via the VAE decoder.

Similar to DALL·E and its visual codebook, **latent space is motivated by the observation that most pixels in an image are imperceptible details that are semantically meaningless**. However, because regular diffusion models are trained and evaluated in the pixel space, it leads to unnecessary computation and thus costly training and inference. Thus, the paper proposes diffusion on compressed images where the imperceptible details are excluded.

<img src="/assets/BERTGPTDiffusion%20Research.assets/stable-diffusion.jpg" alt="Using a VAE to encode images from pixel space to latent space (left)" style="zoom:50%;" />

In Stable Diffusion, the VAE encodes noised images (via $\mathcal{E}$ ) into a low-dimensional latent representation which is fed into the UNet. It then decodes UNet-generated latent representations (via $\mathcal{D}$ ) into human-understandable images. The VAE has a reduction factor of 8 , where the original image pixel space of $3 \times 512 \times 512$ is encoded into latent space of $6 \times 64 \times 64$, thus requiring $1 / 8 \times 1 / 8=1 / 64$ of the memory.  During sampling, only the VAE decoder is needed.

Stable Diffusion uses the CLIP text encoder. (But as Imagen has demonstrated, probably any sufficiently large text-only LLM can be used).

Latent diffusion leads to faster training and sampling because we’re now working in the latent—instead of pixel—space. This leads to lower cost which leads to more experiments. The lower memory requirement also allows sampling run on consumer-grade laptops, putting text-to-image generation in the hands of regular hackers.



> **Remodel the Diffusion models:**  (High-Resolution Image Synthesis with Latent Diffusion Models,  Appendix B. Detailed Information on Denoising Diffusion Models)
>
> Diffusion models can be specified in terms of a signal-to-noise ratio $\operatorname{SNR}(t)=\frac{\alpha_t^2}{\sigma_t^2}$ consisting of sequences $\left(\alpha_t\right)_{t=1}^T$ and $\left(\sigma_t\right)_{t=1}^T$ which, starting from a data sample $x_0$, define a forward diffusion process $q$ as
> $$
> q\left(x_t \mid x_0\right)=\mathcal{N}\left(x_t \mid \alpha_t x_0, \sigma_t^2 \mathbb{I}\right)
> $$
> with the Markov structure for $s<t$ :
> $$
> \begin{aligned}
> q\left(x_t \mid x_s\right) & =\mathcal{N}\left(x_t \mid \alpha_{t \mid s} x_s, \sigma_{t \mid s}^2 \mathbb{I}\right) \\
> \alpha_{t \mid s} & =\frac{\alpha_t}{\alpha_s} \\
> \sigma_{t \mid s}^2 & =\sigma_t^2-\alpha_{t \mid s}^2 \sigma_s^2
> \end{aligned}
> $$
> Denoising diffusion models are generative models $p\left(x_0\right)$ which revert this process with a similar Markov structure running backward in time, i.e. they are specified as
> $$
> p\left(x_0\right)=\int_z p\left(x_T\right) \prod_{t=1}^T p\left(x_{t-1} \mid x_t\right)
> $$
> The evidence lower bound (ELBO) associated with this model then decomposes over the discrete time steps as
> $$
> -\log p\left(x_0\right) \leq \mathbb{K} \mathbb{L}\left(q\left(x_T \mid x_0\right) \mid p\left(x_T\right)\right)+\sum_{t=1}^T \mathbb{E}_{q\left(x_t \mid x_0\right)} \mathbb{K} \mathbb{L}\left(q\left(x_{t-1} \mid x_t, x_0\right) \mid p\left(x_{t-1} \mid x_t\right)\right)
> $$
> > comparing to DDPM: 
> > $$
> > \begin{aligned}
> > -\log p\left(x_0\right) &\leq \mathbb{E}_q \Big[ \log \frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \\
> > & = 
> > \mathbb{E}_q [\underbrace{D_\text{KL}(q(\mathbf{x}_T \vert \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_T))}_{L_T} + \sum_{t=2}^T \underbrace{D_\text{KL}(q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t))}_{L_{t-1}} \underbrace{- \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)}_{L_0}]
> > \end{aligned}
> > $$
>
> The prior $p\left(x_T\right)$ is typically choose as a standard normal distribution and the first term of the ELBO then depends only on the final signal-to-noise ratio $\operatorname{SNR}(T)$. To minimize the remaining terms, a common choice to parameterize $p\left(x_{t-1} \mid x_t\right)$ is to specify it in terms of the true posterior $q\left(x_{t-1} \mid x_t, x_0\right)$ but with the unknown $x_0$ replaced by an estimate $x_\theta\left(x_t, t\right)$ based on the current step $x_t$. This gives (**Variational diffusion models**)
> $$
> \begin{aligned}
> p\left(x_{t-1} \mid x_t\right) & :=q\left(x_{t-1} \mid x_t, x_\theta\left(x_t, t\right)\right) \\
> & =\mathcal{N}\left(x_{t-1} \mid \mu_\theta\left(x_t, t\right), \sigma_{t \mid t-1}^2 \frac{\sigma_{t-1}^2}{\sigma_t^2} \mathbb{I}\right),
> \end{aligned}
> $$
> where the mean can be expressed as
> $$
> \mu_\theta\left(x_t, t\right)=\frac{\alpha_{t \mid t-1} \sigma_{t-1}^2}{\sigma_t^2} x_t+\frac{\alpha_{t-1} \sigma_{t \mid t-1}^2}{\sigma_t^2} x_\theta\left(x_t, t\right) .
> $$
> In this case, the sum of the ELBO simplify to
> $$
> \sum_{t=1}^T \mathbb{E}_{q\left(x_t \mid x_0\right)} \mathbb{K} \mathbb{L}\left(q\left(x_{t-1} \mid x_t, x_0\right) \mid p\left(x_{t-1}\right)=\sum_{t=1}^T \mathbb{E}_{\mathcal{N}(\epsilon \mid 0, \mathbb{I})} \frac{1}{2}(\operatorname{SNR}(t-1)-\operatorname{SNR}(t))\left\|x_0-x_\theta\left(\alpha_t x_0+\sigma_t \epsilon, t\right)\right\|^2\right.
> $$
> Following [30], we use the reparameterization
> $$
> \epsilon_\theta\left(x_t, t\right)=\left(x_t-\alpha_t x_\theta\left(x_t, t\right)\right) / \sigma_t
> $$
> to express the reconstruction term as a denoising objective,
> $$
> \left\|x_0-x_\theta\left(\alpha_t x_0+\sigma_t \epsilon, t\right)\right\|^2=\frac{\sigma_t^2}{\alpha_t^2}\left\|\epsilon-\epsilon_\theta\left(\alpha_t x_0+\sigma_t \epsilon, t\right)\right\|^2
> $$
> and the reweighting, which assigns each of the terms the same weight and results in Eq. (1).

### General

runs the diffusion process in the latent space instead of pixel space, making training cost lower and inference speed faster. It is motivated by the observation that most bits of an image contribute to perceptual details and the semantic and conceptual composition still remains after aggressive compression. 

LDM loosely decomposes the perceptual compression and semantic compression with generative modeling learning by **first trimming off pixel-level redundancy with autoencoder** and **then manipulate/generate semantic concepts with diffusion process on learned latent**.

>  Contribution
>
> - Diffusion model是一种likelihood-based的模型，相比GAN可以取得更好的生成效果。然而该 模型是一种自回归模型，需要反复迭代计算，因而训练和推理都十分昂贵。本文提出一种 diffusion的过程改为在latent space上做的方法，从而大大减少计算复杂度，同时也能达到十分 不错的生成效果。（"democratizing" research on DMs)，在unconditional image synthesis, inpainting, super-resolution都能表现不错
> - 相比于其它进行压缩的方法，本文的方法可以生成更细致的图像，并且在高分辨率 (风景图之类 的，最高达 $1024^2 p x$ 都无压力) 的生成也表现得很好。
> - 提出了cross-attention的方法来实现多模态训练，使得class-condition, text-to-image， layout-to-image也可以实现。

The perceptual compression process relies on an autoencoder model. 

- By using the trained encoder ***E\***, we can encode the full-sized image into lower dimensional latent data (compressed data).   An encoder $\mathcal{E}$ is used to compress the input image $\mathbf{x} \in \mathbb{R}^{H \times W \times 3}$ to a smaller $2 \mathrm{D}$ latent vector $\mathbf{z}=\mathcal{E}(\mathbf{x}) \in \mathbb{R}^{h \times w \times c}$, where the downsampling rate $f=H / h=W / w=2^m, m \in \mathbb{N}$. 

  After encoding the images into latent data, the forward and reverse diffusion processes will be done in the latent space.

  ![Autoencoder](/assets/BERTGPTDiffusion%20Research.assets/1mSHOIu_xDdPAF7-Q5quJmw.png)

- By using the trained decoder ***D\***, we can decode the latent data back into an image.   an decoder $\mathcal{D}$ reconstructs the images from the latent vector, $\tilde{\mathbf{x}}=\mathcal{D}(\mathbf{z})$. The paper explored two types of regularization in autoencoder training to avoid arbitrarily high-variance in the latent spaces.

  - KL-reg: A small KL penalty towards a standard normal distribution over the learned latent, similar to VAE.

  - VQ-reg: Uses a vector quantization layer within the decoder, like VQVAE but the quantization layer is absorbed by the decoder.


![Overview of the Stable Diffusion model](/assets/BERTGPTDiffusion%20Research.assets/1KgT9m7wgbxyCWqmPqETCyQ.png)

1. Forward Diffusion Process → add noise to the **latent data**.
2. Reverse Diffusion Process → remove noise from the **latent data**.

The diffusion and denoising processes happen on the latent vector $\mathbf{z}$. The denoising model is a timeconditioned U-Net, augmented with the cross-attention mechanism to handle flexible conditioning information for image generation (e.g. class labels, semantic maps, blurred variants of an image). The design is equivalent to fuse representation of different modality into the model with cross-attention mechanism. Each type of conditioning information is paired with a domain-specific encoder $\boldsymbol{\tau}_{\boldsymbol{\theta}}$ to project the conditioning input $y$ to an intermediate representation that can be mapped into cross-attention component, $\tau_\theta(y) \in \mathbb{R}^{M \times d_\tau}$ :
$$
\begin{aligned}
& \text { Attention }(\mathbf{Q}, \mathbf{K}, \mathbf{V})=\operatorname{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^{\top}}{\sqrt{d}}\right) \cdot \mathbf{V} \\
& \text { where } \mathbf{Q}=\mathbf{W}_Q^{(i)} \cdot \varphi_i\left(\mathbf{z}_i\right), \mathbf{K}=\mathbf{W}_K^{(i)} \cdot \tau_\theta(y), \mathbf{V}=\mathbf{W}_V^{(i)} \cdot \tau_\theta(y) \\
& \text { and } \mathbf{W}_Q^{(i)} \in \mathbb{R}^{d \times d_\epsilon^i}, \mathbf{W}_K^{(i)}, \mathbf{W}_V^{(i)} \in \mathbb{R}^{d \times d_\tau}, \varphi_i\left(\mathbf{z}_i\right) \in \mathbb{R}^{N \times d_\epsilon^i}, \tau_\theta(y) \in \mathbb{R}^{M \times d_\tau}
\end{aligned}
$$

### Conditioning

![Overview of the conditioning mechanism](/assets/BERTGPTDiffusion%20Research.assets/1iruOz7EYpsibRGRNkpXpVg.png)

![Conditioning mechanism details](/assets/BERTGPTDiffusion%20Research.assets/1IRTbG2rYv0IUH8HHAxWRrQ.png)

The switch in the above diagram is used to control between different types of conditioning inputs:

- For text inputs, they are first converted into embeddings (vectors) using a language model **𝜏\*θ\*** (e.g. BERT, CLIP), and then they are mapped into the U-Net via the (multi-head) ***Attention(Q, K, V)\*** layer.
- For other spatially aligned inputs (e.g. semantic maps, images, inpainting), the conditioning can be done using concatenation.

### Training

![Training objective for the Stable Diffusion model](/assets/BERTGPTDiffusion%20Research.assets/1iA5bAAa68LWL3w0BmSK7MA.png)

The training objective (loss function) is pretty similar to the one in the pure diffusion model. The only changes are:

- Input latent data $zₜ$ instead of the image $xₜ$.
- Added conditioning input $𝜏_θ(y)$ to the U-Net.

### Sampling

![Stable Diffusion sampling process (denoising)](/assets/BERTGPTDiffusion%20Research.assets/1UQ4fb9mBsEh_EvgKijyzWg.png)

> Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding (arXiv:2205.11487v1)
>
> (1) 扩散过程 （latent variable diffusion)
>
> Diffusion models are latent variable models with latents $\mathbf{z}=\left\{\mathbf{z}_t \mid t \in[0,1]\right\}$ that obey a forward process $q(\mathbf{z} \mid \mathbf{x})$ starting at data $\mathbf{x} \sim p(\mathbf{x})$. 
>
> 
>
> - **This forward process is a Gaussian process that satisfies the Markovian structure:**
>
> $$
> q\left(\mathbf{z}_t \mid \mathbf{x}\right)=\mathcal{N}\left(\mathbf{z}_t ; \alpha_t \mathbf{x}, \sigma_t^2 \mathbf{I}\right), \quad q\left(\mathbf{z}_t \mid \mathbf{z}_s\right)=\mathcal{N}\left(\mathbf{z}_t ;\left(\alpha_t / \alpha_s\right) \mathbf{z}_s, \sigma_{t \mid s}^2 \mathbf{I}\right)
> $$
>
> where $0 \leq s<t \leq 1, \sigma_{t \mid s}^2=\left(1-e^{\lambda_t-\lambda_s}\right) \sigma_t^2$, and $\alpha_t, \sigma_t$ specify a differentiable noise schedule whose $\log$ signal-to-noise-ratio, i.e., $\lambda_t=\log \left[\alpha_t^2 / \sigma_t^2\right]$, decreases with $t$ until $q\left(\mathbf{z}_1\right) \approx \mathcal{N}(\mathbf{0}, \mathbf{I})$. 
>
> > $\mathbf{z}_t=\alpha_t \mathbf{x}+\sigma_t \boldsymbol{\epsilon}_{\mathbf{1}}$ ，其中 $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 。
> > 根据正向过程的定义可得： $\mathrm{z}_s=\alpha_s \mathbf{x}+\sigma_s \epsilon_2$ ，则 $\mathbf{x}=\frac{1}{\alpha_s}\left(\mathbf{z}_s-\sigma_s \epsilon_2\right)$ ，将 $\mathbf{x}$ 代入上式 得:
> > $$
> > \mathbf{z}_t=\frac{\alpha_t}{\alpha_s}\left(\mathbf{z}_s-\sigma_s \epsilon_2\right)+\sigma_t \epsilon_1=\frac{\alpha_t}{\alpha_s} \mathbf{z}_s-\frac{\alpha_t \cdot \sigma_s}{\alpha_s} \epsilon_2+\sigma_t \epsilon_1=\frac{\alpha_t}{\alpha_s} \mathbf{z}_s+\sqrt{\sigma_t^2-\frac{\alpha_t^2 \cdot \sigma_s^2}{\alpha_s^2}} \cdot \epsilon
> > $$
> > 定义: $\sigma_{t \mid s}^2=\left(1-e^{\lambda_t-\lambda_s}\right) \sigma_t^2$ ，其中 $\lambda_t=\log \left[\alpha_t^2 / \sigma_t^2\right]$ ，则:
> > $$
> > \mathbf{z}_t=\left(\alpha_t / \alpha_s\right) \mathbf{z}_s+\sqrt{\sigma_{t \mid s}^2} \cdot \epsilon
> > $$
> > 所以:
> > $$
> > q\left(\mathbf{z}_t \mid \mathbf{x}\right)=\mathcal{N}\left(\mathbf{z}_t ; \alpha_t \mathbf{x}, \sigma_t^2 \mathbf{I}\right), \quad q\left(\mathbf{z}_t \mid \mathbf{z}_s\right)=\mathcal{N}\left(\mathbf{z}_t ;\left(\alpha_t / \alpha_s\right) \mathbf{z}_s, \sigma_{t \mid s}^2 \mathbf{I}\right)
> > $$
> > 式中， $0 \leq s<t \leq 1, \sigma_{t \mid s}^2=\left(1-e^{\lambda_t-\lambda_s}\right) \sigma_t^2, \alpha_t, \sigma_t$ 表示可微噪声 schedule， $\lambda_t=\log \left[\alpha_t^2 / \sigma_t^2\right]$ 表示信噪比，其随着时间 $t$ 逐步降低，直到 $q\left(\mathbf{z}_1\right) \approx \mathcal{N}(\mathbf{0}, \mathbf{I})$ 。
> >
> > 
>
> - For generation, the diffusion model is learned to reverse this forward process. Learning to reverse the forward process can be reduced to learning to denoise $\mathbf{z}_t \sim q\left(\mathbf{z}_t \mid \mathbf{x}\right)$ into an estimate $\hat{\mathbf{x}}_\theta\left(\mathbf{z}_t, \lambda_t, \mathbf{c}\right) \approx \mathbf{x}$ for all $t$, where $\mathbf{c}$ is an optional conditioning signal (such as text embeddings or a low resolution image) drawn from the dataset jointly with $\mathbf{x}$. This is accomplished training $\hat{\mathbf{x}}_\theta$ using a weighted squared error loss
>
> $$
> \mathbb{E}_{\boldsymbol{\epsilon}, t}\left[w\left(\lambda_t\right)\left\|\hat{\mathbf{x}}_\theta\left(\mathbf{z}_t, \lambda_t, \mathbf{c}\right)-\mathbf{x}\right\|_2^2\right]
> $$
>
> $$
> \text { where } t \sim \mathcal{U}([0,1]), \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \text {, and } \mathbf{z}_t=\alpha_t \mathbf{x}+\sigma_t \boldsymbol{\epsilon} \text {. }
> $$
>
> This reduction of generation to denoising is justified as optimizing a weighted variational lower bound on the data log likelihood under the diffusion model, or as a form of denoising score matching $[72,65,28,35]$. We use the $\epsilon$ prediction parameterization, defined as $\hat{\mathbf{x}}_\theta\left(\mathbf{z}_t, \lambda_t, \mathbf{c}\right)=\left(\mathbf{z}_t-\sigma_t \epsilon_\theta\left(\mathbf{z}_t, \lambda_t, \mathbf{c}\right)\right) / \alpha_t$, and we impose a squared error loss on $\epsilon_\theta$ in $\epsilon$ space with $t$ sampled according to a cosine schedule [40]. This corresponds to a particular weighting $w\left(\lambda_t\right)$ and leads to a scaled score estimate $\epsilon_\theta\left(\mathbf{z}_t, \lambda_t, \mathbf{c}\right) \approx$ $-\sigma_t \nabla_{\mathbf{z}_t} \log p\left(\mathbf{z}_t \mid \mathbf{c}\right)$, where $p\left(\mathbf{z}_t \mid \mathbf{c}\right)$ is the true density of $\mathbf{z}_t$ given $\mathbf{c}$ under the forward process starting at $\mathrm{x} \sim p(\mathrm{x})[28,35,66]$. Related model designs include the work of [70, 32, 33].
>
> To sample from the diffusion model, we start at $\mathbf{z}_1 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and use the discrete time ancestral sampler [28] and DDIM [64] for certain models. DDIM follows the deterministic update rule
> $$
> \mathbf{z}_s=\alpha_s \hat{\mathbf{x}}_\theta\left(\mathbf{z}_t, \lambda_t, \mathbf{c}\right)+\frac{\sigma_s}{\sigma_t}\left(\mathbf{z}_t-\alpha_t \hat{\mathbf{x}}_\theta\left(\mathbf{z}_t, \lambda_t, \mathbf{c}\right)\right)
> $$
> where $s<t$ follow a uniformly spaced sequence from 1 to 0 . The ancestral sampler arises from a reversed description of the forward process; noting that $q\left(\mathbf{z}_s \mid \mathbf{z}_t, \mathbf{x}\right)=\mathcal{N}\left(\mathbf{z}_s ; \tilde{\boldsymbol{\mu}}_{s \mid t}\left(\mathbf{z}_t, \mathbf{x}\right), \tilde{\sigma}_{s \mid t}^2 \mathbf{I}\right)$, where $\tilde{\boldsymbol{\mu}}_{s \mid t}\left(\mathbf{z}_t, \mathbf{x}\right)=e^{\lambda_l-\lambda_s}\left(\alpha_s / \alpha_t\right) \mathbf{z}_t+\left(1-e^{\lambda_t-\lambda_s}\right) \alpha_s \mathbf{x}$ and $\tilde{\sigma}_{s \mid t}^2=\left(1-e^{\lambda_t-\lambda_s}\right) \sigma_s^2$, it follows the stochastic update rule
> $$
> \mathbf{z}_s=\tilde{\mu}_{s \mid t}\left(\mathbf{z}_t, \hat{\mathbf{x}}_\theta\left(\mathbf{z}_t, \lambda_t, \mathbf{c}\right)\right)+\sqrt{\left(\tilde{\sigma}_{s \mid t}^2\right)^{1-\gamma}\left(\sigma_{t \mid s}^2\right)^\gamma} \boldsymbol{\epsilon}
> $$
> where $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, and $\gamma$ controls the stochasticity of the sampler [40].
>
> > 反转上述扩散过程，可以看作学习解噪 $\mathbf{z}_t \sim q\left(\mathbf{z}_t \mid \mathbf{x}\right)$ ， 即根据 $\mathbf{z}_t$ 和其他条件估计 $\mathbf{x}$ ，可 以表示成:
> > $$
> > \hat{\mathbf{x}}_\theta\left(\mathbf{z}_t, \lambda_t, \mathbf{c}\right) \approx \mathbf{x}
> > $$
> > 式中， $\mathbf{c}$ 为可选的条件信息 (比如文本embeddings 或者低分辨率图像) 。训练 $\hat{\mathbf{x}}_{\boldsymbol{\theta}}$ 使用加权 MSE 损失函数:
> > $$
> > \mathbb{E}_{\epsilon, t}\left[w\left(\lambda_t\right)\left\|\hat{\mathbf{x}}_\theta\left(\mathbf{z}_t, \lambda_t, \mathbf{c}\right)-\mathbf{x}\right\|_2^2\right]
> > $$
> > 式中， $(\mathbf{x}, \mathbf{c})$ 是数据-条件对, $t \sim \mathcal{U}([0,1]), \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I}), \alpha_t, \sigma_t, w_t$ 是关于 $t$ 的函数，其影 响样本生成质量。
> > 因为 $\mathbf{z}_t=\alpha_t \mathbf{x}+\sigma_t \epsilon$ ，对 $\epsilon$ 参数化，则 $\hat{\mathbf{x}}_\theta\left(\mathbf{z}_t, \lambda_t, \mathbf{c}\right)=\left(\mathbf{z}_t-\sigma_t \epsilon_\theta\left(\mathbf{z}_t, \lambda_t, \mathbf{c}\right)\right) / \alpha_t$ ，所以 损失函数可以简化为:
> > $$
> > \mathbb{E}_{\epsilon, t}\left[w\left(\lambda_t\right)\left\|\epsilon_\theta\left(\mathbf{z}_t, \lambda_t, \mathbf{c}\right)-\epsilon\right\|_2^2\right]
> > $$
> > 训练模型时，使用 Classifier-free guidance 方法，在单个扩散模型上同时训练无条件和带条 件目标，具体做法是在训练模型时随机（一般以 $10 \%$ 的概率) 丟弃 $\mathbf{c}$ ，得到 $\epsilon_\theta\left(\mathbf{z}_t, \lambda_t, \mathbf{c}\right)$ 之 后，使用下式更新：
> > $$
> > \tilde{\boldsymbol{\epsilon}}_\theta\left(\mathbf{z}_t, \lambda_t, \mathbf{c}\right)=w \epsilon_\theta\left(\mathbf{z}_t, \lambda_t, \mathbf{c}\right)+(1-w) \epsilon_\theta\left(\mathbf{z}_t, \lambda_t\right)
> > $$
> > 式中, $\epsilon_\theta\left(\mathbf{z}_t, \lambda_t, \mathbf{c}\right)$ 和 $\epsilon_\theta\left(\mathbf{z}_t, \lambda_t\right)$ 分贝是带条件和无条件的 $\boldsymbol{\epsilon}$ 预测值， $w$ 是指导权重，当 $w=1$ 时，抑制了 classifier-free guidance, 当 $w>1$ 会增强 guidance 的影响。
> > 进一步可以估计 score： $\tilde{\boldsymbol{\epsilon}}_\theta\left(\mathbf{z}_t, \lambda_t, \mathbf{c}\right) \approx-\sigma_t \nabla_{\mathbf{z}_t} \log p\left(\mathbf{z}_t \mid \mathbf{c}\right)$ ，其中 $p\left(\mathbf{z}_t \mid \mathbf{c}\right)$ 是以 $\mathbf{c}$ 为 条件关于 $\mathbf{z}_t$ 的概率密度。
> > 为了从扩散模型中采样，可以从随机噪声 $\mathbf{z}_1 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 开始，然后使用 DDIM 方法采样， 具体过程为:
> >
> > - step1：由模型得到 $\epsilon_\theta\left(\mathbf{z}_t, \lambda_t, \mathbf{c}\right)$ 和 $\epsilon_\theta\left(\mathbf{z}_t, \lambda_t\right)$ ，进一步得到 $\tilde{\epsilon}_\theta\left(\mathbf{z}_t, \lambda_t, \mathbf{c}\right)$;
> > - step2: 根据 $\mathbf{z}_t$ 和 $\tilde{\boldsymbol{\epsilon}}_\theta\left(\mathbf{z}_t, \lambda_t, \mathbf{c}\right)$ 得到 $\hat{\mathbf{x}}_\theta\left(\mathbf{z}_t, \lambda_t, \mathbf{c}\right)=\left(\mathbf{z}_t-\sigma_t \tilde{\epsilon}_\theta\left(\mathbf{z}_t, \lambda_t, \mathbf{c}\right)\right) / \alpha_t$ ；
> > - step3: 估计 $\mathbf{z}_s=\alpha_s \hat{\mathbf{x}}_\theta\left(\mathbf{z}_t, \lambda_t, \mathbf{c}\right)+\sigma_s \tilde{\epsilon_\theta}\left(\mathbf{z}_t, \lambda_t, \mathbf{c}\right)$
> >
> > $$
> > \mathbf{z}_s=\alpha_s \hat{\mathbf{x}}_\theta\left(\mathbf{z}_t, \lambda_t, \mathbf{c}\right)+\frac{\sigma_s}{\sigma_t}\left(\mathbf{z}_t-\alpha_t \hat{\mathbf{x}}_\theta\left(\mathbf{z}_t, \lambda_t, \mathbf{c}\right)\right)
> > $$
> >
> > 式中， $s<t$ 在 1 到 0 之间取均匀分布序列，最先的采样来自对扩散过程的反转，注意到:
> > $$
> > q\left(\mathbf{z}_s \mid \mathbf{z}_t, \mathbf{x}\right)=\mathcal{N}\left(\mathbf{z}_s ; \tilde{\boldsymbol{\mu}}_{s \mid t}\left(\mathbf{z}_t, \mathbf{x}\right), \tilde{\sigma}_{s \mid t}^2 \mathbf{I}\right)
> > $$
> > 式 中， $\tilde{\boldsymbol{\mu}}_{s \mid t}\left(\mathbf{z}_t, \mathbf{x}\right)=e^{\lambda_t-\lambda_s}\left(\alpha_s / \alpha_t\right) \mathbf{z}_t+\left(1-e^{\lambda_t-\lambda_s}\right) \alpha_s \mathbf{X}$ ，并 且 $\tilde{\sigma}_{s \mid t}^2=\left(1-e^{\lambda_t-\lambda_s}\right) \sigma_s^2$, 遵循随机更新规则:
> > $$
> > \mathbf{z}_s=\tilde{\boldsymbol{\mu}}_{s \mid t}\left(\mathbf{z}_t, \hat{\mathbf{x}}_\theta\left(\mathbf{z}_t, \lambda_t, \mathbf{c}\right)\right)+\sqrt{\left(\tilde{\sigma}_{s \mid t}^2\right)^{1-\gamma}\left(\sigma_{t \mid s}^2\right)^\gamma \boldsymbol{\epsilon}}
> > $$
> > 式中， $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, 其中 $\gamma$ 控制采样的随机性。

Latent Diffusion Models can be divided into two stages:

1. Training perceptual compression models that strip away irrelevant high-level details and learn a latent space that is semantically equivalent to the high level image pixel-space
a. The loss is a combination of a reconstruction loss, an adversarial loss (remember GANs?) that promotes high quality decoder reconstruction, and regularization terms
$$
L_{\text {Autoencoder }}=\min _{\mathcal{E}, \mathcal{D}} \max _\psi\left(L_{r e c}(x, \mathcal{D}(\mathcal{E}(x)))-L_{a d v}(\mathcal{D}(\mathcal{E}(x)))+\log D_\psi(x)+L_{r e g}(x ; \mathcal{E}, \mathcal{D})\right)
$$
2. Performing a diffusion process in this latent space. There are several benefits to this:
    a. The diffusion process is only focusing on the relevant semantic bits of the data
    b. Performing diffusion in a low dimensional space is significantly more efficient

> 整体框架如图，先训练好一个AutoEncoder（包括一个encoder和decoder）。因此，我们可以利用encoder压缩后的数据做diffusion操作，再用decoder恢复即可。
>
> - Autoencoder训练：L1/L2loss 来作为重建损失，用 $G A N$ 来做对抗攻击，用KL loss来把 latent space拉到正态分布，防止搜索空间过大。
> - 用了encoder降维后，就可以使用latent space diffusion了 具体扩散过程其实没有变，只不 过现在扩散和重建的目标为latent space的向量了。Diffusion model具体实现为 timeconditional UNet。
>
> $$
> L_{L D M}:=\mathbb{E}_{\mathcal{E}(x), \epsilon \sim \mathcal{N}(0,1), t}\left[\left\|\epsilon-\epsilon_\theta\left(z_t, t\right)\right\|_2^2\right]
> $$
>
> ​		The neural backbone $\epsilon_\theta(\circ, t)$ of our model is realized as a time-conditional UNet .
>
> - 为了引入conditioning的信息，提出了domain specific encoder $\tau_\theta(y)$ 不同模态的 (比如 text, class, image...) 转成中间表达(intermediate representation)，再利用cross-attention来 嵌入到UNet中去。
>   Attention $(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) \cdot V$, with
>
> $$
> Q=W_Q^{(i)} \cdot \varphi_i\left(z_t\right), K=W_K^{(i)} \cdot \tau_\theta(y), V=W_V^{(i)} \cdot \tau_\theta(y) .
> $$

<img src="/assets/BERTGPTDiffusion%20Research.assets/image-20230517130729142.png" alt="image-20230517130729142" style="zoom:50%;" />

### Architecture Comparison

#### Pure Diffusion Model

<img src="/assets/BERTGPTDiffusion%20Research.assets/1PICHZIwm-SzP0BITiN5-3g.png" alt="Pure diffusion model architecture" style="zoom:80%;" />

#### Stable Diffusion (Latent Diffusion Model)

<img src="/assets/BERTGPTDiffusion%20Research.assets/1NpQ282NJdOfxUsYlwLJplA.png" alt="Stable Diffusion architecture" style="zoom:80%;" />

## 4.11 Pytorch implementation 

https://github.com/azad-academy/denoising-diffusion-model

https://zhuanlan.zhihu.com/p/549623622

### 扩散过程

code for alphas_bar_sqrt: $\sqrt{\bar{\alpha}}$ , one_minus_alphas_bar_log: $log(1 - \bar{\alpha})$ , one_minus_alphas_bar_sqrt : $\sqrt{1 - \bar{\alpha}}$

```python
num_steps = 1000
beta = torch.tensor(np.linspace(1e-5, 0.2e-2, num_steps))

alphas = 1 - betas   # \alpha_t & =1-\beta_t 
alphas_prod = torch.cumprod(alphas, 0)   #\bar{\alpha}_t & =\prod_{i=1}^t \alpha_i
alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
```

code for $q(\mathbf{x}_{1:T} \vert \mathbf{x}_0)$ and$\sqrt{\bar{\alpha}_t}$,  $\mathbf{x}_0+\sqrt{1-\bar{\alpha}_t}$ , $\sqrt{\bar{\alpha}_t} \mathbf{x}_0+\sqrt{1-\bar{\alpha}_t} \epsilon$

```python
def extract(input, t, x):
	shape = x.shape
	out = torch.gather(input, 0, t.to(input.device))
	reshape = [t.shape[0]] + [1] * (len(shape) - 1)
	return out.reshape(*reshape)

def q_x(x_0, t, noise=None):
    if not noise: noise = torch.randn_like(x_0)
    alphas_t = extract(alphas_bar_sqrt, t, x_0)
    alphas_1_m_t = extract(one_minus_alphas_bar_sqrt, t, x_0)
    return (alphas_t * x_0 + alphas_1_m_t * noise)
```

正向过程正式的计算比较简单直接，正如上面理论部分提到的，通过时间步 T 在每次马尔科夫链的转换过程对样本数据 dataset 添加噪声：

```python
for i in range(num_steps):
    q_i = q_x(dataset, torch.tensor([i]))
```

$q(\mathbf{x}_t \vert \mathbf{x}_0)$ , extract(posterior_mean_coef_1, t, x_0) = $\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t}$, extract(posterior_mean_coef_2, t, x_0) = $\frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}$

mean: $\tilde{\boldsymbol{\mu}}_t (\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0$ , 

posterior_variance: $\tilde{\beta}_t = 1/(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}) 
= 1/(\frac{\alpha_t - \bar{\alpha}_t + \beta_t}{\beta_t(1 - \bar{\alpha}_{t-1})})
= \color{green}{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t} $ and var/posterior_log_variance_clipped: $\Sigma_\theta\left(x_t, t\right)=\exp \left(v \log \beta_t+(1-v) \log \tilde{\beta}_t\right)$

```python
posterior_mean_coef_1 = (betas * torch.sqrt(alphas_prod_p) / (1 - alphas_prod))
posterior_mean_coef_2 = ((1 - alphas_prod_p) * torch.sqrt(alphas) / (1 - alphas_prod))
posterior_variance = betas * (1 - alphas_prod_p) / (1 - alphas_prod)
posterior_log_variance_clipped = torch.log(torch.cat((posterior_variance[1].view(1, 1), posterior_variance[1:].view(-1, 1)), 0)).view(-1)

def q_posterior_mean_variance(x_0, x_t, t):
    coef_1 = extract(posterior_mean_coef_1, t, x_0)
    coef_2 = extract(posterior_mean_coef_2, t, x_0)
    mean = coef_1 * x_0 + coef_2 * x_t
    var = extract(posterior_log_variance_clipped, t, x_0)
    return mean, var
```

### 训练过程 - 逆扩散过程需要训练神经网络模型, 

训练数据batch_x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt， 其中batch_x 是X_0数据， alphas_bar_sqrt: $\sqrt{\bar{\alpha}}$ , one_minus_alphas_bar_sqrt : $\sqrt{1 - \bar{\alpha}}$， 都是一开始就得到的数据，没有用到q_x结果？

```python
from model import Unet
from ema import EMA
import torch.optim as optim


model = Unet()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Create EMA model
ema = EMA(0.9)
ema.register(model)

# Batch size
batch_size = 128
for t in range(num_steps):
    # X is a torch Variable
    permutation = torch.randperm(dataset.size()[0])
    for i in range(0, dataset.size()[0], batch_size):
        # Retrieve current batch
        indices = permutation[i:i+batch_size]
        batch_x = dataset[indices]
        # Compute the loss.
        loss = noise_estimation_loss(model, batch_x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)
        # Before the backward pass, zero all of the network gradients
        optimizer.zero_grad()
        # Backward pass: compute gradient of the loss with respect to parameters
        loss.backward()
        # Perform gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        # Calling the step function to update the parameters
        optimizer.step()
        # Update the exponential moving average
        ema.update(model)

    # Print loss
    if (t % 100 == 0):
        print(loss)
```



损失函数, alphas_bar_sqrt: $\sqrt{\bar{\alpha}}$ , one_minus_alphas_bar_sqrt : $\sqrt{1 - \bar{\alpha}}$， $\mathbb{E}_{t \sim [1, T], \mathbf{x}_0, \boldsymbol{\epsilon}_t} \Big[\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta( x_t, t)\|^2 \Big]$. 其中$x_0$， $\epsilon _t$从前向过程已知, we already know $x_t=\sqrt{\bar{\alpha}_t} x_0+\sqrt{1-\bar{\alpha}_t} \epsilon_t$ 

<img src="/assets/BERTGPTDiffusion%20Research.assets/image-20230605160414054.png" alt="image-20230605160414054" style="zoom:50%;" />

```python
def noise_estimation_loss(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    batch_size = x_0.shape[0]
    # Select a random step for each example
    t = torch.randint(0, n_steps, size=(batch_size // 2 + 1,))
    t = torch.cat([t, n_steps - t - 1], dim=0)[:batch_size].long()
    # x0 multiplier，这里用到了alphas_bar_sqrt
    a = extract(alphas_bar_sqrt, t, x_0)
    # eps multiplier, 这里用到了one_minus_alphas_bar_sqrt
    am1 = extract(one_minus_alphas_bar_sqrt, t, x_0)
	# get the noise epsilon sampling from Normal Dist, just random same size with x_0 and follow N(0,1)
    e = torch.randn_like(x_0)
    # model input x_t 
    x = x_0 * a + e * am1 
    # get the prodict noise epsilon form model 
    output = model(x, t)
    return (e - output).square().mean() # mean squre of the loss (e-\hat{e})
```

### inference (sampling)


$$
\begin{aligned}
& \text { Algorithm } 2 \text { Sampling } \\
& \text { 1: } \mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\
& \text { 2: for } t=T, \ldots, 1 \text { do } \\
& \text { 3: } \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \text { if } t>1, \text { else } \mathbf{z}=\mathbf{0} \\
& \text { 4: } \quad \mathbf{x}_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t, t\right)\right)+\sigma_t \mathbf{z} \\

&\text{; where we have } \mathbf{x}_{t-1} = \mathcal{N}(\mathbf{x}_{t-1}; \frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \Big), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)) \\

& \text { 5: end for } \\
& \text { 6: return } \mathbf{x}_0
\end{aligned}
$$

```python
x_seq = p_sample_loop(model, dataset.shape,num_steps,alphas,betas,one_minus_alphas_bar_sqrt)
```

$ \text{eps\_factor} = \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}$  , eps_theta = $\boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t, t\right)$, mean=$=\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t, t\right)\right)$

```python
def p_sample(model, x, t,alphas,betas,one_minus_alphas_bar_sqrt):
    t = torch.tensor([t])
    # Factor to the model output
    eps_factor = ((1 - extract(alphas, t, x)) / extract(one_minus_alphas_bar_sqrt, t, x))
    # Model output
    eps_theta = model(x, t)
    # Final values
    mean = (1 / extract(alphas, t, x).sqrt()) * (x - (eps_factor * eps_theta))
    # Generate z
    z = torch.randn_like(x)
    # Fixed sigma
    sigma_t = extract(betas, t, x).sqrt()
    sample = mean + sigma_t * z
    return (sample)

def p_sample_loop(model, shape,n_steps,alphas,betas,one_minus_alphas_bar_sqrt):
    cur_x = torch.randn(shape)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i,alphas,betas,one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
    return x_seq
```

## 4.12 Connection with noise-conditioned score networks (NCSN)- DDPM

[Song & Ermon (2019)](https://arxiv.org/abs/1907.05600) proposed a score-based generative modeling method where samples are produced via [Langevin dynamics](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#connection-with-stochastic-gradient-langevin-dynamics) **using gradients of the data distribution estimated with score matching**. The score of each sample $\mathbf{x}$’s density probability is defined as its gradient $\nabla_{\mathbf{x}} \log q(\mathbf{x})$. A score network $\mathbf{s}_\theta: \mathbb{R}^D \to \mathbb{R}^D$ is trained to estimate it, $\mathbf{s}_\theta(\mathbf{x}) \approx \nabla_{\mathbf{x}} \log q(\mathbf{x})$

> **TBS:**
>
> To make it scalable with high-dimensional data in the deep learning setting, they proposed to use either *denoising score matching* ([Vincent, 2011](http://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf)) or *sliced score matching* (use random projections; [Song et al., 2019](https://arxiv.org/abs/1905.07088)). Denosing score matching adds a pre-specified small noise to the data $q(\tilde{\mathbf{x}} \vert \mathbf{x})$ and estimates $q(\tilde{\mathbf{x}})$ with score matching.

> Recall that Langevin dynamics can sample data points from a probability density distribution using only the score $\nabla_{\mathbf{x}} \log q(\mathbf{x})$ in an iterative process.

> However, according to the manifold hypothesis, most of the data is expected to concentrate in a low dimensional manifold, even though the observed data might look only arbitrarily high-dimensional. It brings a negative effect on score estimation since the data points cannot cover the whole space. In regions where data density is low, the score estimation is less reliable. After adding a small Gaussian noise to make the perturbed data distribution cover the full space $\mathbb{R}^D$, the training of the score estimator network becomes more stable. Song \& Ermon (2019) improved it by perturbing the data with the noise of different levels and train a noise-conditioned score network to jointly estimate the scores of all the perturbed data at different noise levels.

The schedule of increasing noise levels resembles the forward diffusion process. 

<font color=red>If we use the diffusion process annotation, the score approximates $\mathbf{s}_\theta\left(\mathbf{x}_t, t\right) \approx \nabla_{\mathbf{x}_t} \log q\left(\mathbf{x}_t\right)$. Given a Gaussian distribution $\mathbf{x} \sim \mathcal{N}\left(\mu, \sigma^2 \mathbf{I}\right)$, we can write the derivative of the logarithm of its density function as $\nabla_{\mathbf{x}} \log p(\mathbf{x})=\nabla_{\mathbf{x}}\left(-\frac{1}{2 \sigma^2}(\mathbf{x}-\boldsymbol{\mu})^2\right)=-\frac{\mathbf{x}-\boldsymbol{\mu}}{\sigma^2}=-\frac{\epsilon}{\sigma}$ where $\boldsymbol{\epsilon} \mathcal{N}(\mathbf{0}, \mathbf{I})$. Recall that $q\left(\mathbf{x}_t \mid \mathbf{x}_0\right) \sim \mathcal{N}\left(\sqrt{\bar{\alpha}_t} \mathbf{x}_0,\left(1-\bar{\alpha}_t\right) \mathbf{I}\right)$ and therefore： </font>
$$
\mathbf{s}_\theta\left(\mathbf{x}_t, t\right) \approx \nabla_{\mathbf{x}_t} \log q\left(\mathbf{x}_t\right)=\mathbb{E}_{q\left(\mathbf{x}_0\right)}\left[\nabla_{\mathbf{x}_t} q\left(\mathbf{x}_t \mid \mathbf{x}_0\right)\right]=\mathbb{E}_{q\left(\mathbf{x}_0\right)}\left[-\frac{\epsilon_\theta\left(\mathbf{x}_t, t\right)}{\sqrt{1-\bar{\alpha}_t}}\right]=-\frac{\epsilon_\theta\left(\mathbf{x}_t, t\right)}{\sqrt{1-\bar{\alpha}_t}}
$$

## 4.13 Conditioned Generation (条件控制生成)

从方法上来看，条件控制生成的方式分两种：事后修改（Classifier-Guidance）和事前训练（Classifier-Free）。

> [生成扩散模型漫谈（九）：条件控制生成结果 - 科学空间|Scientific Spaces (kexue.fm)](https://kexue.fm/archives/9257)
>
> Classifier-Guidance方案最早出自[《Diffusion Models Beat GANs on Image Synthesis》](https://arxiv.org/abs/2105.05233)，最初就是用来实现按类生成的；后来[《More Control for Free! Image Synthesis with Semantic Diffusion Guidance》](https://arxiv.org/abs/2112.05744)推广了“Classifier”的概念，使得它也可以按图、按文来生成。
>
> Classifier-Guidance方案的训练成本比较低（熟悉NLP的读者可能还会想起与之很相似的[PPLM模型](https://arxiv.org/abs/1912.02164)），但是推断成本会高些，而且控制细节上通常没那么到位。至于Classifier-Free方案，最早出自[《Classifier-Free Diffusion Guidance》](https://arxiv.org/abs/2207.12598)，后来的[DALL·E 2](https://arxiv.org/abs/2204.06125)、[Imagen](https://arxiv.org/abs/2205.11487)等吸引人眼球的模型基本上都是以它为基础做的，值得一提的是，该论文上个月才放到Arxiv上，但事实上去年已经中了NeurIPS 2021。应该说，Classifier-Free方案本身没什么理论上的技巧，它是条件扩散模型最朴素的方案，出现得晚只是因为重新训练扩散模型的成本较大吧，在数据和算力都比较充裕的前提下，Classifier-Free方案变现出了令人惊叹的细节控制能力。

### Classifier Guided Diffusion 

用随机微分方程解释DDPM and DDIM 模型

#### Diffusion guidance Explanation 1 （==Suitable for Algorithm 2==）

[What are Diffusion Models? | Lil'Log (lilianweng.github.io)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#nice)

Guidance is a technique to explicitly incorporate image class—or text prompt—directly in the diffusion process. (This is the often tweaked [`guidance_scale`](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion#diffusers.StableDiffusionPipeline.__call__.guidance_scale) hyperpameter.)

**The [classifier-guidance paper (2021) Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) noted that GANs relied heavily on class labels**, often via **class-conditioned normalization or discriminators** with heads designed to behave like classifiers. This suggests that class information is crucial to the success of GANs for generation. So, to take a leaf from GANs, **they use a classifier $p_\phi(y\vert x)$** to improve image generation via diffusion.

To explicit incorporate class information into the diffusion process, Dhariwal \& Nichol (2021) trained a classifier $f_\phi\left(y \mid \mathbf{x}_t, t\right)$ on noisy image $\mathbf{x}_t$ and use gradients $\nabla_{\mathbf{x}} \log f_\phi\left(y \mid \mathbf{x}_t\right)$ to guide the diffusion sampling process toward the conditioning information $y$ (e.g. a target class label) by altering the noise prediction. **Recall that $\nabla_{\mathbf{x}_t} \log q\left(\mathbf{x}_t\right)=-\frac{1}{\sqrt{1-\bar{\alpha} t}} \boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t, t\right)$** and we can write the score function for the joint distribution $q\left(\mathbf{x}_t, y\right)$ as following,
$$
\begin{aligned}
\nabla_{\mathbf{x}_t} \log q\left(\mathbf{x}_t, y\right) & =\nabla_{\mathbf{x}_t} \log q\left(\mathbf{x}_t\right)+\nabla_{\mathbf{x}_t} \log q\left(y \mid \mathbf{x}_t\right) \\
& \approx-\frac{1}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta\left(\mathbf{x}_t, t\right)+\nabla_{\mathbf{x}_t} \log f_\phi\left(y \mid \mathbf{x}_t\right) \text{； 第二项目是为了分类增加额外的项目}\\
& =-\frac{1}{\sqrt{1-\bar{\alpha}_t}}\left(\epsilon_\theta\left(\mathbf{x}_t, t\right)-\sqrt{1-\bar{\alpha}_t} \nabla_{\mathbf{x}_t} \log f_\phi\left(y \mid \mathbf{x}_t\right)\right)
\end{aligned}
$$
Thus, a new classifier-guided predictor $\overline{\boldsymbol{\epsilon}}_\theta$ would take the form as following,
$$
\bar{\epsilon}_\theta\left(\mathbf{x}_t, t\right)=\epsilon_\theta\left(x_t, t\right)-\sqrt{1-\bar{\alpha}_t} \nabla_{\mathbf{x}_t} \log f_\phi\left(y \mid \mathbf{x}_t\right) \tag {14- Conditional Sampling for DDIM}
$$
因为我们已知微分方程 $\begin{aligned}
& X_{t+1}=X_t+\frac{\varepsilon}{2} \nabla \log p\left(X_t\right)+\mathcal{N}(0, \varepsilon) \\
& X_t \sim p(x), \quad \forall t>t_{\infty}
\end{aligned}$  

We can then use the exact same sampling procedure as used for regular DDIM, but with the modified noise predictions $\hat{\epsilon}_\theta\left(x_t\right)$ instead of $\epsilon_\theta\left(x_t\right)$. Algorithm \& summaries the corresponding sampling algorithm.

To control the strength of the classifier guidance, we can add a weight $w$ to the delta part,
$$
\bar{\epsilon}_\theta\left(\mathbf{x}_t, t\right)=\epsilon_\theta\left(x_t, t\right)-\sqrt{1-\bar{\alpha}_t} w \nabla_{\mathbf{x}_t} \log f_\phi\left(y \mid \mathbf{x}_t\right)
$$
The resulting ablated diffusion model (ADM) and the one with additional classifier guidance (ADM-G) are able to achieve better results than SOTA generative models (e.g. BigGAN).

<img src="/assets/BERTGPTDiffusion%20Research.assets/conditioned-DDPM.png" alt="img" style="zoom: 33%;" />

#### Diffusion guidance Explanation 2

[生成扩散模型漫谈（九）：条件控制生成结果 - 科学空间|Scientific Spaces (kexue.fm)](https://kexue.fm/archives/9257)

当给定训练好的Diffusion Models，如DDPM所述其逆扩散过程可以描述为 $p_\theta\left(\mathbf{x}_t \mid \mathbf{x}_{t+1}\right)$ 。假如此 时我们要求逆扩散的图像必须属于某种类型 $\mathbf{y}$ ，那么逆扩散过程就应该被重新定义为 $p_\theta\left(\mathbf{x}_t \mid \mathbf{x}_{t+1}, \mathbf{y}\right)$ 
为了获取类型 $\mathbf{y}$ ，我们需要一个训练好的分类器 $p_\phi\left(\mathbf{y} \mid \mathbf{x}_t, t\right)$ ，这个分类器跟普通分类器的区别 是，必须要见过加噪图像 $\mathbf{x}_t$ ，因此重新训练是不可避免的。此时逆扩散过程就变成了 $p_{\theta, \phi}\left(\mathbf{x}_t \mid \mathbf{x}_{t+1}, \mathbf{y}\right)$ 。其可以化简为
$$
\begin{aligned}
p_{\theta, \phi}\left(\mathbf{x}_t \mid \mathbf{x}_{t+1}, \mathbf{y}\right) & =\ldots \\
& =Z p_\theta\left(\mathbf{x}_t \mid \mathbf{x}_{t+1}\right) p_\phi\left(\mathbf{y} \mid \mathbf{x}_t\right) \\
& =\ldots
\end{aligned}
$$
$$
=p(\mathbf{z}) \quad ; \text { where } \mathbf{z} \sim \mathcal{N}(\mu+\Sigma g, \Sigma)
$$
其中， $Z$ 是一个概率密度归一化的常数， $g=\left.\nabla_{x_t} \log p_\phi\left(y \mid x_t\right)\right|_{x_t=\mu}$ 。因此，Classifier Guided Sampling 的过程可以总结如下图

>（论文）In this section, we show that conditional sampling can be achieved with a transition operator proportional to $p_\theta\left(x_t \mid x_{t+1}\right) p_\phi\left(y \mid x_t\right)$, where $p_\theta\left(x_t \mid x_{t+1}\right)$ approximates $q\left(x_t \mid x_{t+1}\right)$ and $p_\phi\left(y \mid x_t\right)$ approximates the label distribution for a noised sample $x_t$.
>
>We start by defining a conditional Markovian noising process $\hat{q}$ similar to $q$, and **assume that $\hat{q}\left(y \mid x_0\right)$ is a known** (==data likelihood==) and readily available label distribution for each sample.
>$$
>\begin{aligned}
>\hat{q}\left(x_0\right) & :=q\left(x_0\right) \\
>\hat{q}\left(y \mid x_0\right) & :=\text { Known labels per sample } \\
>\hat{q}\left(x_{t+1} \mid x_t, y\right) & :=q\left(x_{t+1} \mid x_t\right) \\
>&\text{; 什么意思，加不加标签的正向过程没有变化么？应该是可以肯定的}\\
>\hat{q}\left(x_{1: T} \mid x_0, y\right) & :=\prod_{t=1}^T \hat{q}\left(x_t \mid x_{t-1}, y\right)
>\end{aligned}
>$$
>While we defined the noising process $\hat{q}$ conditioned on $y$, we can prove that $\hat{q}$ behaves exactly like $q$ **when not conditioned on $y$**. Along these lines, we first derive the unconditional noising operator $\hat{q}\left(x_{t+1} \mid x_t\right)$ :
>$$
>\begin{aligned}
>\hat{q}\left(x_{t+1} \mid x_t\right) & =\int_y \hat{q}\left(x_{t+1}, y \mid x_t\right) d y \\
>& =\int_y \hat{q}\left(x_{t+1} \mid x_t, y\right) \hat{q}\left(y \mid x_t\right) d y \\
>& =\int_y q\left(x_{t+1} \mid x_t\right) \hat{q}\left(y \mid x_t\right) d y \\
>& =q\left(x_{t+1} \mid x_t\right) \int_y \hat{q}\left(y \mid x_t\right) d y \\
>& =q\left(x_{t+1} \mid x_t\right) \\
>& =\hat{q}\left(x_{t+1} \mid x_t, y\right)
>\end{aligned}
>$$
>Following similar logic, we find the joint distribution $\hat{q}\left(x_{1: T} \mid x_0\right)$ :
>$$
>\begin{eqnarray}
>\hat{q}\left(x_{1: T} \mid x_0\right) && =\int_y \hat{q}\left(x_{1: T}, y \mid x_0\right) d y \\
>&& =\int_y \hat{q}\left(y \mid x_0\right) \hat{q}\left(x_{1: T} \mid x_0, y\right) d y \\
>&& =\int_y \hat{q}\left(y \mid x_0\right) \prod_{t=1}^T \hat{q}\left(x_t \mid x_{t-1}, y\right) d y \\
>&& =\int_y^T \hat{q}\left(y \mid x_0\right) \prod_{t=1}^T q\left(x_t \mid x_{t-1}\right) d y \\
>&& =\prod_{t=1}^T q\left(x_t \mid x_{t-1}\right) \int_y \hat{q}\left(y \mid x_0\right) d y \\
>&& =\prod_{t=1}^T q\left(x_t \mid x_{t-1}\right) \\
>&& =q\left(x_{1: T} \mid x_0\right)  \tag {44 - Conditional Sampling} 
>\end{eqnarray}
>$$
>Using Equation 44, we can now derive $\hat{q}\left(x_t\right)$ :
>$$
>\begin{aligned}
>\hat{q}\left(x_t\right) & =\int_{x_{0: t-1}} \hat{q}\left(x_0, \ldots, x_t\right) d x_{0: t-1} \\
>& =\int_{x_{0: t-1}} \hat{q}\left(x_0\right) \hat{q}\left(x_1, \ldots, x_t \mid x_0\right) d x_{0: t-1} \\
>& =\int_{x_{0: t-1}} q\left(x_0\right) q\left(x_1, \ldots, x_t \mid x_0\right) d x_{0: t-1} \\
>& =\int_{x_{0: t-1}} q\left(x_0, \ldots, x_t\right) d x_{0: t-1} \\
>& =q\left(x_t\right)
>\end{aligned}
>$$
>Using the identities $\hat{q}\left(x_t\right)=q\left(x_t\right)$ and $\hat{q}\left(x_{t+1} \mid x_t\right)=q\left(x_{t+1} \mid x_t\right)$, it is trivial to show via Bayes rule that the **unconditional reverse process $\hat{q}\left(x_t \mid x_{t+1}\right)=q\left(x_t \mid x_{t+1}\right)$.**
>
>One observation about $\hat{q}$ is that it gives rise to a noisy classification function, ==$\hat{q}\left(y \mid x_t\right)$. We can show that this classification distribution does not depend on $x_{t+1}$ (a noisier version of $x_t$ )==, a fact which we will later use:
>$$
>\begin{aligned}
>\hat{q}\left(y \mid x_t, x_{t+1}\right) & =\hat{q}\left(x_{t+1} \mid x_t, y\right) \frac{\hat{q}\left(y \mid x_t\right)}{\hat{q}\left(x_{t+1} \mid x_t\right)} \text{; where the } q(x_t) \text{ is removed both molecular and denominator } \\
>& =\hat{q}\left(x_{t+1} \mid x_t\right) \frac{\hat{q}\left(y \mid x_t\right)}{\hat{q}\left(x_{t+1} \mid x_t\right)} \\
>& =\hat{q}\left(y \mid x_t\right)
>\end{aligned}
>$$
>We can now derive the conditional reverse process:
>$$
>\begin{aligned}
>\hat{q}\left(x_t \mid x_{t+1}, y\right) & =\frac{\hat{q}\left(x_t, x_{t+1}, y\right)}{\hat{q}\left(x_{t+1}, y\right)} \\
>& =\frac{\hat{q}\left(x_t, x_{t+1}, y\right)}{\hat{q}\left(y \mid x_{t+1}\right) \hat{q}\left(x_{t+1}\right)} \\
>& =\frac{\hat{q}\left(x_t \mid x_{t+1}\right) \hat{q}\left(y \mid x_t, x_{t+1}\right) \hat{q}\left(x_{t+1}\right)}{\hat{q}\left(y \mid x_{t+1}\right) \hat{q}\left(x_{t+1}\right)} \\
>& =\frac{\hat{q}\left(x_t \mid x_{t+1}\right) \hat{q}\left(y \mid x_t, x_{t+1}\right)}{\hat{q}\left(y \mid x_{t+1}\right)} \\
>& =\frac{\hat{q}\left(x_t \mid x_{t+1}\right) \hat{q}\left(y \mid x_t\right)}{\hat{q}\left(y \mid x_{t+1}\right)} \\
>& =\frac{q\left(x_t \mid x_{t+1}\right) \hat{q}\left(y \mid x_t\right)}{\hat{q}\left(y \mid x_{t+1}\right)} \\
>& = Z q\left(x_t \mid x_{t+1}\right) \hat{q}\left(y \mid x_t\right)
>\end{aligned}
>$$
>
>- ==The $\hat{q}\left(y \mid x_{t+1}\right)$ term can be treated as a constant since it does not depend on $x_t$==. We thus want to sample from the distribution $Z q\left(x_t \mid x_{t+1}\right) \hat{q}\left(y \mid x_t\right)$ where $Z$ is a normalizing constant. 
>
>- ==We already have a neural network approximation of $q\left(x_t \mid x_{t+1}\right)$, called $p_\theta\left(x_t \mid x_{t+1}\right)$, so all that is left is an approximation of $\hat{q}\left(y \mid x_t\right)$.==
>
>- ==This can be obtained by training a classifier $p_\phi\left(y \mid x_t\right)$ on noised images $x_t$ derived by sampling from $q\left(x_t\right)$.==
>
>$$
>p_{\theta, \phi}\left(x_t \mid x_{t+1}, y\right)=Z p_\theta\left(x_t \mid x_{t+1}\right) p_\phi\left(y \mid x_t\right) \\
>\tag {2 - Conditional Sampling for DDIM}
>$$
>
>



重用已经训练好的无条件生成模型$p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)$，我们利用贝叶斯定理得
$$
\begin{equation}p(\boldsymbol{x}_{t-1}|\boldsymbol{y}) = \frac{p(\boldsymbol{x}_{t-1})p(\boldsymbol{y}|\boldsymbol{x}_{t-1})}{p(\boldsymbol{y})}\end{equation}
$$
在每一项上面补上条件$x_t$，就得到
$$
\begin{equation}p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{y}) = \frac{p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)p(\boldsymbol{y}|\boldsymbol{x}_{t-1}, \boldsymbol{x}_t)}{p(\boldsymbol{y}|\boldsymbol{x}_t)}\label{eq:bayes-1}\end{equation}
$$
注意，在前向过程中，$x_t$是由$x_{t−1}$加噪声得到的，噪声不会对分类有帮助，所以$x_t$的加入对分类不会有任何收益，因此有$p(\boldsymbol{y}|\boldsymbol{x}_{t-1}, \boldsymbol{x}_t)=p(\boldsymbol{y}|\boldsymbol{x}_{t-1})$，从而
$$
\begin{equation}p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{y}) = \frac{p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)p(\boldsymbol{y}|\boldsymbol{x}_{t-1})}{p(\boldsymbol{y}|\boldsymbol{x}_t)} = p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t) e^{\log p(\boldsymbol{y}|\boldsymbol{x}_{t-1}) - \log p(\boldsymbol{y}|\boldsymbol{x}_t)}\label{eq:bayes-2}\end{equation}
$$
==到此我们都得到了==
$$
\begin{aligned}
p_{\theta, \phi}\left(\mathbf{x}_t \mid \mathbf{x}_{t+1}, \mathbf{y}\right) 
& =Z p_\theta\left(\mathbf{x}_t \mid \mathbf{x}_{t+1}\right) p_\phi\left(\mathbf{y} \mid \mathbf{x}_t\right) 
\end{aligned}
$$
以下是不同的泰勒展开不同的推导

##### 1, 泰勒展开1

> ==**当 $T$ 足够大时， $p\left(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1}\right)$ 的方差足够小，也就是说只有 $\boldsymbol{x}_t$ 与 $\boldsymbol{x}_{t-1}$ 很接近时概率才会明显大于 0 。反过来也是成立的，即也只有 $\boldsymbol{x}_t$ 与 $\boldsymbol{x}_{t-1}$ 很接近时 $p\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{y}\right)$ 或 $p\left(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1}, \boldsymbol{y}\right)$ 才明显大于o**，我们只需要重点考虑这个范围内的概率变化。为此，我们用泰勒展 开:==
> $$
> \log p\left(\boldsymbol{y} \mid \boldsymbol{x}_{t-1}\right)-\log p\left(\boldsymbol{y} \mid \boldsymbol{x}_t\right) \approx\left(\boldsymbol{x}_{t-1}-\boldsymbol{x}_t\right) \cdot \nabla_{\boldsymbol{x} t} \log p\left(\boldsymbol{y} \mid \boldsymbol{x}_t\right)
> $$
> 严格来讲还有一项关于 $t$ 的变化项，但是那一项跟 $x_{t-1}$ 无关，属于不影响 $x_{t-1}$ 概率的常数项，因此我们没有写出。假设原来有 $p\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t\right)=\mathcal{N}\left(\boldsymbol{x}_{t-1} ; \boldsymbol{\mu}\left(\boldsymbol{x}_t\right), \sigma_t^2 \boldsymbol{I}\right) \propto e^{-\left\|\boldsymbol{x}_{t-1}-\boldsymbol{\mu}\left(\boldsymbol{x}_t\right)\right\|^2 / 2 \sigma_t^2}$ ，那么此时近似地有
> $$
> \begin{aligned}
> p\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{y}\right) & \propto e^{-\left\|\boldsymbol{x}_{t-1}-\boldsymbol{\mu}\left(\boldsymbol{x}_t\right)\right\|^2 / 2 \sigma^2+\left(\boldsymbol{x}_{t-1}-\boldsymbol{x}_t\right) \cdot \nabla_{\boldsymbol{x}_t} \log p\left(\boldsymbol{y} \mid \boldsymbol{x}_t\right)} \\
> & \propto e^{\left.-\| \boldsymbol{x}_{t-1}-\boldsymbol{\mu}\left(\boldsymbol{x}_t\right)- \boldsymbol{\sigma}_t^2 \nabla_{\boldsymbol{x}_t} \log p\left(\boldsymbol{y} \mid \boldsymbol{x}_t\right)\right) \|^2 / 2 \sigma_t^2}
> \end{aligned}
> $$
>
> > <font color=red> $p\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t\right)$难道不用DDIM or DDPM推导出的结果么。。。。</font> ==不需要这么细的推导，因为只要拿到新增项，就可以复用DDIM or DDPM的训练和sampling过程==
> > $$
> > \color{red} 
> > \text {DENOISING DIFFUSION IMPLICIT MODELS }\\
> > \boldsymbol{x}_{t-1}=\sqrt{\alpha_{t-1}} \underbrace{\left(\frac{\boldsymbol{x}_t-\sqrt{1-\alpha_t} \epsilon_\theta^{(t)}\left(\boldsymbol{x}_t\right)}{\sqrt{\alpha_t}}\right)}_{\text {"predicted } \boldsymbol{x}_0 \text { " }}+\underbrace{\sqrt{1-\alpha_{t-1}-\sigma_t^2} \cdot \epsilon_\theta^{(t)}\left(\boldsymbol{x}_t\right)}_{\text {“direction pointing to } \boldsymbol{x}_t \text { " }}+\underbrace{\sigma_t \epsilon_t}_{\text {random noise }}
> > $$
>
> 从这个结果可以看出， $p\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{y}\right)$ 近似于 $\mathcal{N}\left(\boldsymbol{x}_{t-1} ; \boldsymbol{\mu}\left(\boldsymbol{x}_t\right)+\sigma_t^2 \nabla_{\boldsymbol{x}_t} \log p\left(\boldsymbol{y} \mid \boldsymbol{x}_t\right), \sigma_t^2 \boldsymbol{I}\right)$ ，所以只需要把生成过程的采样改为
> $$
> \boldsymbol{x}_{t-\mathbf{1}}=\boldsymbol{\mu}\left(\boldsymbol{x}_t\right)+\underbrace{\sigma_t^2 \nabla_{\boldsymbol{x}_t} \log p\left(y \mid \boldsymbol{x}_t\right)}_{\text {新增顶 }}+\sigma_t \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{I})
> $$
> **这就是Classifier-Guidance方案的核心结果**。值得注意的是，本文的推导结果跟原论文略有不同，原论文新增项是
> $$
> \left.\sigma_t^2 \nabla_{\boldsymbol{x}_t} \log p\left(\boldsymbol{y} \mid \boldsymbol{x}_t\right)\right|_{\boldsymbol{x}_t=\boldsymbol{\mu}\left(\boldsymbol{x}_t\right)}
> $$
> 也就是梯度项在 $\boldsymbol{\mu}\left(\boldsymbol{x}_t\right)$ 处的结果而非 $\boldsymbol{x}_t$ 处，而一般情况下 $\boldsymbol{\mu}\left(\boldsymbol{x}_t\right)$ 的零阶近似正是 $\boldsymbol{x}_t$ ，所以两者结果是差不多的。
>
> 

##### 2, 泰勒展开2

> We can assume that $\log _\phi p\left(y \mid x_t\right)$ has low curvature compared to $\Sigma^{-1}$. This assumption is reasonable in the limit of infinite diffusion steps, where $\|\Sigma\| \rightarrow 0$. In this case, we can approximate $\log p_\phi\left(y \mid x_t\right)$ using a Taylor expansion around $x_t=\mu$ as 
>
> ==不太清楚二种泰勒展开$\log p\left(\boldsymbol{y} \mid \boldsymbol{x}_{t-1}\right)-\log p\left(\boldsymbol{y} \mid \boldsymbol{x}_t\right)$和 $\log p_\phi\left(y \mid x_t\right) |_{x_t =u}$ 的区别==
> $$
> \begin{aligned}
> \log p_\phi\left(y \mid x_t\right) & \left.\approx \log p_\phi\left(y \mid x_t\right)\right|_{x_t=\mu}+\left.\left(x_t-\mu\right) \nabla_{x_t} \log p_\phi\left(y \mid x_t\right)\right|_{x_t=\mu} \\
> & =\left(x_t-\mu\right) g+C_1
> \end{aligned}
> $$
> Here, $g=\left.\nabla_{x_t} \log p_\phi\left(y \mid x_t\right)\right|_{x_t=\mu}$, and $C_1$ is a constant. This gives
> $$
> \begin{aligned}
> \log \left(p_\theta\left(x_t \mid x_{t+1}\right) p_\phi\left(y \mid x_t\right)\right) & \approx-\frac{1}{2}\left(x_t-\mu\right)^T \Sigma^{-1}\left(x_t-\mu\right)+\left(x_t-\mu\right) g+C_2 \\
> & =-\frac{1}{2}\left(x_t-\mu-\Sigma g\right)^T \Sigma^{-1}\left(x_t-\mu-\Sigma g\right)+\frac{1}{2} g^T \Sigma g+C_2 \\
> & =-\frac{1}{2}\left(x_t-\mu-\Sigma g\right)^T \Sigma^{-1}\left(x_t-\mu-\Sigma g\right)+C_3 \\
> & =\log p(z)+C_4, z \sim \mathcal{N}(\mu+\Sigma g, \Sigma)
> \end{aligned}
> $$
> Recall that our diffusion model (==DDPM==) predicts the previous timestep $x_t$ from timestep $x_{t+1}$ using a Gaussian distribution:
> $$
> \begin{aligned}
> p_\theta\left(x_t \mid x_{t+1}\right) & =\mathcal{N}(\mu, \Sigma) \\
> \log p_\theta\left(x_t \mid x_{t+1}\right) & =-\frac{1}{2}\left(x_t-\mu\right)^T \Sigma^{-1}\left(x_t-\mu\right)+C
> \end{aligned}
> $$
> **We can safely ignore the constant term C4, since it corresponds to the normalizing coefficient Z in Equation 2.**We have thus found that the conditional transition operator can be approximated by a Gaussian similar to the unconditional transition operator, but with its mean shifted by **Σg.** Algorithm 1 summaries the corresponding sampling algorithm. We include an optional scale factor s for the gradients.
>
> ==**Above go to Algorithm 1**==
>
> In particular, if we have a model $θ(x_t)$ that predicts the noise added to a sample, then this can be used to derive a score function:
> $$
> \nabla_{x_t} \log p_\theta\left(x_t\right)=-\frac{1}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta\left(x_t\right)
> $$
> ==貌似这里和Algorithm1 没有联系：$p_{\theta, \phi}\left(\mathbf{x}_t \mid \mathbf{x}_{t+1}, \mathbf{y}\right) 
>  =Z p_\theta\left(\mathbf{x}_t \mid \mathbf{x}_{t+1}\right) p_\phi\left(\mathbf{y} \mid \mathbf{x}_t\right) $== We can now substitute this into the ==score function for $p\left(x_t\right) p\left(y \mid x_t\right)$==**这个score function的来源看下面解释explain3**:
> $$
> \begin{aligned}
> \nabla_{x_t} \log \left(p_\theta\left(x_t\right) p_\phi\left(y \mid x_t\right)\right) & =\nabla_{x_t} \log p_\theta\left(x_t\right)+\nabla_{x_t} \log p_\phi\left(y \mid x_t\right) \\
> & =-\frac{1}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta\left(x_t\right)+\nabla_{x_t} \log p_\phi\left(y \mid x_t\right)
> \end{aligned}
> $$
> Finally, we can define a new epsilon prediction $\hat{\epsilon}\left(x_t\right)$ which corresponds to the score of the joint distribution:
> $$
> \hat{\epsilon}\left(x_t\right):=\epsilon_\theta\left(x_t\right)-\sqrt{1-\bar{\alpha}_t} \nabla_{x_t} \log p_\phi\left(y \mid x_t\right)
> $$
> We can then use the exact same sampling procedure as used for regular DDIM, but with the modified noise predictions $\hat{\epsilon}_\theta\left(x_t\right)$ instead of $\epsilon_\theta\left(x_t\right)$. Algorithm \& summaries the corresponding sampling algorithm.
>
> ==**Above go to Algorithm 2**==



#### Diffusion guidance Explanation 3 （==Suitable for Algorithm 2==）

[Guidance: a cheat code for diffusion models – Sander Dieleman](https://sander.ai/2022/05/26/guidance.html)

Diffusion models are generative models, which means they model a high-dimensional data distribution $p(x)$. Rather than trying to approximate $p(x)$ directly (which is what likelihood-based models do), they try to predict the so-called ==**score function*, $\nabla_x \log p(x)$.**==

To sample from a diffusion model, an input is initialized to random noise, and is then iteratively denoised by taking steps **in the direction of the score function (i.e. the direction in which the log-likelihood increases fastest),** with some additional noise mixed in to avoid getting stuck in modes of the distribution. This is called **[Stochastic Gradient Langevin Dynamics (SGLD)](https://en.wikipedia.org/wiki/Stochastic_gradient_Langevin_dynamics).** This is a bit of a caricature of what people actually use in practice nowadays, but it’s not too far off the truth.

In conditional diffusion models, we have an additional input $\boldsymbol{y}$ (for example, a class label or a text sequence) and we try to model the conditional distribution $p(x \mid y)$ instead. In practice, this means learning to ==**predict the conditional score function $\nabla_x \log p(x \mid y)$.**==

One neat aspect of the score function is that it is invariant to normalization of the distribution: if we only know the distribution $p(x)$ up to a constant, i.e. we have $p(x)=\frac{\bar{p}(x)}{Z}$ and we only know $\tilde{p}(\boldsymbol{x})$, then we can still compute the score function:
$$
\nabla_x \log \tilde{p}(x)=\nabla_x \log (p(x) \cdot Z)=\nabla_x(\log p(x)+\log Z)=\nabla_x \log p(x)
$$
where we have made use of the linearity of the gradient operator, and the fact that the normalization constant $Z=\int \tilde{p}(x) \mathrm{d} x$ does not depend on $x$ (so its derivative w.r.t. $\boldsymbol{x}$ is zero).

Unnormalized probability distributions come up all the time, so this is a useful property. For conditional models, it enables us to apply Bayes' rule to decompose the score function into an unconditional component, and a component that "mixes in" the conditioning information:
$$
\begin{gathered}
p(x \mid y)=\frac{p(y \mid x) \cdot p(x)}{p(y)} \\
\Longrightarrow \log p(x \mid y)=\log p(y \mid x)+\log p(x)-\log p(y) \\
\Longrightarrow \nabla_x \log p(x \mid y)=\nabla_x \log p(y \mid x)+\nabla_x \log p(x)
\end{gathered}
$$
where we have used that $\nabla_x \log p(y)=0$. In other words, ==**we can obtain the conditional score function as simply the sum of the unconditional score function and a conditioning term**==. (Note that the conditioning term $\nabla_x \log p(y \mid x)$ is not itself a score function, because the gradient is w.r.t. $\boldsymbol{x}$, not $\boldsymbol{y}$.)

> Solve $\nabla_x \log p(y \mid x)$ : The first thing to notice is that $p(y∣x)$ is exactly what classifiers and other discriminative models try to fit: $x$ is some high-dimensional input, and $y$ is a target label. If we have a differentiable discriminative model that estimates $p(y∣x)$, then we can also easily obtain $\nabla_x \log p(y \mid x)$. **All we need to turn an unconditional diffusion model into a conditional one, is a classifier!** 引用前面的：==训练一个分类器: This can be obtained by training a classifier $p_\phi\left(y \mid x_t\right)$ on noised images $x_t$ derived by sampling from $q\left(x_t\right)$.==

**The observation that diffusion models can be conditioned *post-hoc* in this way was mentioned by Sohl-Dickstein et al.[4](https://sander.ai/2022/05/26/guidance.html#fn:equilibrium) and Song et al.[5](https://sander.ai/2022/05/26/guidance.html#fn:sde),** 

but Dhariwal and Nichol[6](https://sander.ai/2022/05/26/guidance.html#fn:beatgans) really drove this point home, and showed how *classifier guidance* can dramatically improve sample quality by enhancing the conditioning signal, even when used in combination with traditional conditional modelling. 

To achieve this, they **scale the conditioning term** by a factor:
$$
\nabla_x \log p_\gamma(x \mid y) = \nabla_x \log p(x) + \gamma \nabla_x \log p(y \mid x) .
$$
$γ$ is called the **guidance scale**, and cranking it up beyond 1 has the effect of **amplifying the influence of the conditioning signal**. It is *extremely* effective, especially compared to e.g. the truncation trick for GANs[7](https://sander.ai/2022/05/26/guidance.html#fn:biggan), which serves a similar purpose.

If we revert the gradient and the logarithm operations that we used to go from Bayes’ rule to classifier guidance, it’s easier to see what’s going on:
$$
p_\gamma(x \mid y) \propto p(x) \cdot p(y \mid x)^\gamma .
$$

We are raising the conditional part of the distribution to a power, which corresponds to **tuning the temperature** of that distribution: **γ is an inverse temperature parameter. If γ>1, this sharpens the distribution and focuses it onto its modes**, by shifting probability mass from the least likely to the most likely values (i.e. the temperature is lowered). Classifier guidance allows us to apply this temperature tuning only to the part of the distribution that captures the influence of the conditioning signal.

### Classifier-Free Guidance

[Guidance: a cheat code for diffusion models – Sander Dieleman](https://sander.ai/2022/05/26/guidance.html)

[What are Diffusion Models? | Lil'Log (lilianweng.github.io)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#nice)

> As the name implies, it does not require training a separate classifier， Instead, one trains a conditional diffusion model $p(x∣y)$, with *conditioning dropout*: some percentage of the time, the conditioning information y is removed (10-20% tends to work well). In practice, it is often replaced with a special input value representing the absence of conditioning information. The resulting model is now able to function both as a conditional model p(x∣y), and as an unconditional model p(x), depending on whether the conditioning signal is provided. One might think that this comes at a cost to conditional modelling performance, but the effect seems to be negligible in practice.

**Without an independent classifier $f_\phi$, it is still possible to run conditional diffusion steps** by incorporating the scores from a conditional and an unconditional diffusion model ([Ho & Salimans, 2021](https://openreview.net/forum?id=qw8AKxfYbI)). 

==Let unconditional denoising diffusion model $p_{\boldsymbol{\theta}}(\mathrm{x})$ parameterized through a score estimator $\boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t, t\right)$ and the conditional model $p_\theta(\mathbf{x} \mid y)$ parameterized through $\boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t, t, y\right)$. These two models can be learned via a single neural network.==

==**Precisely, a conditional diffusion model $p_\theta(\mathbf{x} \mid y)$ is trained on paired data $(\mathbf{x}, y)$, where the conditioning information $y$ gets discarded periodically at random such that the model knows how to generate images unconditionally as well**, i.e.==
$$
\epsilon_\theta\left(\mathbf{x}_t, t\right)=\epsilon_\theta\left(\mathbf{x}_t, t, y=\varnothing\right)
$$
**The gradient of an implicit classifier can be represented with conditional and unconditional score estimators**. Once plugged into the classifier-guided modified score, the score contains no dependency on a separate classifier.
$$
\begin{aligned}
\nabla_{\mathbf{x}_t} \log p\left(y \mid \mathbf{x}_t\right) & =\nabla_{\mathbf{x}_t} \log p\left(\mathbf{x}_t \mid y\right)-\nabla_{\mathbf{x}_t} \log p\left(\mathbf{x}_t\right) \\
& =-\frac{1}{\sqrt{1-\bar{\alpha}_t}}\left(\epsilon_\theta\left(\mathbf{x}_t, t, y\right)-\epsilon_\theta\left(\mathbf{x}_t, t\right)\right) \\
\bar{\epsilon}_\theta\left(\mathbf{x}_t, t, y\right) & =\epsilon_\theta\left(\mathbf{x}_t, t, y\right)-\sqrt{1-\bar{\alpha}_t} w \nabla_{\mathbf{x}_t} \log p\left(y \mid \mathbf{x}_t\right) \\
& =\epsilon_\theta\left(\mathbf{x}_t, t, y\right)+w\left(\epsilon_\theta\left(\mathbf{x}_t, t, y\right)-\epsilon_\theta\left(\mathbf{x}_t, t\right)\right) \\
& =(w+1) \epsilon_\theta\left(\mathbf{x}_t, t, y\right)-w \epsilon_\theta\left(\mathbf{x}_t, t\right) \\
&\text {潜在的含义是} \log p\left(y \mid \mathbf{x}_t\right) \text{不再训练额外的分类器，而是通过训练} p(x∣y)拿到
\end{aligned}
$$
或者一下推导也可以得到：

>[Guidance: a cheat code for diffusion models – Sander Dieleman](https://sander.ai/2022/05/26/guidance.html)
>
>We have expressed the conditioning term as a function of the conditional and unconditional score functions, both of which our diffusion model provides. We can now substitute this into the formula for classifier guidance:
>$$
>\nabla_x \log p_\gamma(x \mid y)=\nabla_x \log p(x)+\gamma\left(\nabla_x \log p(x \mid y)-\nabla_x \log p(x)\right)
>$$
>or equivalently:
>$$
>\nabla_x \log p_\gamma(x \mid y)=(1-\gamma) \nabla_x \log p(x)+\gamma \nabla_x \log p(x \mid y)
>$$
>This is a [barycentric combination](https://people.eecs.ku.edu/~jrmiller/Courses/VectorGeometry/AffineTransformations.html) of the conditional and the unconditional score function. For γ=0, we recover the unconditional model, and for γ=1 we get the standard conditional model. But γ>1 is where the magic happens === This makes the resulting gradient much more robust==. Below are some examples from OpenAI’s GLIDE model[8](https://sander.ai/2022/05/26/guidance.html#fn:glide), obtained using classifier-free guidance.

Their experiments showed that classifier-free guidance can achieve a good balance between FID (distinguish between synthetic and generated images) and IS (quality and diversity).

The guided diffusion model, GLIDE (Nichol, Dhariwal \& Ramesh, et al. 2022), explored both guiding strategies, CLIP guidance and classifier-free guidance, and found that the latter is more preferred. They hypothesized that it is because CLIP guidance exploits the model with adversarial examples towards the CLIP model, rather than optimize the better matched images generation.

> 下面的推导较为怪异，不过基本和上面推导类似：
>
> 上文的Classifier-Guide采样方式，需要在采样阶段准备好额外的分类器，因此DDPM的原作者又 提出一种Classifier-Free-Guide 的方式。假如无引导的Diffusion Model为 $\epsilon_\theta\left(x_t\right)$ ，那么带分类 信息引导的可以写为
> $$
> \hat{\epsilon}_\theta\left(x_t \mid y\right)=\epsilon_\theta\left(x_t\right)+s \nabla_{x_t} \log p^i\left(x_t \mid y\right)
> $$
> 由于
> $$
> \begin{aligned}
> \nabla_{x_t} \log p^i\left(x_t \mid y\right) & \propto \nabla_{x_t} \log \left[p\left(x_t \mid y\right) /\left(x_t\right)\right] \\
> & =\nabla_{x_t} \log p\left(x_t \mid y\right)-\nabla_{x_t} \log p\left(x_t\right) \\
> & \propto \epsilon^*\left(x_t \mid y\right)-\epsilon^*\left(x_t\right)
> \end{aligned}
> $$
> 其中， $\epsilon^*\left(x_t \mid y\right), \epsilon^*\left(x_t\right)$ 表示真实分布。因此，采样时的 $\hat{\epsilon}_\theta\left(x_t \mid y\right)$ 可以被写作
> $$
> \hat{\epsilon}_\theta\left(x_t \mid y\right)=\epsilon_\theta\left(x_t\right)+s \cdot\left(\epsilon_\theta\left(x_t \mid y\right)-\epsilon_\theta\left(x_t\right)\right)
> $$
> 接下来就是如何训练出 $\epsilon_\theta\left(x_t \mid y\right), \epsilon_\theta\left(x_t\right)$ 。这两者在Diffusion Model会共用模型，并将带标 记的样本与无标记的样本混合一起训练，对于无标记样本则将 $y=n u l l$ 用于区分即可。

## 4.14 build your own diffusion mode generally

1. Choose the denoiser model: UNet + self attention block
2. Choose the Training loss: MSE loss
$$
\bar{\nabla}_\theta\left\|\boldsymbol{\epsilon}-\boldsymbol{\epsilon}_\theta\left(\sqrt{\bar{\alpha}_t} \mathbf{x}_0+\sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}, t\right)\right\|^2
$$
3. Choose the Sampler (testing): DDIM sampler

$$
\boldsymbol{x}_{t-1}=\sqrt{\alpha_{t-1}} \underbrace{\left(\frac{\boldsymbol{x}_t-\sqrt{1-\alpha_t} \epsilon_\theta^{(t)}\left(\boldsymbol{x}_t\right)}{\sqrt{\alpha_t}}\right)}_{\text {"predicted } \boldsymbol{x}_0 \text { " }}+\underbrace{\sqrt{1-\alpha_{t-1}-\sigma_t^2} \cdot \epsilon_\theta^{(t)}\left(\boldsymbol{x}_t\right)}_{\text {"direction pointing to } \boldsymbol{x}_t \text { " }}+\underbrace{\sigma_t \epsilon_t}_{\text {random noise }}
$$

## 4.15.Text conditioning: Influencing image output via text

[Text-to-Image: Diffusion, Text Conditioning, Guidance, Latent Space (eugeneyan.com)](https://eugeneyan.com/writing/text-to-image/)

### Contrastive Language-Image Pre-training (CLIP; 2021)

**[Contrastive Language-Image Pre-training (CLIP; 2021)](https://arxiv.org/abs/2103.00020)**. **It embeds text and image in the same space via a projection layer**. Thus, it can efficiently learn visual concepts, in the form of text, via natural language supervision and perform zero-shot classification.

[(52) Contrastive Language-Image Pre-training (CLIP) - YouTube](https://www.youtube.com/watch?v=BcfAkQagEWU)

> what makes CLIP really special is *“the appreciation of using natural language as a training signal”*. It does demand access to supervised dataset in which we know which text matches which image. It is trained on 400 million (text, image) pairs, collected from the Internet. The query list contains all the words occurring at least 100 times in the English version of Wikipedia. Interestingly, they found that Transformer-based language models are 3x slower than a bag-of-words (BoW) text encoder at zero-shot ImageNet classification. **Using contrastive objective instead of trying to predict the exact words associated with images** (i.e. a method commonly adopted by image caption prediction tasks) can further improve the data efficiency another 4x.

<img src="/assets/BERTGPTDiffusion%20Research.assets/image-20230610153230803.png" alt="image-20230610153230803" style="zoom:80%;" />

<img src="/assets/BERTGPTDiffusion%20Research.assets/clip.jpg" alt="CLIP pre-training and zero-shot classification" style="zoom: 80%;" />

<img src="/assets/BERTGPTDiffusion%20Research.assets/image-20230610160726600.png" alt="image-20230610160726600" style="zoom:80%;" />

1. In the pre-training stage, the image and text encoders are trained to predict which images are paired with which texts in a dataset of 400M image-caption pairs

2. CLIP is trained to maximize the **cosine similarity of the image and text embeddings** of image-caption pairs via a multi-modal embedding space. 

   1. 图像和文本分别使用encoder编码为embedding，图像可使用 ResNet，文本可使用 Transformer 
   2. 假如每个Batch中样本量为N，图像embedding和文本embedding两两做内积则可以得到一个(N,N)维度的矩阵，其中第i行第j列表示对应图像和文本的相似度
   3. 模型的目标就是使对角线的相似度最大，而非对角线相似度为0，也是借鉴了对比学习的思路

   > ```python
   > # image_encoder - ResNet or Vision Transformer 
   > # text_encoder - CBOW or Text Transformer 
   > # I[n, h, w, c] - minibatch of aligned images 
   > # T[n, l] - minibatch of aligned texts 
   > # W_i[d_i, d_e] - learned proj of image to embed 
   > # W_t[d_t, d_e] - learned proj of text to embed 
   > # t - learned temperature parameter 
   > # extract feature representations of each modality 
   > I_f = image_encoder(I) #[n, d_i] 
   > T_f = text_encoder(T) #[n, d_t] 
   > # joint multimodal embedding [n, d_e] 
   > I_e = l2_normalize(np.dot(I_f, W_i), axis=1) 
   > T_e = l2_normalize(np.dot(T_f, W_t), axis=1) 
   > # scaled pairwise cosine similarities [n, n] 
   > logits = np.dot(I_e, T_e.T) * np.exp(t) 
   > # symmetric loss function 
   > labels = np.arange(n) 
   > loss_i = cross_entropy_loss(logits, labels, axis=0) 
   > loss_t = cross_entropy_loss(logits, labels, axis=1) 
   > loss = (loss_i + loss_t)/2
   > ```
   >
   > ```python
   > def cross_entropy(preds, targets, reduction='none'):
   >     log_softmax = nn.LogSoftmax(dim=-1)
   >     loss = (-targets * log_softmax(preds)).sum(1)
   >     if reduction == "none":
   >         return loss
   >     elif reduction == "mean":
   >         return loss.mean()
   > ```
   >
   > 函数 cosine_similarity 计算向量之间的L2归范化的点积(L2-normalized dot product)。 那就是, 如果 x 和 y 是两个行向量,则它们的余弦相似度(cosine similarity) k 定义如下:
   > $$
   > k(x,y)=xy^T‖x‖‖y‖
   > $$
   > 之所以被称之为 余弦相似度, 是因为 Euclidean (L2) normalization 把两个向量投影到单位球 (unit sphere),这时它们的点积就是两个向量之间的夹角的余弦值。
   >
   > ```python
   > import numpy as np 
   > vec1 = np.array([1, 2, 3, 4]) 
   > vec2 = np.array([5, 6, 7, 8]) 
   > cos_sim = vec1.dot(vec2) / (np.linalg.norm(vec1) *  np.linalg.norm(vec2)) print(cos_sim)
   > ```

3. This is implemented via a linear projection to map each encoder’s representation to the multi-modal embedding space (lines 13 - 15 below). 

4. As a result, the text and image embeddings are now in the same space. **Thus, given a text embedding, we can apply k-nearest neighbors to find similar images.**

5. **Language Guidance**

   > 基于文本条件的图像生成，即希望生成的图像符合文本的描述。在逆向过程中，**每个迭代步要对有噪声的图像和文本计算embedding相似度，作为引导**
   >
   > 了解了CLIP如何定义图像-文本的相似度，那么定义Language Guidance的 $F_\phi\left(x_t, y, t\right)$ 也就比 较容易明白了，具体如下:
   > $$
   > F_\phi\left(x_t, l, t\right)=E_I^{\prime}\left(x_t, t\right) \cdot E_L(l)
   > $$
   > 其中， $l$ 为用于引导的文本， $E_L$ 表示文本encoder。 $x_t$ 为图像， $E_I^{\prime}$ 表示图像encoder。因此，CLIP中图像encoder必须要使用噪声图像finetune。其效果如下
   >
   > <img src="/assets/BERTGPTDiffusion%20Research.assets/v2-e5396b5509f38a1edd10ad12919d54c2_720w.webp" alt="img" style="zoom:33%;" />

6. **Image Guidance**

   > 基于图像条件的图像生成，希望生成的图像与参考的图像尽可能相似
   >
   > 图片引导是指希望生成的图片与一张参考图片相似。我们将参考图记为 $x_0^{\prime}$ ，根据前述DDPM中 的 $q\left(x_t \mid x_0\right)$ 公式，我们可以根据当前逆向过程的 $t$ 获得对应程度的加噪图片 $x_t^{\prime}$ 。通过对比 $x_t^{\prime}$ 与 $x_t$ 引导生成。此处作者提出了三种不同的图片引导函数。
   >
   > 
   >
   > Image Guidance的 $F_\phi\left(x_t, x_t^{\prime}, t\right)$ 定义如下:
   > $$
   > F_\phi\left(x_t, x_t^{\prime}, t\right)=E_I^{\prime}\left(x_t, t\right) \cdot E_I^{\prime}\left(x_t^{\prime}, t\right)
   > $$
   > 其中，定义 $x_0^{\prime}$ 为用于引导的无橾声图像， $x_t^{\prime}$ 为 $x_0^{\prime}$ 生成的加橾图像。
   >
   > <img src="/assets/BERTGPTDiffusion%20Research.assets/webp.webp" alt="img" style="zoom:33%;" />
   >
   > 同样的，CLIP中图像 encoder必须要使用噪声图像finetune。其效果如下
   >
   > <img src="/assets/BERTGPTDiffusion%20Research.assets/v2-58459d39be0e70f297212799d7f96cdc_720w.webp" alt="img" style="zoom:33%;" />
   >
   > - 图片内容引导: 希望图片的内容内容与参考图相似
   > $$
   > F\left(x_t, x_t^{\prime}, t\right)=E_I^{\prime}\left(x_t, t\right) \cdot E_I^{\prime}\left(x_t^{\prime}, t\right)
   > $$
   > - 图片结构引导: 进一步的，我们希望加入更强的引导，即在空间结构上的相似性。这里对比的是 encoder 的spatial feature map
   > $$
   > F\left(x_t, x_t^{\prime}, t\right)=-\sum_i \frac{1}{C H W}\left\|E_I^{\prime}\left(x_t, t\right)_j-E_I^{\prime}\left(x_t^{\prime}, t\right)_j\right\|_2^2
   > $$
   > - 图片风格引导: 基于Gram 矩阵，希望生成图片的风格符合参考图片
   > $F\left(x_t, x_t^{\prime}, t\right)=-\sum_i\left\|G_I^{\prime}\left(x_t, t\right)_j-G_I^{\prime}\left(x_t^{\prime}, t\right)_j\right\|_2^2$ ，此处 $G^{\prime}()_j$ 是 $E_I^{\prime}$ 第层特征的 gram matrix。



![image-20230616141106845](/assets/BERTGPTDiffusion%20Research.assets/image-20230616141106845.png)

#### CLIP guidance

As mentioned earlier, CLIP comprises two parts, an image encoder $f(x)$ and a caption encoder $g(c)$. This approach is similar to guided diffusion except that the perturbation is the dot product of $g(c)$ and $f\left(\mathbf{x}_t\right)$
$$
\hat{\mu}_\theta\left(\mathbf{x}_t \mid c\right)=\mu_\theta\left(\mathbf{x}_t \mid c\right)+s \cdot \Sigma_\theta\left(\mathbf{x}_t \mid c\right) \nabla_{\mathbf{x}_t} \log \left(f\left(\mathbf{x}_t\right) \cdot g(c)\right)
$$
Similar to classifier based guidance, **they train the CLIP model with noised images.**

#### GLIDE Training

While GLIDE was not the first Diffusion Model, its important contribution was in modifying them to allow for **text-conditional image generation**. In particular, one will notice that Diffusion Models *start* from randomly sampled Gaussian noise. It at first unclear how to tailor this process to generate *specific* images. If a Diffusion Model is trained on a human face dataset, it will reliably generate photorealistic images of human faces; but what if someone wants to generate a face with a *specific* feature, like brown eyes or blonde hair?

GLIDE extends the core concept of Diffusion Models by **augmenting the training process with additional textual information**, ultimately resulting in text-conditional image generation. Let's take a look at the training process for GLIDE:



<img src="/assets/BERTGPTDiffusion%20Research.assets/v2-78ba27465f8ed85a5c7b06d05b8b3c3b_b.webp" alt="动图" style="zoom: 67%;" />

no-classifer guidence 可以更好的将条件信息加入到扩散模型的训练中去以得到更好的训练效果，但同时也会增加训练成本。OpenAI 就基于no-classifier guidence 的思想，整了一个超大规模的基于扩散模型的文本图像生成模型GLIDE。其中算法的核心即将前面的类别条件更新为了文本条件：

- 首先将文本编码为 $\mathrm{K}$ 个 token 序列。
- 然后将token输入到 Transformer。
- transformer输出的最后一个token作为扩散模型的条件。其中每一步中都基于生成图像 $\mathrm{E}_{\mathrm{I}}(\mathrm{x})$ 与 文本 $\mathrm{E}_{\mathrm{L}}(\mathrm{l})$ 之间的相似度来计算梯度 $\mathrm{F}\left(\mathrm{x}_{\mathrm{t}}, \mathrm{l}, \mathrm{t}\right)=\mathrm{E}_{\mathrm{I}}^{\prime}\left(\mathrm{x}_{\mathrm{t}}\right) \cdot \mathrm{E}_{\mathrm{L}}(\mathrm{l})$ 。且 $G L I D E$ 是无分类器的扩 散引导:

$$
\hat{\epsilon}_\theta\left(x_t \mid \text { Caption }\right)=\epsilon_\theta\left(x_t\right)+s \cdot\left(\epsilon_\theta\left(x_t, \text { Caption }\right)-\epsilon_\theta\left(x_t\right)\right)
$$

这里无非就是把原来的label $y$换成了 caption，**实际上就是运用了足够量的image-text pair**从而可以把caption当作是某种程度上的label。（随机替换为空序列以实现unconditional的训练方式）

由于此时的生成图像质量一般般，文章也提供了图像编辑的方式（具体操作为：将选中区域mask掉，将图像也作为一个condition连同文本输入到模型中去）

### DALL·E (2021)

CLIP was quickly followed up by **[DALL·E (2021)](https://arxiv.org/abs/2102.12092), one of the first text-to-image generation models open to the public**

DALL·E由OpenAI在2021年初提出，旨在训练一个输入文本到输出图像的自回归解码器。由CLIP的成功经验可知，文本特征和图像特征可以编码在同一特征空间中，因此我们可以使用Transformer将文本和图像特征自回归建模为单个数据流（“autoregressively models the text and image tokens as a single stream of data”）。

DALL·E的训练过程分成两个阶段，一是训练一个变分自编码器用于图像编解码，二是训练一个文本和图像的自回归解码器用于预测生成图像的Tokens，

![img](/assets/BERTGPTDiffusion%20Research.assets/3190ca843b20efdce34291cc54b3e5dd27b16f.jpg)

#### DALL·E mini Model Architecture  - Training Process

Images and descriptions are both provided during training and flow through the system in the following order:

- Images are encoded through a [VQGAN](https://arxiv.org/abs/2012.09841) encoder, which turns images into a sequence of tokens.
- Descriptions are encoded through a [BART](https://arxiv.org/abs/1910.13461) encoder.
- The output of the BART encoder and encoded images are fed through the BART decoder, which is an auto-regressive model whose goal is to predict the next token.
- Loss is the [softmax cross-entropy](https://wandb.ai/sauravm/Activation-Functions/reports/Activation-Functions-Softmax--VmlldzoxNDU1Njgy#📢-softmax-+-cross-entropy-loss-(caution:-math-alert)) between the model prediction logits and the actual image encodings from the VQGAN.

<img src="/assets/BERTGPTDiffusion%20Research.assets/0uYCW3oCeVXVZLEOQ.png" alt="img" style="zoom:67%;" />

#### Inference Process

At inference time, one only has captions available and wants to generate images:

- The caption is encoded through the BART encoder.

- A <BOS> token (special token identifying the “Beginning Of Sequence”) is fed through the BART decoder.

- Image tokens are sampled sequentially based on the decoder’s predicted distribution over the next token.

- Sequences of image tokens are decoded through the VQGAN decoder.
  CLIP is used to select the best generated images.

  ![img](/assets/BERTGPTDiffusion%20Research.assets/0wU__CnIvF0X7U4Gd.png)

#### General DALLE

1. **learning the vocabulary of the image-text pairs **（变分自编码器用于图像编解码）:  At a high level, **DALL·E starts by compressing images into 8,192 discrete tokens in a visual codebook** (Z in the image below).  DALL·E trains a **discrete variational encoder (dVAE)** to compress 256 x 256 images into 32 x 32=1024 integers image tokens (vocabulary size = 8,192). The parameters of the dVAE are then frozen when training the transformer.

   训练离散的变分自编码器（dVAE），对图片进行压缩，将256X256的pixel压缩为32X32的token序列 ，原本每个像素的值为[0,255]，现在经过codebook对图像的特征块进行token离散化，codebook大小为8192，这样就可以将32*32的token矩阵转为1024的token序列，并且以此作为image部分的输入特征。

   > Why compress images into tokens in a codebook? The authors explained that using pixels directly as image tokens would require too much memory for high-resolution images. **As a result, model capacity is spent on high-frequency details (i.e., pixels) instead of low-frequency structure (i.e., lines) that make images visually recognizable**. (This is the same reason Stable diffusion encodes images into the latent space before running diffusion.)

![Visual example of a codebook from the VQGAN paper](/assets/BERTGPTDiffusion%20Research.assets/vqgan-codebook.jpg)

和VAE一样我们用概率编码器和概率解码器，分别建模隐层特征的后验概率分布和生成图像的似然概率分布，使用建模由Transformer预测的文本和图像的联合概率分布作为先验（在第一阶段初始化为均匀分布），同理可得优化目标的证据下界，
$$
\log p_{\theta, \psi}(x, y) \geq \mathbb{E}_{z \sim q_\phi(z \mid x)} \log p_\theta(x \mid y, z)-\beta D_{K L}\left(q_\phi(y, z \mid x) \| p_\psi(y, z)\right)
$$
在第一阶段的训练过程中，DALL·E使用了一个离散变分自编码器（Discrete VAE）简称dVAE，是Vector Quantized VAE（VQ-VAE）的升级版。在VAE中我们用一个概率分布刻画了连续的隐层空间，通过随机采样得到隐层编码，但是这个编码并不像离散的语言文字具有确定性。为了学习图像隐层空间的“语言”，VQ-VAE使用了一组可学习的向量量化表示隐层空间，这个量化的隐层空间我们称为Embedding Space或者Codebook/Vocabulary。VQ-VAE的训练过程和预测过程旨在寻找与图像编码向量距离最近的隐层向量，再将映射得到的向量语言解码成图像（图12），损失函数由三部分构成，分别优化重构损失、更新Embedding Space和更新编码器，梯度终止
$$
L_{V Q-V A E}=\log (p(x \mid q(x)))+\left\|s g\left[z_e(x)\right]-e\right\|^2+\left\|z_e(x)-s g[e]\right\|^2
$$
![VQ-VAE example](/assets/BERTGPTDiffusion%20Research.assets/68747470733a2f2f692e696d6775722e636f6d2f5239564d5744362e706e67.png)

VQ-VAE由于最近邻选择假设使其后验概率是确定的，即距离最近的隐层向量概率为1其余为0，不具有随机性；距离最近的向量选择过程不可导，使用了straight-through estimator方法将的梯度传递给。

![img](/assets/BERTGPTDiffusion%20Research.assets/f5738b1380288b21dd0891ff9598704a45ca58.jpg)

为了优化上述问题，DALL·E使用Gumbel-Softmax构建了新的dVAE（图13），解码器的输出变为Embedding Space上32*32个K=8192维分类概率，在训练过程中对分类概率的Softmax计算加入噪声引入随机性，使用逐步减小的温度让概率分布近似one-hot编码，对隐层向量的选择重参数化使其可导（式(11)），推理过程中仍取最近邻。
$$
\begin{aligned}
&y_i=\frac{e^{\left(g_i+\log \left(q\left(e_i \mid x\right)\right)\right) / \tau}}{\sum_{j=1}^K e^{\left(g_j+\log \left(q\left(e_j \mid x\right)\right)\right) / \tau}}\\
&z=\sum_{j=1}^K y_j e_j
\end{aligned}
$$
当第一阶段训练完成后，我们可以固定dVAE对于每对文本-图像生成预测目标的图像Tokens。

2. The second part was about **learning the prior distribution over the text and image tokens.**  Text部分则直接通过BPE进行token化，长度限制为256，最终输入到模Transformer中的**输入数据**为concat[text token, image token]，对于一个图像文本对，文本特征(256)，图像特征(32x32)，然后将这两个特征拼接成一个1280长度的序列，再输入至GPT中-按照自回归的方式进行训练。

   - What they did here is **concatenate 256 tokens obtained from encoding the input text prompts with the encoded 1024 tokens from their corresponding image**. image captions are lowercased and truncated to a max length of 256 tokens before being encoded (vocabulary size = 16,384). The image tokens are then concatenated after the text tokens (example below).

     ![Example of concatenated text and image tokens in DALL·E](/assets/BERTGPTDiffusion%20Research.assets/dalle-sequence.jpg)

     - an autoregressive transformer (i.e., predict the next item in a sequence) is trained to learn the joint distribution over the text and image tokens. **The transformer is decoder-only,** where each image token can attend to all text tokens earlier in the sequence.  

     - training a transformer to model this autoregressively as a single stream of data of 1024+256=1080 tokens. The result is that from an initial set of at least 256 tokens, the model will "autocomplete" the remaining ones such that **an image is generated that is consistent to the initial tokens** [3].
     
     - To generate images from text, the text prompt is embedded and fed into the transformer. The transformer then generates the sequence of image tokens. Finally, the dVAE decodes the image tokens to return a 256 x 256 image.

       ![img](/assets/BERTGPTDiffusion%20Research.assets/v2-68aee588111332bc36975912295e622f_720w.webp)

       ![img](/assets/BERTGPTDiffusion%20Research.assets/v2-eb3622e724086366a9bc39f18e43c1ea_720w.webp)
     
     **训练目标**就是通过近似的变分下界（VLB）来训练：
     $$
     \ln p_{\theta, \varphi}(x, y) \geq \sum_{z \sim q_\phi(z \mid x)}\left(\ln p_\theta(x \mid y, z)-\beta D_{K L}\left(q_\phi(y, z \mid x), p_{\varphi}(y, z)\right)\right)
     $$
     在第二阶段训练过程中，DALL·E使用BPE方法将文本先编码成和图像Tokens相同维度d=3968的文本Tokens，再将文本Tokens和图像Tokens Concat到一起，加入位置编码和Padding编码，使用Transformer Encoder进行自回归预测，为了提升计算速度，DALL·E还采用了Row、Column、Convolutional三种稀疏化的attention mask机制。
     
     DALL·E中的Transformer结构由64层attention层组成，每层的注意力头数为62，每个注意力头的维度为64，因此，每个token的向量表示维度为3968。如图所示，attention层使用了行注意力mask、列注意力mask和卷积注意力mask三种稀疏注意力。
     
     ![img](/assets/BERTGPTDiffusion%20Research.assets/v2-7eabf68f79230423439bc0d12ff94919_720w.png)
     
     In summary, with the dVAE from the first stage and the autoregressive transformer from the second one, a single step of DALL-E would have to (1) use the transformer to predict the following 1024 image tokens from the first 256 tokens obtained from the input text-prompt and (2) take the full stream of 1024 image tokens that are generated by the transformer and generate an image using the dVAE to map from the embedding space onto the image space.
     
     > [1] The name DALL-E comes from a wordplay combining **WALL-E**, the Disney's Pixar character, and **Dalí** from *Salvador Dalí*, the famous spanish painter.
     >
     > [2] Oord, Aaron van den, Oriol Vinyals, and Koray Kavukcuoglu. "Neural discrete representation learning." (2017) [[Link\]](https://arxiv.org/pdf/1711.00937.pdf)
     >
     > [3] This is similar to what GTP-3 (another language model by OpenAI) does to generate text from an initial text-input. Although GTP-3 is more than 10 times larger than DALL-E with 175 billion parameters ([Source](https://arxiv.org/abs/2005.14165)).


​	3.  推理阶段，给定一张候选图片和一条文本，通过transformer可以得到融合后的token，然后用dVAE的decoder生成图片，最后通过预训练好的CLIP计算出文本和生成图片的匹配分数，采样越多数量的图片，就可以通过CLIP得到不同采样图片的分数排序(详细过程可以看非官方实现[DALLE-pytorch/dalle_pytorch.py](https://link.zhihu.com/?target=https%3A//github.com/lucidrains/DALLE-pytorch/blob/961bba948124a135120db477ef9a55329a7feac8/dalle_pytorch/dalle_pytorch.py%23L447))

### DALL·E 2 (2022)

[How DALL-E 2 Actually Works (assemblyai.com)](https://www.assemblyai.com/blog/how-dall-e-2-actually-works/)

**[DALL·E 2 (aka unCLIP, 2022)](https://arxiv.org/abs/2204.06125) builds on the previous two papers** **by using the text and image encoder from CLIP and the autoregressive transformer from DALL·E. Similarly**, unCLIP is trained on a dataset of image-caption pairs which are embedded via CLIP text and image encoders into **text embeddings ($z_t$) and image embeddings ($z_i$).**

1. generate a [CLIP model](https://vaclavkosar.com/ml/openai-dall-e-2-and-dall-e-1#openais-clip) text embedding for text caption
2. “prior” network generates CLIP image embedding from text embedding
3. diffusion decoder generates image from the image embedding

![How the encoded text (blue) generates images via the prior and decoder](/assets/BERTGPTDiffusion%20Research.assets/unclip.jpg)

![DALL-E 2 decoder](/assets/BERTGPTDiffusion%20Research.assets/dall-e-2-decoder.png)

- **the prior** $\left(p\left(z_i \mid y\right)\right)$ learns to produce **CLIP image embeddings** $\left(z_i\right)$ conditioned on the text prompt $(y)$. 

- The decoder $\left(p\left(x \mid z_i, y\right)\right)$ then produces the image conditioned on the CLIP image embedding $\left(z_i\right)$ and optional text prompt $(y)$. 
- In other words, to generate images from text prompts $(p(x \mid y)$ ), we first sample CLIP image embeddings via the prior before decoding them via the decoder.

$$
p(x \mid y)=P\left(x, z_i \mid y\right)=P\left(x \mid z_i, y\right) P\left(z_i \mid y\right)
$$
**The paper shared two approaches to learn the prior: autoregressive and diffusion.**

- **The autoregressive approach** (clip) is similar to that of DALL·E where text conditioning is done by having the text embedding early in the sequence. They also prepend a dot product token (of text and image embedding) between the text and image embedding. This allowed the autoregressive prior to condition the model on the higher dot product since a higher text-image dot product indicates images that are more representative of the caption.

- **For the diffusion approach**, they **trained a decoder-only transformer with a casual attention mask** on a **sequence of encoded text, text embedding, time step embedding, noised CLIP image embedding, and final embedding**. **The** final embedding’s output is then used to predict the unnoised CLIP image embedding. Interestingly, in contrast to DDPM, they found it better to train the model to directly predict the unnoised image, instead of predicting the noise and then subtracting from the noisy image.

The latter shows one way text conditioning can be applied to diffusion. The transformer attends to the text information in the sequence and uses it to predict the final output.

> DALLE2的模型结构如上图，其中扩散模型是基于GLIDE的。
>
> 虚线上半部分是预训练好的CLIP。一侧输入文本，一侧是图像，用于得到表征。
> 虚线下半部分是text-to-image的生成过程。这一过程是二阶的过程，即文本变图像特征，再特性特征变图像。首先文本特征输入autoregressive或者diffusion prior以得到初步的图像特征（实验证明diffusion效率更高，因此一般选用diffusion），然后该特征会进一步作为condition到反向扩散模型中生成最后的图片。
>
> 值得注意的是 GLIDE 模型以两种方式使用投影的 CLIP 文本嵌入。第一种是将它们添加到 GLIDE 现有的时间步嵌入中，第二种是通过创建四个额外的上下文 token，它们连接到 GLIDE 文本编码器的输出序列。

#### Step 1 - Linking Textual and Visual Semantics

The **link between textual semantics and their visual representations** in DALL-E 2 is learned by another OpenAI model called **CLIP** (**C**ontrastive **L**anguage-**I**mage **P**re-training).

#### Step 2 - Generating Images from Visual Semantics

After training, the CLIP model is frozen and DALL-E 2 moves onto its next task - **learning to *reverse* the image encoding mapping that CLIP just learned**. CLIP learns a representation space in which it is easy to determine the relatedness of textual and visual encodings, but our interest is in image **generation**. We must therefore learn how to exploit the representation space to accomplish this task.

In particular, OpenAI employs a modified version of another one of its previous models, [GLIDE](https://arxiv.org/abs/2112.10741?ref=assemblyai.com) ([Ablated Diffusion Model](https://arxiv.org/abs/2105.05233?ref=assemblyai.com) (ADM), Classifier-Free Guidance) ,  to perform this image generation. The GLIDE model learns to *invert* the image encoding process in order to stochastically decode CLIP image embeddings.

![img](/assets/BERTGPTDiffusion%20Research.assets/CLIP_to_GLIDE-1.png)

An image of a Corgi playing a flamethrowing trumpet passed through CLIP's image encoder. GLIDE then uses this encoding to generate a new image that maintains the salient features of the original. (modified from [source](https://arxiv.org/abs/2204.06125?ref=assemblyai.com))

As depicted in the image above, it should be noted that the goal is **not** to build an autoencoder and *exactly* reconstruct an image given its embedding, but to instead generate an image which **maintains the salient features of the original image** given its embedding. In order perform this image generation, GLIDE uses a **Diffusion Model**.

Therefore, DALL-E 2's modified GLIDE learns to **generate semantically consistent images conditioned on CLIP image encodings**.  It is also important to note that the reverse-Diffusion process is stochastic, and therefore variations can easily be generated by inputting the *same* image encoding vectors through the modified GLIDE model multiple times.

#### Step 3 - Mapping from Textual Semantics to Corresponding Visual Semantics

Recall that, in addition to our *image* encoder, CLIP also learns a *text* encoder. DALL-E 2 uses another model, which the authors call the **prior**, in order to map **from the text encodings** of image captions **to the** **image encodings** of their corresponding images. **The DALL-E 2 authors experiment with both Autoregressive Models and Diffusion Models for the prior, but ultimately find that they yield comparable performance**. Given that the Diffusion Model is much more computationally efficient, it is selected as the prior for DALL-E 2.

<img src="/assets/BERTGPTDiffusion%20Research.assets/text_to_image_encoding_2.png" alt="img" style="zoom:50%;" />

Prior mapping from a text encoding to its corresponding image encoding (modified from [source](https://arxiv.org/abs/2204.06125?ref=assemblyai.com)).

> Prior Training
>
> The Diffusion Prior in DALL-E 2 consists of a **decoder-only Transformer**. It operates, **with a causal attention mask**, on an ordered sequence of
>
> 1. The tokenized text/caption.
> 2. The CLIP text encodings of these tokens.
> 3. An encoding for the diffusion timestep.
> 4. The noised image passed through the CLIP image encoder.
> 5. Final encoding whose output from Transformer is used to predict the unnoised CLIP image encoding.
>
> - Conditioning on the Caption
>   - The Diffusion Prior is conditioned **not only on the CLIP text embedding of the caption, but also the caption itself**. The former is a deterministic function of the latter and this dual-conditioning is therefore fully permissible.
> - Classifier-Free Guidance
>   - To improve sample quality, sampling is randomly conducted using classifier-free guidance 10% of the time by dropping the text-conditioning information.
> - Double Sample Generation
>   - To improve quality during sampling time, two image embeddings are generated with the prior and the one with the higher dot product with the text embedding is selected. It is unclear why the authors use the dot product here as opposed to the cosine similarity.
> - **Why do we need the prior?** The authors note that training such a prior is not strictly necessary for a **caption-to-image model**. 
>   - **One option would be to condition only on the caption itself**. This would simply yield the model GLIDE, and the authors perform a thorough analysis comparing the two in the paper. 
>   - Another option would be to feed into the **decoder the CLIP text embedding**, rather than **using the prior to generate a CLIP image embedding** from it and then use that. The authors found experimentally that the former produces reasonable results, although results not as good as those of the latter. Ultimately, using the prior **improves image diversity**.

#### Step 4 - Putting It All Together

At this point, we have all of DALL-E 2's functional components and need only to chain them together for text-conditional image generation:

1. First the CLIP text encoder maps the image description into the **representation space**. The **link between textual semantics and their visual representations** in DALL-E 2 is learned by another OpenAI model called **CLIP** (**C**ontrastive **L**anguage-**I**mage **P**re-training)
2. Then the diffusion prior maps from the CLIP text encoding to a **corresponding CLIP image encoding**.
3. Finally, the modified-GLIDE generation model maps from the representation space into the image space via reverse-Diffusion, **generating one of many possible images that conveys the semantic information** within the input caption.

### Imagen （2022）

**[Imagen (2022)](https://arxiv.org/abs/2205.11487) takes it further by using a text encoder that wasn’t even trained on image-caption pairs (🤯).** It uses the encoder network of the [T5](https://arxiv.org/abs/1910.10683).  **This is a departure from CLIP-based approaches, where the text encoder is specifically trained on image-caption pairs and the text embeddings are projected into a multi-modal embedding space.**

Imagen 的组成：(1) 文本编码器，用于将文本映射成 embeddings 序列 ；(2) 级联条件扩散模型，用于将 embeddings 序列映射成图像，并逐步增加图像分辨率（参见 图 A.4)，以下将详细描述这些组件

![img](/assets/BERTGPTDiffusion%20Research.assets/image-20220606210644557.png)

1. 在当前的文本到图像模型中通常使用在配对的图文数据集上训练的文本编码器，比如 CLIP。大型的语言模型可以作为另一种选择，例如 BERT、 GPT、 T5 ， 语言模型在纯文本语料库上训练，训练数据远多于成对的图像-文本数据， 所以其可以接触更加丰富和广泛分布的文本。语言模型通常也大得多（例如，PaLM 有 540 B 参数，而 CoCa 有1B 参数）

   Imagen 研究比较了预训练文本编码器：BERT 、T5 和 CLIP。这些文本编码器的权重是冻结的，这样做的好处是可以离线计算文本嵌入，在训练文本到图像生成模型期间，计算或内存占用可以忽略不计。经过实验比较发现文本编码器大小会影响文本到图像生成质量， T5-XXL 在图像-文本对齐、图像逼真度方面可以取得最好的成绩。

    It works because extremely large language models (LLMs), by virtue of sheer size, can still learn useful representations despite not being explicitly trained on text-to-image tasks. The benefit is that LLMs can learn on a text-only corpus which is easily larger than image-text datasets. Furthermore, they found that **scaling the text encoder size is more impactful than UNet size in image-text alignment and image fidelity.**

![Text encoder size > UNet size; dynamic thresholding > static thresholding](/assets/BERTGPTDiffusion%20Research.assets/imagen-curves.jpg)

![img](/assets/BERTGPTDiffusion%20Research.assets/image-20220606211106144.png)

#### 实施细节

Imagen 的提出的改进主要体现在：

- 引入新的动态阈值技术，这样采样器可以使用非常大的无分类器指导权重；
- 在超分辨率模型中引入噪声增强，提高图像逼真度；
- 引入一种新的高效 U-Net 架构，这种架构具有更高的计算效率、更高的内存效率和更快的收敛速度；

#### 阈值技术

增加 classifier-free guidance 的指导权重可以提高图像-文本的对齐，但会影响图像逼真度， 产生高度饱和和不自然的图像。导致这个现象的原因是高指导权重引起训练测试不匹配: 在 每个采样步 $t, x$ 的预测值 $\hat{x}_0^t$ 必须与训练数据在同一范围内，即在 $[-1,1]$ 内。但使用高指 导权重会使 $x$ 预测值超出这些界限。这样就导致训练测试不匹配的情形，扩散模型在整个 采样过程中会迭代应用自身输出，这样的采样过程会导致产生不自然的图像，有时甚至发 散。

> #### Large Guidance Weight Samplers
>
> Classifier-Free Guidance is a very powerful way to improve the caption alignment of generated images, but it has [been](https://arxiv.org/pdf/2112.10741.pdf?ref=assemblyai.com) [previously](https://arxiv.org/abs/2105.05233?ref=assemblyai.com) [observed](https://arxiv.org/abs/2204.06125?ref=assemblyai.com) that extremely high guidance weights damage fidelity by yielding saturated and unnatural images.
>
> The Imagen authors investigate this phenomenon and find that it arises from a **train-test mismatch**. In particular, the pixel values for the training data are scaled to the range [-1, 1], but **high guidance weights cause the network outputs to exceed these bounds** at given timestep. To make matters worse, since the same model is iteratively applied to its own output during diffusion, this effect compounds as the diffusion process proceeds, leading even potentially to divergence.
>
> **High guidance weights are found to be crucial for achieving State-of-the-Art image quality**, so avoiding the problem by simply using lower guidance weights is not an option. Instead, the authors address the problem by devising two methods to threshold pixel values - **static thresholding** and **dynamic thresholding**. These methods address the train-test mismatch noted above and dynamic thresholding in particular is found to be critical to Imagen's performance.

为解决上述训练测试不匹配问题，引入了阈值技术:

- 静态阈值: 对 $x$ 的预测值逐元素裁剪到 $[-1,1]$ ，称为静态阈值。这样方法对于大引导权重的 采样至关重要，并可以防止产生空白图片。尽管如此，随着指导权重的增加，静态阈值处理 仍会使图像出现过饱和或细节少的问题，伪代码如下:

  ```python
  def sample():
      for t in reversed(range(T)):
          # Forward pass to get x0_t from z_t.
          x0_t = nn(z_t, t)
          # Static thresholding.
          x0_t = jnp.clip(x0_t, -1.0, 1.0)
          # Sampler step.
          z_tm1 = sampler_step(x0_t, z_t, t)
          z_t = z_tm1
      return x0_t
  ```

  > ##### Static Thresholding
  >
  > In static thresholding, the pixel values at each timestep are simply clipped to the range [-1, 1]. This process can be visualized in the example below.
  >
  > <video src="/assets/BERTGPTDiffusion%20Research.assets/static_threshold.mp4"></video>
  >
  > For the sale of example, let our pixel values be normally distributed. Applying static thresholding to these values means that any distribution weight that it outside of the pixel bounds (light red area) is pushed onto -1 for negative values and 1 for positive values. As we can see, as the variance of the distribution grows, the probability of being at an extreme value grows.

- 动态國值: 在每个采样步将 $s$ 设置为 $x_0^t$ 中的某个百分位绝对像素值，如果 $s>1$. 那么我 们将 $\boldsymbol{x}_0^t$ 调整到 $[-\mathrm{s}, \mathrm{s}]$ 范围内，然后除以 $s$. 动态阈值处理可以推动饱和像素（接近 -1 或 1) 向内收缩，从而主动防止像素在每一步产生饱和。这可显著提高图像的真实感，以及更好的 图像文本对齐，尤其是在使用非常大的引导权重时，伪代码如下:

  ```python
  def sample(p: float):
      for t in reversed(range(T)):
          # Forward pass to get x0_t from z_t.
          x0_t = nn(z_t, t)
          # Dynamic thresholding (ours).
          s = jnp.percentile(jnp.abs(x0_t), p,axis=tuple(range(1, x0_t.ndim)))
          s = jnp.max(s, 1.0)
          x0_t = jnp.clip(x0_t, -s, s) / s
          # Sampler step.
          z_tm1 = sampler_step(x0_t, z_t, t)
          z_t = z_tm1
      return x0_t
  ```

  ![img](/assets/BERTGPTDiffusion%20Research.assets/image-20220606111105610.png)

> ##### Dynamic Thresholding
>
> With dynamic thresholding, **a certain percentile absolute pixel value is chosen**. At each timestep, if that percentile value *s* exceeds 1, then the pixel values are thresholded to [-*s*, *s*] and divided by *s*. This process can be visualized in the below video:
>
> 
>
> <video src="/assets/BERTGPTDiffusion%20Research.assets/dynamic_threshold.mp4"></video>
>
> Dynamic thresholding has the effect of bringing all pixel values back to the range [-1, 1], but operating on all pixels and not just those at the extreme. There is a "gravitational pull" back to 0 which balances the potential for divergence under an iteratively applied model.
>
> The authors find that this method leads to much better photorealism and alignment, especially for large guidance weights.

Imagen does text conditioning by first tokenizing the input text and encoding it via the T5 encoder. The encoded text then passes through a pooling step (image below).

<video src="/assets/BERTGPTDiffusion%20Research.assets/cap_cond.mp4" auto-play="true"></video>

<video src="/assets/BERTGPTDiffusion%20Research.assets/super_res.mp4"></video>

#### Timestep Conditioning

In Imagen (and generally Diffusion Models as a whole), the same denoising U-Net is used at every timestep. Recall that different amounts of noise are removed at different timesteps in a Diffusion Model. We must therefore devise a way to inject timestep information into the model (i.e. *condition* on the timestep). The Imagen authors utilize a technique introduced by the original Transformer paper called **positional encoding**

<img src="/assets/BERTGPTDiffusion%20Research.assets/image-20230731150119299.png" alt="image-20230731150119299" style="zoom:67%;" />

In Imagen, a unique **timestep encoding vector** is generated for each timestep (corresponding to "word position" in the original positional embedding implementation). At different resolutions in the U-Net, this vector is projected to having *c* components, where *c* is the number of channels in the U-Net at that resolution. After projection, each component of the vector is added to the corresponding channel (across its height and width) in the image.

This process is visualized below for the case of a 3x3 image 

<video src="/assets/BERTGPTDiffusion%20Research.assets/time_enc.mp4"></video>

#### Caption Conditioning

The text embedding is then combined with the image and time step embedding (image below). The model is conditioned via cross-attention over the text embedding. This is implemented by concatenating the text embedding to the key-value pairs of each self-attention layer in the UNet. Cross-attention on the text embedding outperformed simple mean or attention-based pooling.

We've yet to incorporate information from our image caption into the Diffusion Model U-Net, so we need to do that now. This caption conditioning happens in two ways.

- First, the output vectors from the T5 text encoder are pooled and added into the timestep embedding from above. This process is visualized in the below image:

<img src="/assets/BERTGPTDiffusion%20Research.assets/imagen-conditioning.jpg" alt="Conditioning on time and text embeddings in Imagen" style="zoom:67%;" />

- Next, the model is conditioned on the entire encoding *sequence* **by adding cross attention over the text embeddings** at several resolutions. <font color=red>The cross attention is implemented by concatenating the text embedding sequence to the key-value pairs of each self-attention layer</font>.

  The text embedding (green and red boxes below) is used throughout the image generation step. First, it’s used to generate the initial 64 x 64 image from noise (blue box). Then, it is used to increase the image resolution to 256 x 256 and then 1,024 x 1,024 (yellow boxes).

![High-level overview of Imagen](/assets/BERTGPTDiffusion%20Research.assets/imagen-high-level.jpg)

With text conditioning, we can now generate images based on text prompts. But text conditioning alone is insufficient to generate high-quality images that adhere to the text prompt—*we also need guidance.*

#### Classifier-Free Guidance

Imagen also takes advantage of Classifier-Free Guidance. Classifier-Free Guidance is a method of increasing the image fidelity of a Diffusion Model at the cost of image diversity. The method is named as such due to the fact that it is a related and simpler version/extension of a previous method called Classifier Guidance, which was used for the same purposes.

Classifier-Free Guidance **works by training a Diffusion Model to be both conditional and unconditional *at the same time*.** In order to do this, the Diffusion Model is cast as a conditional model and **<font color=red>is trained with the conditioning information randomly dropped out a small fraction of the time (by replacing the conditional information with a NULL value)</font>**. To use the model in an unconditional way, the NULL value is simply provided as the "conditional information" to the model.

Given such a model, Classifier-Free guidance works loosely by *interpolating between the unconditional and conditional gradients* during inference. By magnifying the effect of the conditional gradient (i.e. making the "**guidance weight**" greater than 1), better samples can be obtained:

![img](/assets/BERTGPTDiffusion%20Research.assets/guidance.png)

Although Classifier-Free Guidance was first introduced by [Ho and Salimans](https://openreview.net/pdf?id=qw8AKxfYbI&ref=assemblyai.com), it was soon after notably used in OpenAI's [GLIDE](https://arxiv.org/pdf/2112.10741.pdf?ref=assemblyai.com) in order to create very high quality (albeit lower diversity) images. For a great resource on Classifier/Classifier-Free Guidance, check out [this ](https://benanne.github.io/2022/05/26/guidance.html?ref=assemblyai.com)write-up.

According to Imagen's paper, **Imagen depends critically on classifier-free guidance for effective text conditioning**.





### 零次学习（Zero-Shot Learning)

#### reference 参考文献

[1]Learning To Detect Unseen Object Classes by Between-Class Attribute Transfer

[2]Transductive Multi-View Zero-Shot Learning.

[3]Hubness and Pollution: Delving into Class-Space Mapping for Zero-Shot Learning.

[4]Ridge Regression, Hubness, and Zero-Shot Learning.

[5]Zero-Shot Visual Recognition using Semantics-Preserving Adversarial Embedding Network.

[6]Zero-Shot Learning via Class-Conditioned Deep Generative Models.

[7]Semantic Autoencoder for Zero-Shot Learning.

[8]Zero-Shot Recognition using Dual Visual-Semantic Mapping Paths.

[9]An Empirical Study and Analysis of Generalized Zero-Shot Learning for Object Recognition in the Wild.

[10]An embarrassingly simple approach to zero-shot learning

[11]Zero-shot recognition using dual visualsemantic mapping paths

[12]Predicting visual exemplars of unseen classes for zero-shot learning

[13]Preserving Semantic Relations for Zero-Shot Learning

[14]Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly

[15]Recent Advances in Zero-shot Recognition

[16][http://people.duke.edu/~ww107/material/ZSL.pdf](https://link.zhihu.com/?target=http%3A//people.duke.edu/~ww107/material/ZSL.pdf)

[17]Attribute-Based Synthetic Network (ABS-Net): Learning More From Pseudo Feature Representation

#### General 

假设小暗（纯粹因为不想用小明）和爸爸，到了动物园，看到了马，然后爸爸告诉他，这就是马；之后，又看到了老虎，告诉他：“看，这种身上有条纹的动物就是老虎。”；最后，又带他去看了熊猫，对他说：“你看这熊猫是黑白色的。”然后，爸爸给小暗安排了一个任务，让他在动物园里找一种他从没见过的动物，叫斑马，并告诉了小暗有关于斑马的信息：“斑马有着马的轮廓，身上有像老虎一样的条纹，而且它像熊猫一样是黑白色的。”最后，小暗根据爸爸的提示，在动物园里找到了斑马（意料之中的结局。。。）。

ZSL就是希望我们的模型能够对其从没见过的类别进行分类，让机器具有推理能力，实现真正的智 能。其中零次 (Zero-shot) 是指对于要分类的类别对象，一次也不学习。这样的能力听上去很具 有吸引力，那么到底是怎么实现的呢?
假设我们的模型已经能够识别马，老虎和熊猫了，现在需要该模型也识别玟马，那么我们需要像爸 爸一样告诉模型，怎样的对象才是斑马，但是并不能直接让模型看见斑马。所以模型需要知道的信 息是马的样本、老虎的样本、熊猫的样本和样本的标签，以及关于前三种动物和斑马的描述。将其 转换为常规的机器学习，这里我们只讨论一般的图片分类问题:
(1) 训练集数据 $X_{t r}$ 及其标签 $Y_{t r}$ ，包含了模型需要学习的类别 (马、老虎和熊猫)，这里和 传统的监督学习中的定义一致；
(2) 测试集数据 $X_{t e}$ 及其标签 $Y_{t e}$ ，包含了模型需要辨识的类别（玟马），这里和传统的监督 学习中也定义一直;
(3) 训练集类别的描述 $A_{t r}$ ，以及测试集类别的描述 $A_{t e}$ ；我们将每一个类别 $y_i \in Y$ ，都 表示成一个语义向量 $a_i \in A$ 的形式，而这个语义向量的每一个维度都表示一种高级的属性，比 如“黑白色”、“有尾巴”、“有羽毛”等等，当这个类别包含这种属性时，那在其维度上被设置为非零 值。对于一个数据集来说，语义向量的维度是固定的，它包含了能够较充分描述数据集中类别的属 性。

.![img](/assets/BERTGPTDiffusion%20Research.assets/v2-d8efa9870a3ce5ee028277ec57033036_b.png)

在ZSL中，我们希望利用 $X_{t r}$ 和 $Y_{t r}$ 来训练模型，而模型能够具有识别 $X_{t e}$ 的能力，因此模型 需要知道所有类别的描述 $A_{t r}$ 和 $A_{t e}$ 。ZSL这样的设置其实就是上文中小暗识别玩的的过程 中，爸爸为他提供的条件。

![img](/assets/BERTGPTDiffusion%20Research.assets/v2-33a9764b792911eedce07dd4974e46f5_b.png)

我们面对的是一个图片分类问题，即对测 试集的样本 $X_{t e}$ 进行分类，而我们分类时需要借助类别的描述 $A$ ，由于每一个类别 $y_i \in Y$ ， 都对应一个语义向量 $a_i \in A$ ，因此我们现在可以忘掉 $Y$ ，直接使用 $A$ 。我们把 $X$ (利用深 度网络提取的图片特征，比如GoogleNet提取为1024维) 称为特征空间 (visual feature space)， 把类别的语义表示 $A$ ，称为语义空间。我们要做的，其实就是建立特征空间与语义空间之间的映 射。
对于分类，我们能想到的最简单的形式就是岭回归 (ridge regression)，俗称均方误差加范数约 束，具体形式为:
$$
\min \left\|X_{t r} W-A_{t r}\right\|^2+\eta \Omega(W)
$$
其中， $\Omega()$ 通常为 2 范数约束， $\eta$ 为超参，对 $W$ 求导，并让导为 0 ，即可求出 $W$ 的值。测试 时，利用 $W$ 将 $x_i \in X_{t e}$ 投影到语义空间中，并在该空间中寻找到离它最近的 $a_i \in A_{t e}$ ，则样本的类别为 $a_i$ 所对应的标签 $y_i \in Y_{t r}$ 。

#### ZSL中存在的问题

**领域漂移问题（domain shift problem）**

该问题的正式定义首先由[2]提出。简单来说，就是同一种属性，在不同的类别中，视觉特征的表现可能很大。如图3所示，斑马和猪都有尾巴，因此在它的类别语义表示中，“有尾巴”这一项都是非0值，但是两者尾巴的视觉特征却相差很远。如果斑马是训练集，而猪是测试集，那么利用斑马训练出来的模型，则很难正确地对猪进行分类。

![img](/assets/BERTGPTDiffusion%20Research.assets/v2-733de891fa7f478740b35228dad776c2_b.png)

> 由于样本的特征维度往往比语义的维度大，所以建立从 X 到 S 的映射往往会丢失信息，为了保留更多的信息，保持更多的丰富性，最流行的做法是将映射到语义空间中的样本，再重建回去，这样学习到的映射就能够得到保留更多的信息。因此，在原来简单岭回归[1]的基础上，可以将目标函数改为：[7]
> $$
> \min \left\|X_{t r}-W^T A_{t r}\right\|^2+\lambda\left\|W X_{t r}-A_{t r}\right\|^2
> $$
> 从目标函数可以看出，这其实完成的是一个简易的自编码器过程，我们简称这个算法为SAE
>
> [^]: zphilip48:是倒过来从语义到图片vector做一次训练么？
> [###4.2.2]: 
> [###4.2.2 Tractable]: 
> [#4.Diffusion Models]: 
> [#Diffusion Models]: 
> [#1]: 
> [# 4.2.2 Tractable]: 
> [# Tractable]: 

**枢纽点问题（Hubness problem）**

这其实是高维空间中固有的问题：在高维空间中，某些点会成为大多数点的最近邻点。这听上去有些反直观，细节方面可以参考[3]。由于ZSL在计算最终的正确率时，使用的是K-NN，所以会受到hubness problem的影响，并且[4]中，证明了基于岭回归的方法会加重hubness problem问题。

> 目前对于枢纽点问题的解决主要有两种方法：
>
> a. 如果模型建立的方式为岭回归，那么可以建立从语义空间到特征空间的映射，从而不加深hubness problem对结果的影响[4]，也就是说将目标函数（1）改为：
> $$
> \min \left\|X_{t r}-A_{t r} W\right\|^2+\eta \Omega(W)
> $$
> b.可以使用生成模型，比如自编码器、GAN等，生成测试集的样本，这样就变成了一个传统的监督分类问题，不存在K-NN的操作，所以不存在hubness problem的影响。

**语义间隔（semantic gap）**

样本的特征往往是视觉特征，比如用深度网络提取到的特征，而语义表示却是非视觉的，这直接反应到数据上其实就是：样本在特征空间中所构成的流型与语义空间中类别构成的流型是不一致的。（如图4所示）这使得直接学习两者之间的映射变得困难。

![img](/assets/BERTGPTDiffusion%20Research.assets/v2-869ec7e6e0f91229f8f66997ce59123a_b.png)

> 语义间隔问题的本质是二者的流形结构不一致，因此，解决此问题的着手点就在于将两者的流形调整到一致，再学习两者之间的映射[8]。最简单的方法自然是将类别的语义表示调整到样本的流型上，即用类别语义表示的K近邻样本点，重新表示类别语义即可。

## *4.16 Latent diffusion model* (**LDM**; [Rombach & Blattmann, et al. 2022](https://arxiv.org/abs/2112.10752))

runs the diffusion process in the latent space instead of pixel space, making training cost lower and inference speed faster. It is motivated by the observation that most bits of an image contribute to perceptual details and the semantic and conceptual composition still remains after aggressive compression. LDM loosely decomposes the perceptual compression and semantic compression with generative modeling learning by first trimming off pixel-level redundancy with autoencoder and then manipulate/generate semantic concepts with diffusion process on learned latent.



# 5 Introducing BART

refer to [Introducing BART | TensorGoose (sshleifer.github.io)](https://sshleifer.github.io/blog_v2/jupyter/2020/03/12/bart.html#Encoder-Decoder)

### Overview

For the past few weeks, I worked on integrating BART into [transformers](https://github.com/huggingface/transformers/). This post covers the high-level differences between BART and its predecessors and how to use the new `BartForConditionalGeneration` to summarize documents. Leave a comment below if you have any questions!

### Background: Seq2Seq Pretraining

In October 2019, teams from Google and Facebook published new transformer papers: [T5](https://arxiv.org/abs/1910.10683) and [BART](https://arxiv.org/abs/1910.13461). Both papers achieved better downstream performance on generation tasks, like abstractive summarization and dialogue, with two changes:

- add **a causal decoder to BERT's bidirectional encoder architecture**
- replace BERT's fill-in-the blank cloze task with a more complicated mix of pretraining tasks

#### Bert vs. GPT2

As the BART authors write,

> (BART) can be seen as **generalizing Bert (due to the bidirectional encoder) and GPT2 (with the left to right decoder).**

Bert is pretrained to try to **==predict masked tokens, and uses the whole sequence to get enough info to make a good guess==**. This is good for tasks where the prediction at position `i` is allowed to utilize information from positions after `i`, but less useful for tasks, like text generation, where the prediction for position `i` can only depend on previously generated words.

**In code, the idea of "what information can be used use when predicting the token at position `i`" is controlled by an argument called `attention_mask`[1](https://sshleifer.github.io/blog_v2/jupyter/2020/03/12/bart.html#fn-2). A value of 1 in the attention mask means that the model can use information for the column's word when predicting the row's word.**

Here is Bert's "Fully-visible"[2](https://sshleifer.github.io/blog_v2/jupyter/2020/03/12/bart.html#fn-3) `attention_mask`:

![img](/assets/BERTGPTDiffusion%20Research.assets/diagram_bert_v5.png)

1. the same parameter that is used to make model predictions invariant to pad tokens.[↩](https://sshleifer.github.io/blog_v2/jupyter/2020/03/12/bart.html#fnref-2)
2. "Fully-Visible" and "bidirectional" are used interchangeably. Same with "causal" and "autoregressive".[↩](https://sshleifer.github.io/blog_v2/jupyter/2020/03/12/bart.html#fnref-3)

**==GPT2, meanwhile, is pretrained to predict the next word using a causal mask, and is more effective for generation tasks==**, but less effective on downstream tasks where the whole input yields information for the output.

Here is the `attention_mask` for GPT2:

![img](/assets/BERTGPTDiffusion%20Research.assets/diagram_bartpost_gpt2.jpg)

The prediction for "eating", only utilizes previous words: "`<BOS>` I love".

#### Encoder-Decoder ( BART = BERT + GPT)

Our new friends, like BART, get the best of both worlds.

The encoder's `attention_mask` is fully visible, like BERT:![img](/assets/BERTGPTDiffusion%20Research.assets/seq2seq_enc_v5.png)

The decoder's `attention_mask` is causal, like GPT2:

![img](/assets/BERTGPTDiffusion%20Research.assets/seq2seq_dec.png)

**The encoder and decoder are connected by cross-attention**, **==where each decoder layer performs attention over the final hidden state of the encoder output==**. This presumably nudges the models towards generating output that is closely connected to the original input.

> # [What are the inputs to the first decoder layer](https://datascience.stackexchange.com/questions/88981/what-are-the-inputs-to-the-first-decoder-layer-in-a-transformer-model-during-the)
>
> Following your example:
>
> - The source sequence would be `How` `are` `you` `<EOS>`
> - The input to the encoder would be `How` `are` `you` `<EOS>`. Note that there is no `<start>` token here.
> - The target sequence would be `I` `am` `fine` `<EOS>` . The output of the decoder will be compared against this in the training.
> - The input to the decoder would be `<start>` `I` `am` `fine` .
>
> Notice that the input to the decoder is the target sequence shifted one position to the right by the token that signals the beginning of the sentence. The logic of this is that the output at each position should receive the previous tokens (and not the token at the same position, of course), which is achieved with this shift together with the self-attention mask.

> # why does the paper say this maximum sentence length thing
>
> going through the seq2seq-translation tutorial on pytorch (https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#sphx-glr-intermediate-seq2seq-translation-tutorial-py) and found the following sentence:
> Because there are sentences of all sizes in the training data, to actually create and train this layer we have to choose a maximum sentence length (input length, for encoder outputs) that it can apply to. Sentences of the maximum length will use all the attention weights, while shorter sentences will only use the first few.
> which didn't really make sense to me. My understanding of attention is that attention is computed as follows (according to the Pointer Network paper) at time step $t$ :
> $$
> \begin{gathered}
> u^{<t, j>}=v^{\top} \tanh \left(W_1 e_j+W_2 d_t\right)=N N_u\left(e_j, d_t\right) \\
> \alpha^{<t, j>}=\operatorname{softmax}\left(u^{<t, j>}\right)=\frac{\exp \left(u^{<t, j>}\right)}{Z^{<t>}}=\frac{\exp \left(u^{<t, j>}\right)}{\sum_{k=1}^{T_x} \exp \left(u^{<t, k>}\right)} \\
> d_{<i+1>}^{\prime}=\sum_{j=1}^{T_x} \alpha^{<t, j>} e_j
> \end{gathered}
> $$
> which basically means that a specific attention weight is not dependent on the length of the encoder (i.e. the encoder can change size and the above equation won't be affected because $T_x$ can be variable size).
>
> - It is only an efficiency issue. In theory, the attention mechanism can work with arbitrarily long sequences. The reason is that batches must be padded to the same length.
>
> - The tutorial mentioned in the question appears to have the peculiar mechanism
>   $$
>   w_i \propto \exp \left(a_i^T v\right)
>   $$
>
>   Where $a_i$ is the $i$ th row of a learned weight matrix $A$. I say that it is peculiar because the weight on the $i$ th input element does not actually depend on any of the $u_i$ at all! In fact we can view this mechanism as attention over word slots -- how much attention to put to the first word, the second word, third word etc, which does not pay any attention to which words are occupying which slots.
>
>   Since $A$, a learned weight matrix, must be fixed in size, then the number of word slots must also be fixed, which means the input sequence length must be constant (shorter inputs can be padded). Of course this peculiar attention mechanism doesn't really make sense at all, so I wouldn't read too much into it.
>
> - ([time series - What are attention mechanisms exactly? - Cross Validated (stackexchange.com)](https://stats.stackexchange.com/questions/344508/what-are-attention-mechanisms-exactly/345441#345441)) Attention is a method for aggregating a set of vectors $v_i$ into just one vector, often via a lookup vector $\boldsymbol{u}$. Usually, $\boldsymbol{v}_{\boldsymbol{i}}$ is either the inputs to the model or the hidden states of previous time-steps, or the hidden states one level down (in the case of stacked LSTMs).
>
>   The result is often called the context vector $\boldsymbol{c}$, since it contains the context relevant to the current time-step.
>
>   This additional context vector $c$ is then fed into the RNN/LSTM as well (it can be simply concatenated with the original input). Therefore, the context can be used to help with prediction.
>
>   The simplest way to do this is to compute probability vector $p=\operatorname{softmax}\left(V^T u\right)$ and $c=\sum_i p_i v_i$ where $V$ is the concatenation of all previous $v_i$. A common lookup vector $u$ is the current hidden state $h_t$.
>
>   There are many variations on this, and you can make things as complicated as you want. For example, instead using $v_i^T u$ as the logits, one may choose $f\left(v_i, u\right)$ instead, where $f$ is an arbitrary neural network.
>
>   **A common attention mechanism for sequence-to-sequence models uses**
>   **$p=\operatorname{softmax}\left(q^T \tanh \left(W_1 v_i+W_2 h_t\right)\right)$, where $v$ are the hidden states of the encoder, and $h_t$ is the current hidden state of the decoder. $q$ and both $W \mathrm{~s}$ are parameters.**
>
>   Some papers which show off different variations on the attention idea:
>
>   - Pointer Networks use attention to reference inputs in order to solve combinatorial optimization problems.
>
>   - Recurrent Entity Networks maintain separate memory states for different entities (people/objects) while reading text, and update the correct memory state using attention.
>
>   - Transformer models also make extensive use of attention. Their formulation of attention is slightly more general and also involves key vectors $k_i$ : the attention weights $p$ are actually computed between the keys and the lookup, and the context is then constructed with the $\boldsymbol{v}_{\boldsymbol{i}}$.
>
>   Here is a quick implementation of one form of attention, although I can't guarantee correctness beyond the fact that it passed some simple tests.



> # The restriction in the maximum length
>
> ==The restriction in the maximum length== of the transformer input is due to the needed **amount of memory** to compute the self-attention over it.
>
> The amount of memory needed by the self-attention in the Transformer is **quadratic on the length of the input**. This means that increasing the maximum length of the input, increases drastically the needed memory for self-attention. The maximum length is that which makes the model use up the whole memory of the GPU for at least one sentence (once the other elements of the model are also taken into account, like the embeddings which take a lot of memory).
>
> [Transformer-XL](https://openreview.net/forum?id=HJePno0cYm) is certainly a way to take into account as much context as possible in language modeling (its role is analogous to truncated back-propagation through time in LSTM language models). However, the gradients are not propagated through the attention over the memory segment, only through the current segment.
>
> There have been several architectural attempts to reduce the amount of memory needed by transformers, like using [locality-constraints in the attention](https://openreview.net/forum?id=SkVhlh09tX) (**Dynamic Convolutions model**) or using [locality-sensitive hashing](https://openreview.net/forum?id=rkgNKkHtvB) (**Reformer model**).
>
> There have been other implementation attempts, like **gradient checkpointing**(e.g. [this](https://qywu.github.io/2019/05/22/explore-gradient-checkpointing.html)), which is a general technique to run computations that don't fit at once in the GPU memory

> ## Cross-attention Algorithm
>
> - Let us have embeddings (token) sequences S1 and S2
> - Calculate Key and Value from sequence S1
> - Calculate Queries from sequence S2
> - Calculate [attention matrix](https://vaclavkosar.com/ml/transformers-self-attention-mechanism-simplified) from Keys and Queries
> - Apply queries to the attention matrix
> - Output sequence has dimension and length of sequence S2
>
> $$
> \text { In an equation: } \operatorname{sof} \operatorname{tmax}\left(\left(W_Q S_2\right)\left(W_K S_1\right)^{\top}\right) W_V S_1
> $$
>
> 
>
> ![cross-attention perceiver io detail](/assets/BERTGPTDiffusion%20Research.assets/cross-attention-detail-perceiver-io.png)
>
> ## Cross-attention vs Self-attention
>
> Except for inputs, cross-attention calculation is the same as [self-attention](https://vaclavkosar.com/ml/transformers-self-attention-mechanism-simplified). Cross-attention combines asymmetrically two separate embedding sequences of same dimension, in contrast self-attention input is a single embedding sequence. One of the sequences serves as a query input, while the other as a key and value inputs. Alternative [cross-attention in SelfDoc](https://vaclavkosar.com/ml/cross-attention-in-transformer-architecture#cross-attention-in-selfdoc), uses query and value from one sequence, and key from the other.
>
> [The feed forward layer](https://vaclavkosar.com/ml/Feed-Forward-Self-Attendion-Key-Value-Memory) is related to cross-attention, except the feed forward layer does use softmax and one of the input sequences is static. [Augmenting Self-attention with Persistent Memory paper](https://vaclavkosar.com/ml/Feed-Forward-Self-Attendion-Key-Value-Memory) shows that Feed Forward layer calculation made the same as self-attention.
>
> ### Cross-Attention in Transformer Decoder
>
> [Cross-Attention in Transformer Architecture (vaclavkosar.com)](https://vaclavkosar.com/ml/cross-attention-in-transformer-architecture)
>
> Cross-attention is widely used in [encoder-decoder](https://vaclavkosar.com/ml/Encoder-only-Decoder-only-vs-Encoder-Decoder-Transfomer) or multi-modality use cases.
>
> Cross-attention was described in the [Transformer](https://vaclavkosar.com/ml/transformers-self-attention-mechanism-simplified) paper, but it was not given this name yet. Transformer decoding starts with full input sequence, but empty decoding sequence. Cross-attention introduces information from the input sequence to the layers of the decoder, such that it can predict the next output sequence token. The [decoder](https://vaclavkosar.com/ml/Encoder-only-Decoder-only-vs-Encoder-Decoder-Transfomer) then adds the token to the output sequence, and repeats this autoregressive process until the EOS token is generated.
>
> ![Cross-Attention in the Transformer decoder of Attention is All You Need paper](/assets/BERTGPTDiffusion%20Research.assets/cross-attention-in-transformer-decoder.png)
>
> ![img](/assets/BERTGPTDiffusion%20Research.assets/15xN9xmT4QPua9Cpd4fjqCw.png)
>
> ## Cross-attention Implementation
>
> Have a look at [CrossAttention implementation](https://github.com/huggingface/diffusers/blob/4125756e88e82370c197fecf28e9f0b4d7eee6c3/src/diffusers/models/cross_attention.py) in Diffusers library, which can generate images with **Stable Diffusion**. In this case the cross-attention is used to **condition transformers inside a UNet layer with a text prompt for image generation**. The constructor shows, how we can also have **different dimensions** and if you step through with a debugger, you will also see the **different sequence length between the two modalities** .
>
> ```python
> class CrossAttention(nn.Module):
>     r"""
>     A cross attention layer.
> 
>     Parameters:
>         query_dim (`int`): The number of channels in the query.
>         cross_attention_dim (`int`, *optional*):
>             The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
>         heads (`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
>         dim_head (`int`,  *optional*, defaults to 64): The number of channels in each head.
>         dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
>         bias (`bool`, *optional*, defaults to False):
>             Set to `True` for the query, key, and value linear layers to contain a bias parameter.
>     """
>         query = attn.to_q(hidden_states)
>         query = attn.head_to_batch_dim(query)
> 
>         encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
>         key = attn.to_k(encoder_hidden_states)
>         value = attn.to_v(encoder_hidden_states)
>         key = attn.head_to_batch_dim(key)
>         value = attn.head_to_batch_dim(value)
> 
>         attention_probs = attn.get_attention_scores(query, key, attention_mask)
>         hidden_states = torch.bmm(attention_probs, value)
> ```
>
> ### Cross-Attention in Stable Diffusion
>
> Stable Diffusion uses cross-attention **for image generation to condition transformers with a text prompt** inside the denoising U-Net layer.
>
> [![stable diffusion architecture with cross-attention](https://vaclavkosar.com/images/stable-diffusion-architecture.png)stable diffusion architecture with cross-attention](https://vaclavkosar.com/images/stable-diffusion-architecture.png)
>
> 
>
> ### Cross-Attention in Perceiver IO
>
> [Perceiver IO](https://arxiv.org/pdf/2107.14795.pdf) is a general-purpose multi-modal architecture that can handle wide variety of inputs as well as outputs. Perceiver can be applied to for example [image-text classification](https://vaclavkosar.com/ml/Multimodal-Image-Text-Classification). Perceiver IO uses [cross-attention](https://vaclavkosar.com/ml/cross-attention-in-transformer-architecture) for merging:
>
> - multimodal input sequences (e.g. image, text, audio) into a low dimensional latent sequence
> - “output query” or “command” to decode the output value e.g. predict this masked word
>
> [![Perceiver IO architecture](https://vaclavkosar.com/images/cross-attention-perceiver-io.png)Perceiver IO architecture](https://vaclavkosar.com/images/cross-attention-perceiver-io.png)
>
> 
>
> Advantage of the Perceiver architecture is that in general you can work with very large inputs. Architecture [Hierarchical Perceiver](https://arxiv.org/pdf/2202.10890.pdf) has ability to process even longer input sequences by splitting into subsequences and then merging them. Hierarchical Perceiver also learns the positional encodings with a separate training step with a reconstruction loss.
>
> ### Cross-Attention in SelfDoc
>
> [![selfdoc cross-attention](https://vaclavkosar.com/images/selfdoc-cross-attention.png)selfdoc cross-attention](https://vaclavkosar.com/images/selfdoc-cross-attention.png)
>
> 
>
> In [Selfdoc](https://arxiv.org/pdf/2106.03331.pdf), cross-attention is integrated in a special way. First step of their Cross-Modality Encoder, instead uses value and query from sequence A and then key from the sequence B.
>
> ### Other Cross-Attention Examples
>
> - [DeepMind’s RETRO Transformer uses cross-attention to incorporate the database retrived sequences](https://vaclavkosar.com/ml/DeepMinds-RETRO-Transformer-Model)
> - [Code example: HuggingFace BERT (key, value are from the encoder, while query is from the decoder)](https://github.com/huggingface/transformers/blob/198c335d219a5eb4d3f124fdd1ce1a9cd9f78a9b/src/transformers/models/bert/modeling_bert.py#L268)
> - [CrossVit - here only simplified cross-attention is used](https://arxiv.org/pdf/2103.14899.pdf)
> - [On the Strengths of Cross-Attention in Pretrained Transformers for Machine Translation](https://arxiv.org/pdf/2104.08771v1.pdf)

#### Pretraining: Fill In the Span

Bart and T5 are both pretrained[1](https://sshleifer.github.io/blog_v2/jupyter/2020/03/12/bart.html#fn-5) on tasks where **spans** of text are replaced by masked tokens. The model must learn to reconstruct the original document. Figure 1 from the BART paper explains it well:

![img](/assets/BERTGPTDiffusion%20Research.assets/text_infilling.png)In this example, the original document is A B C D E. the span `[C, D]` is masked before encoding and an extra mask is inserted before B, leaving the corrupted document `'A _ B _ E'` as input to the encoder.

==The decoder (autogressive means "uses a causal mask") must reconstruct the original document, using the encoder's output and previous uncorrupted tokens.==

------

1. This is a bit of a simplification. Both papers experiment with many different pretraining tasks, and find that this one performs well. T5 uses a "replace corrupted spans" task. Instead of putting masks, they put in a random token.[↩](https://sshleifer.github.io/blog_v2/jupyter/2020/03/12/bart.html#fnref-5)

### Summarization

In summarization tasks, the `input` sequence is the document we want to summarize, and the `output` sequence is a ground truth summary. Seq2Seq archictectures can be directly finetuned on summarization tasks, without any new randomly initialized heads. The pretraining task is also a good match for the downstream task. In both settings, the input document must be copied from the input with modification. The numbers confirm this: all the new fancy Seq2Seq models do a lot better than the old less-fancy guys on the CNN/Daily Mail abstractive summarization task, and BART does especially well.

| Model          | Rouge2 | Model Size | Pretraining |
| :------------- | -----: | :--------- | :---------- |
| PT-Gen         |  17.28 | 22 M       | None        |
| TransformerAbs |  17.76 | 200M       | None        |
| BertSumABS     |  19.39 | 220 M      | Encoder     |
| UniLM          |   20.3 | 340 M      | Seq2Seq     |
| T5-base        |  20.34 | 770 M      | Seq2Seq     |
| Bart           |  21.28 | 406 M      | Seq2Seq     |
| T5-11B         |  21.55 | 11 B       | Seq2Seq     |

- `BertSumABS` (from [*Text Summarization with Pretrained Encoders*](https://arxiv.org/abs/1908.08345), uses a Seq2Seq architecture but doesn't pretrain the decoder. `TransformerAbs`, from the same paper, uses a slightly smaller model and no pretraining.
- `PT-Gen` is from [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/pdf/1704.04368.pdf)
- [UniLM](https://arxiv.org/abs/1905.03197) is a "Prefix-LM" with a similar masking strategy to Bart and T5.

# 6 Language Modeling

## Causal Language Modeling (CLM):

- Implementation: In CLM, the model is trained to predict the next token in the sequence, given the previous tokens. During training, the input tokens are fed into the model, and the model predicts the probability distribution of the next token. The loss is calculated based on the model’s predictions and the actual target tokens, which are just the input tokens shifted by one position.
- Architecture: CLM is typically used with autoregressive models like GPT. These models use a unidirectional (left-to-right) Transformer architecture, where each token can only attend to the tokens that come before it. This prevents the model from “cheating” by attending to the target tokens during training.
- Output Model: A fine-tuned CLM model can generate coherent text by predicting one token at a time, making it suitable for text generation tasks. However, it may not be as effective at capturing bidirectional context compared to MLM models.

## Masked Language Modeling (MLM):

- Implementation: In MLM, the model is trained to predict masked tokens within the input sequence. During preprocessing, a certain percentage of tokens are randomly masked, and the model is trained to predict the original tokens at those masked positions. The loss is calculated based on the model’s predictions and the actual target tokens (the original tokens that were masked).
- Architecture: MLM is used with models like BERT, which use a bidirectional Transformer architecture. Unlike CLM models, MLM models can attend to all tokens in the input sequence during training, allowing them to capture context from both left and right.
- Output Model: A fine-tuned MLM model is better at understanding context and relationships between words in a sequence, making it suitable for tasks like text classification, sentiment analysis, named entity recognition, or question answering.

## Sequence-to-Sequence (seq2seq) Modeling:

- Implementation: In seq2seq modeling, the model is trained to generate output sequences based on input sequences. The model consists of two parts: an encoder that encodes the input sequence into a latent representation, and a decoder that generates the output sequence based on this latent representation. The loss is calculated based on the model’s predictions and the actual target output tokens.
- Architecture: Seq2seq models typically use an encoder-decoder architecture, where both the encoder and decoder can be based on the Transformer architecture (e.g., T5, BART) or other architectures like LSTMs (e.g., the original seq2seq model). The encoder processes the input sequence and generates a context representation, while the decoder generates the output sequence based on the encoder’s output and its own hidden state.
- Output Model: A fine-tuned seq2seq model is better at tasks where the model needs to generate coherent output text based on input text, such as summarization, translation, or question answering.

# 7. GAN Model

[能量视角下的GAN模型（二）：GAN＝“分析”＋“采样” - 科学空间|Scientific Spaces (kexue.fm)](https://kexue.fm/archives/6331)

## 7.1. Drag Your GAN理解**项目地址**：https://github.com/Zeqiang-Lai/DragGAN
**论文地址**：https://vcai.mpi-inf.mpg.de/projects/DragGAN/
**代码地址**：https://github.com/XingangPan/DragGAN

[Drag Your GAN理解 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/632113718)

Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold是一篇在SIGGRAPH 2023会议上发表的论文¹，介绍了一种基于GAN的图像变形方法，可以让用户通过拖动图像上的任意点来精确地控制生成对象的姿态、形状、表情和布局²。该方法包括两个主要组成部分：

1) 基于特征的运动监督，用于驱动手柄点向目标位置移动；
2)  一种新的点跟踪方法，利用GAN的判别特征来持续定位手柄点的位置¹。该方法可以应用于多种类别的图像，如动物、汽车、人类、风景等，即使在遮挡内容和形状变形等困难场景下，也能生成逼真的输出²。该方法还可以通过GAN反演来操作真实图像¹。该方法的代码已经开源在GitHub上

Drag Your GAN的输入数据是一个图像和两个点，分别表示手柄点和目标点。输出数据是一个变形后的图像，使得手柄点移动到目标点的位置。损失函数由两部分组成，一部分是判别损失，用于衡量生成图像的真实性和质量，另一部分是运动损失，用于衡量手柄点和目标点之间的距离



Drag Your GAN的实现步骤大致如下：

1. 首先，需要选择一个预训练好的GAN模型，如StyleGAN2¹，并将其转换为PyTorch版本。
2. 然后，需要为GAN模型定义一个特征提取器，用于从生成器的中间层提取特征图。特征提取器可以是一个简单的卷积层或者一个更复杂的网络。
3. 接着，需要为GAN模型定义一个判别器，用于判断生成图像的真实性和质量。判别器可以是一个简单的全连接层或者一个更复杂的网络。
4. 然后，需要为GAN模型定义一个点跟踪器，用于在特征图上定位手柄点的位置。点跟踪器可以是一个简单的卷积层或者一个更复杂的网络。
5. 接着，需要为GAN模型定义一个运动监督器，用于计算手柄点在特征图上的运动向量，并将其加到生成器的输入向量上。运动监督器可以是一个简单的全连接层或者一个更复杂的网络。
6. 然后，需要为GAN模型定义一个优化器，用于更新生成器的输入向量，并最小化判别器的损失和运动监督器的损失。优化器可以是Adam或者其他梯度下降方法。
7. 最后，需要为GAN模型定义一个用户交互界面，用于在图像上选择手柄点和目标点，并触发点跟踪器、运动监督器和优化器来实现图像变形¹。用户交互界面可以是Gradio或者其他可视化工具。

[调转到标题1]: 

[maximum this likelihood]: 
