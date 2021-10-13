# KAIST AI605 Assignment 1: Text Classification

## Environment

You will only use Python 3.7 and PyTorch 1.9, which is already available on Colab:

```python
from platform import python_version
from collections import Counter
import re
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"python {python_version()}")
print(f"torch {torch.__version__}")
```

> ```text
> python 3.7.12
> torch 1.9.0+cu111
> ```

## 1. Limitations of Vanilla RNNs

In Lecture 02, we saw that a multi-layer perceptron (MLP) without activation function is equivalent to a single linear transformation with respect to the inputs.
One can define a vanilla recurrent neural network without activation as, given inputs $\textbf{x}_1 \dots \textbf{x}_T$, the outputs $\textbf{h}_t$ is obtained by
$$\textbf{h}_t = \textbf{V}\textbf{h}_{t-1} + \textbf{U}\textbf{x}_t + \textbf{b},$$
where $\textbf{V}, \textbf{U}, \textbf{b}$ are trainable weights.

> **Problem 1.1** *(2 point)*
> Show that such recurrent neural network (RNN) without activation function is equivalent to a single linear transformation with respect to the inputs,
> which means each $\textbf{h}_t$ is a linear combination of the inputs.

> **Solution 1.1**
> Let's substitute $\textbf{h}_t$ with the fact that $\textbf{h}_{t-1} = \textbf{V}\textbf{h}_{t-2} + \textbf{U}\textbf{x}_{t-1} + \textbf{b}$.
> Then we get,
> $$
>     \begin{align*}
>         \textbf{h}_t
>         &= \textbf{V}\textbf{h}_{t-1} + \textbf{U}\textbf{x}_t + \textbf{b} \\
>         &= \textbf{V}(\textbf{V}\textbf{h}_{t-2} + \textbf{U}\textbf{x}_{t-1} + \textbf{b}) + \textbf{U}\textbf{x}_t + \textbf{b} \\
>         & \qquad\qquad\qquad\qquad\vdots \\
>         &= \textbf{V}(\textbf{V}( \cdots \textbf{V}(\textbf{U}\textbf{x}_{1} + \textbf{b}) + \textbf{U}\textbf{x}_{2} + \textbf{b}) \cdots) + \textbf{U}\textbf{x}_t + \textbf{b} \\
>         &= \textbf{V}^{t-1}\textbf{U}\textbf{x}_1 + \textbf{V}^{t-2}\textbf{U}\textbf{x}_2 + \cdots + \textbf{V}\textbf{U}\textbf{x}_{t-1} + \textbf{U}\textbf{x}_t + \left(\sum_{i=1}^t\textbf{V}^{t-1}\right)\textbf{b} \\
>         &= \textbf{W}_{t-1}\textbf{x}_1 + \textbf{W}_{t-2}\textbf{x}_2 + \cdots + \textbf{W}_1\textbf{x}_{t-1} + \textbf{W}_0\textbf{x}_t + \textbf{c}_t
>     \end{align*}
> $$
> where $\textbf{W}_t = \textbf{V}^t\textbf{U}$ and $\textbf{c}_t = \left(\sum_{i=1}^t\textbf{V}^{t-1}\right)\textbf{b}$.
> And this is the single linear transformation with respect to the inputs $\textbf{x}_1, \cdots \textbf{x}_t$.

In Lecture 05 and 06, we will see how RNNs can model non-linearity via activation function, but they still suffer from exploding or vanishing gradients. We can mathematically show that, if the recurrent relation is
$$ \textbf{h}_t = \sigma (\textbf{V}\textbf{h}_{t-1} + \textbf{U}\textbf{x}_t + \textbf{b}) $$
then
$$ \frac{\partial \textbf{h}_t}{\partial \textbf{h}_{t-1}} = \text{diag}(\sigma' (\textbf{V}\textbf{h}_{t-1} + \textbf{U}\textbf{x}_t + \textbf{b}))\textbf{V}$$
so
$$\frac{\partial \textbf{h}_T}{\partial \textbf{h}_1} \propto \textbf{V}^{T-1}$$
which means this term will be very close to zero if the norm of $\bf{V}$ is smaller than 1 and really big otherwise.

> **Problem 1.2** *(2 points)*
> Explain how exploding gradient can be mitigated if we use gradient clipping.

> **Solution 1.2**
> As given above, the gradient of $\textbf{h}_T$ with respect to $\textbf{h}_1$ is proportional to $\textbf{V}^{T-1}$, which is
> $$\textbf{g}_T = \frac{\partial \textbf{h}_T}{\partial \textbf{h}_1} \propto \textbf{V}^{T-1}.$$
> This means that if the norm of $\textbf{V}$ is larger than $1$, the gradient increases exponentially as $T$ increases, which leads to the exploding of the gradient.
> By using gradient clipping, the clipped gradient becomes
> $$
>     \hat{\textbf{g}}_T = \begin{cases}
>         c \times \frac{\textbf{g}_T}{\left\Vert\textbf{g}_T\right\Vert} & \text{ if } \left\Vert\textbf{g}_T\right\Vert > c \\
>         \textbf{g}_T & \text{otherwise}
>     \end{cases}
> $$
> where $c$ is the positive constant.
> Then the norm of the gradient is bounded by $c$, which is $\left\Vert\hat{\textbf{g}}_T\right\Vert \le c$.
> So, the gradient clipping upper bounds the norm of the gradient by $c$, which prevents the gradient from exploding.

> **Problem 1.3** *(2 points)*
> Explain how vanishing gradient can be mitigated if we use LSTM.
> See the Lecture 05 and 06 slides for the definition of LSTM.

> **Solution 1.3**
> The cell state of LSTM is
> $$\textbf{c}_t = \textbf{f}_t \circ \textbf{c}_{t-1} + \textbf{i}_t \circ \tilde{\textbf{c}}_t,$$
> the gradient of $\textbf{c}_T$ with respect to $\textbf{c}_1$ can be computed as follows:
> $$
>     \begin{align*}
>         \textbf{g}_T
>         &= \frac{\partial \textbf{c}_T}{\partial \textbf{c}_1} \\
>         &= \frac{\partial \textbf{c}_T}{\partial \textbf{c}_{T-1}} \cdot \frac{\partial \textbf{c}_{T-1}}{\partial \textbf{c}_{T-2}} \cdots \frac{\partial \textbf{c}_2}{\partial \textbf{c}_{1}} \\
>         &= \textbf{f}_T \cdot \textbf{f}_{T-1} \cdots \textbf{f}_{2} \\
>         &= \prod_{i=2}^T \textbf{f}_{i},
>     \end{align*}
> $$
> where $\textbf{f}_t = \sigma_g\left(\textbf{W}_f\textbf{x}_t + \textbf{U}_f\textbf{h}_{t-1} + \textbf{b}_f\right)$ and $\sigma_g$ is a sigmoid function.
> Since $\textbf{f}_t$ is the output of the sigmoid function, the value of each element is bounded by $(0, 1)$,
> and this means the gradient remains if each $\textbf{f}_t$ for $t = 2, \cdots, T$ is close to $1$.
> The difference from RNN is that the gradient of RNN is proportional to the product of the same matrix $\textbf{V}$ and the gradient of LSTM is the product of the different vectors $\textbf{f}_t$.
> Therefore, the gradient of RNN vanishes exponentially, whereas LSTM does not.

## 2. Creating Vocabulary from Training Data

Creating the vocabulary is the first step for every natural language processing model.
In this section, you will use Stanford Sentiment Treebank (SST), a popular dataset for sentiment classification, to create your vocabulary.

### Obtaining SST via Hugging Face

We will use `datasets` package offered by Hugging Face, which allows us to easily download various language datasets, including Stanford Sentiment Treebank.

First, install the package:

```python
try:
    import datasets
except ImportError:
    !pip install datasets
```

Then download SST and print the first example:

```python
from datasets import load_dataset

sst = load_dataset("sst")
train0 = sst["train"][0]
print(f"Sentence: \"{train0['sentence']}\"")
print(f"Label:    {train0['label']:.6f}")
```

> ```text
> No config specified, defaulting to: sst/default
> Reusing dataset sst (/root/.cache/huggingface/datasets/sst/default/1.0.0/b8a7889ef01c5d3ae8c379b84cc4080f8aad3ac2bc538701cbe0ac6416fb76ff)
> ```

> ```text
> Sentence: "The Rock is destined to be the 21st Century 's new `` Conan '' and that he 's going to make a splash even greater than Arnold Schwarzenegger , Jean-Claud Van Damme or Steven Segal ."
> Label:    0.694440
> ```

Note that each `label` is a score between 0 and 1. You will round it to either 0 or 1 for binary classification (positive for 1, negative for 0).
In this first example, the label is rounded to 1, meaning that the sentence is a positive review.
You will only use `sentence` as the input; please ignore other values.

> **Problem 2.1** *(2 points)*
> Using space tokenizer, create the vocabulary for the training data and report the vocabulary size here.
> Make sure that you add an `UNK` token to the vocabulary to account for words (during inference time) that you haven't seen.
> See below for an example with a short text.

```python
# Space tokenization
words = [word for data in sst["train"] for word in data["sentence"].split(" ")]
tokens = sorted(list(set(words)))  # to ensure reproducibility (set doesn't guarantee order)
print(f"Number of tokens: {len(tokens)}")

# Constructing vocabulary with `UNK`
vocab = ["PAD", "UNK"] + tokens
word2id = {word: word_id for word_id, word in enumerate(vocab)}
print(f"Number of vocabs: {len(vocab)}")
print(f"ID of 'star': {word2id['star']}")
```

> ```text
> Number of tokens: 18280
> Number of vocabs: 18282
> ID of 'star': 15904
> ```

> **Problem 2.2** *(1 point)*
> Using all words in the training data will make the vocabulary very big.
> Reduce its size by only including words that occur at least 2 times.
> How does the size of the vocabulary change?

```python
# Include words that occur at least 2 times
tokens = [token for token, count in Counter(words).items() if count >= 2]
print(f"Number of tokens: {len(tokens)}")

# Constructing vocabulary with `UNK`
vocab = ["PAD", "UNK"] + tokens
word2id = {word: word_id for word_id, word in enumerate(vocab)}
print(f"Number of vocabs: {len(vocab)}")
print(f"ID of 'star': {word2id['star']}")
```

> ```text
> Number of tokens: 8736
> Number of vocabs: 8738
> ID of 'star': 2308
> ```

## 3. Text Classification with Multi-Layer Perceptron and Recurrent Neural Network

You can now use the vocabulary constructed from the training data to create an embedding matrix.
You will use the embedding matrix to map each input sequence of tokens to a list of embedding vectors.
One of the simplest baseline is to fix the input length (with truncation of padding), flatten the word embeddings,
apply a linear transformation followed by an activation, and finally classify the output into the two classes:

```python
words = [word.lower() for data in sst["train"] for word in re.split(r"\W+", data["sentence"]) if word]
tokens = [token for token, count in Counter(words).items() if count >= 2]
vocab = ["pad", "unk"] + tokens
word2id = {word: word_id for word_id, word in enumerate(vocab)}
```

```python
def tokenize(sentence, length):
    tokens = [word for word in re.split(r"\W+", sentence.lower()) if word]
    tokens = (tokens + ["pad"] * (length - len(tokens)))[:length]
    return tokens

def binarize(label):
    return 1 if label > 0.5 else 0

def preprocess(sentences, labels, length):
    x = [tokenize(sentence, length) for sentence in sentences]
    y = [binarize(label) for label in labels]
    return x, y
```

Let's see the examples:

```python
length = 8
input_sentence = "What a nice day!"
input_tensor = torch.LongTensor([[
    word2id.get(token, 1) for token in tokenize(input_sentence, length)
]])  # the first dimension is minibatch size
print(f"{input_sentence} -> {input_tensor}")
```

> ```text
> What a nice day! -> tensor([[266,  18, 297, 591,   0,   0,   0,   0]])
> ```

```python
class ModelBase(nn.Sequential):
    def __init__(self, hiddens, dim, embed=None):
        super().__init__(*(([] if embed is None else [nn.Embedding(len(embed), dim)])
                         + (hiddens if type(hiddens) is list else [hiddens])
                         + [nn.ReLU(), nn.Linear(dim, 2)]))
```

```python
class MLPModel(ModelBase):
    def __init__(self, dim, length, embed=None):
        super().__init__([
            nn.Flatten(),
            nn.Linear(dim * length, dim),
        ], dim, embed)
```

```python
torch.manual_seed(19)
baseline = MLPModel(dim=3, length=length, embed=vocab)  # dim is usually bigger, e.g. 128
logits = baseline(input_tensor)
print(f"softmax logits: {F.softmax(logits, dim=1).detach()}")  # probability for each class
```

> ```text
> softmax logits: tensor([[0.6985, 0.3015]])
> ```

Now we will compute the loss, which is the negative log probability of the input text's label being the target label (`1`),
which in fact turns out to be equivalent to the [cross entropy](https://en.wikipedia.org/wiki/Cross_entropy)
between the probability distribution and a one-hot distribution of the target label
(note that we use `logits` instead of `softmax(logits)` as the input to the cross entropy, which allow us to avoid numerical instability).

```python
ce = nn.CrossEntropyLoss()
label = torch.LongTensor([1])  # The ground truth label for "What a nice day!" is positive.
loss = ce(logits, label)  # Loss, a.k.a L
print(f"loss: {loss.detach():.6f}")
```

> ```text
> loss: 1.198835
> ```

Once we have the loss defined, only one step remains! We compute the gradients of parameters with respective to the loss and update.
Fortunately, PyTorch does this for us in a very convenient way.
Note that we used only one example to update the model, which is basically a Stochastic Gradient Descent (SGD) with minibatch size of 1.
A recommended minibatch size in this exercise is at least 16.
It is also recommended that you reuse your training data at least 10 times (i.e. 10 *epochs*).

```python
optimizer = torch.optim.SGD(baseline.parameters(), lr=0.1)
optimizer.zero_grad()  # reset process
loss.backward()  # compute gradients
optimizer.step()  # update parameters
```

Once you have done this, all weight parameters will have `grad` attributes that contain their gradients with respect to the loss.

```python
print(baseline[2].weight.grad)
```

> ```text
> tensor([[-0.0000, -0.0000, -0.0000,  0.0000, -0.0000, -0.0000,  0.0000,  0.0000,
>           0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -0.0000, -0.0000,  0.0000,
>          -0.0000, -0.0000,  0.0000, -0.0000, -0.0000,  0.0000, -0.0000, -0.0000],
>         [-0.0022, -0.0209, -0.0229,  0.0020, -0.0056, -0.0060,  0.0074,  0.0346,
>           0.0035,  0.0151,  0.0065,  0.0169,  0.0153, -0.0066, -0.0034,  0.0153,
>          -0.0066, -0.0034,  0.0153, -0.0066, -0.0034,  0.0153, -0.0066, -0.0034],
>         [-0.0000, -0.0000, -0.0000,  0.0000, -0.0000, -0.0000,  0.0000,  0.0000,
>           0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -0.0000, -0.0000,  0.0000,
>          -0.0000, -0.0000,  0.0000, -0.0000, -0.0000,  0.0000, -0.0000, -0.0000]])
> ```

Now, define the helper class to train the model and get the data loaders

```python
class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, dataloader):
        self.model.train()
        moving_loss = None
        for x_batch, y_batch in dataloader:
            logits = self.model(x_batch)
            loss = self.criterion(logits, y_batch)
            moving_loss = (loss.detach() if moving_loss is None else
                           0.2 * moving_loss + 0.8 * loss.detach())
            self.optimizer.zero_grad()  # reset process
            loss.backward()  # compute gradients
            self.optimizer.step()  # update parameters
        return moving_loss

    def test_epoch(self, dataloader):
        with torch.no_grad():
            self.model.eval()
            correct, total = 0, 0
            for x_batch, y_batch in dataloader:
                logits = self.model(x_batch)
                y_pred = torch.argmax(logits, dim=1)
                correct += torch.sum(y_pred == y_batch).item()
                total += len(y_batch)
            return correct * 100 / total

    def train(self, train_loader, valid_loader, epochs=10, print_every=10):
        print(f"{self.model.__class__.__name__}")
        best_acc = 0.
        for i in range(epochs):
            train_loss = self.train_epoch(train_loader)
            if (i + 1) % print_every == 0:
                train_acc = self.test_epoch(train_loader)
                valid_acc = self.test_epoch(valid_loader)
                print(f"Epoch {i+1:3d}: Train Loss {train_loss:.6f}, "
                      f"Train Acc: {train_acc:.2f}, Valid Acc: {valid_acc:.2f}")
                best_acc = max(best_acc, valid_acc)
        print(f"Best Valid Acc: {best_acc:.2f}%")
```

```python
def get_dataloader(x, y, x_trans, y_trans, batch_size, shuffle):
    dataset = TensorDataset(torch.vstack([x_trans(x_item) for x_item in x]),
                            torch.cat([y_trans(y_item) for y_item in y]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
```

```python
length = 32
batch_size = 128

unk_id = word2id["unk"]
x2long = lambda x_item: torch.LongTensor([word2id.get(token, unk_id) for token in x_item]).to(device)
y2long = lambda y_item: torch.LongTensor([y_item]).to(device)

sst_train, sst_valid = sst["train"], sst["validation"]

x_train, y_train = preprocess(sst_train["sentence"], sst_train["label"], length)
x_valid, y_valid = preprocess(sst_valid["sentence"], sst_valid["label"], length)

train_loader = get_dataloader(x_train, y_train, x2long, y2long, batch_size, shuffle=True)
valid_loader = get_dataloader(x_valid, y_valid, x2long, y2long, batch_size, shuffle=False)
```

> **Problem 3.1** *(2 points)*
> Properly train a MLP baseline model on SST and report the model's accuracy on the dev data.

```python
torch.manual_seed(19)
model = MLPModel(dim=64, length=length, embed=vocab).to(device)
Trainer(
    model, torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=7e-4),
).train(train_loader, valid_loader, epochs=100)
```

> ```text
> MLPModel
> Epoch  10: Train Loss 0.322388, Train Acc: 89.30, Valid Acc: 55.77
> Epoch  20: Train Loss 0.165132, Train Acc: 97.34, Valid Acc: 57.95
> Epoch  30: Train Loss 0.106883, Train Acc: 98.60, Valid Acc: 58.67
> Epoch  40: Train Loss 0.059070, Train Acc: 99.88, Valid Acc: 60.85
> Epoch  50: Train Loss 0.037730, Train Acc: 99.73, Valid Acc: 62.31
> Epoch  60: Train Loss 0.024387, Train Acc: 99.95, Valid Acc: 65.49
> Epoch  70: Train Loss 0.035860, Train Acc: 99.59, Valid Acc: 63.94
> Epoch  80: Train Loss 0.022126, Train Acc: 99.98, Valid Acc: 66.21
> Epoch  90: Train Loss 0.027626, Train Acc: 99.65, Valid Acc: 63.03
> Epoch 100: Train Loss 0.042715, Train Acc: 99.81, Valid Acc: 65.58
> Best Valid Acc: 66.21%
> ```

> **Problem 3.2** *(2 points)*
> Implement a recurrent neural network (without using PyTorch's RNN module) where the output of the linear layer not only depends on the current input but also the previous output.
> Report the model's accuracy on the dev data.

```python
class RNNCell(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.V = nn.Linear(hidden_features, hidden_features, bias=False)
        self.U = nn.Linear(in_features, hidden_features)
        self.tanh = nn.Tanh()

    def forward(self, h, x):
        h = self.tanh(self.V(h) + self.U(x))
        return h
```

```python
class RNN(nn.Module):
    def __init__(self, in_features, hidden_features, num_layers):
        super().__init__()
        self.hidden_features = hidden_features
        self.layers = nn.ModuleList([
            RNNCell((in_features if i == 0 else hidden_features), hidden_features)
            for i in range(num_layers)
        ])
        self.W = nn.Linear(hidden_features, hidden_features)

    def forward(self, x):
        h = [torch.zeros((x.shape[0], self.hidden_features), device=x.device)
             for _ in range(len(self.layers))]
        for t in range(x.shape[1]):
            xt = x[:, t, :]
            for i, layer in enumerate(self.layers):
                h[i] = layer(h[i], h[i - 1] if i > 0 else xt)
        o = self.W(h[-1])
        return o
```

```python
class RNNModel(ModelBase):
    def __init__(self, dim, embed=None):
        super().__init__(RNN(dim, dim, 1), dim, embed)
```

```python
torch.manual_seed(19)
model = RNNModel(dim=64, embed=vocab).to(device)
Trainer(
    model, torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=7e-4),
).train(train_loader, valid_loader, epochs=100)
```

> ```text
> RNNModel
> Epoch  10: Train Loss 0.645090, Train Acc: 53.15, Valid Acc: 50.41
> Epoch  20: Train Loss 0.640592, Train Acc: 54.39, Valid Acc: 51.14
> Epoch  30: Train Loss 0.625174, Train Acc: 55.65, Valid Acc: 49.50
> Epoch  40: Train Loss 0.591905, Train Acc: 59.74, Valid Acc: 50.50
> Epoch  50: Train Loss 0.513830, Train Acc: 66.06, Valid Acc: 47.50
> Epoch  60: Train Loss 0.198416, Train Acc: 96.42, Valid Acc: 52.86
> Epoch  70: Train Loss 0.012408, Train Acc: 99.18, Valid Acc: 52.13
> Epoch  80: Train Loss 0.017013, Train Acc: 99.23, Valid Acc: 50.77
> Epoch  90: Train Loss 0.034220, Train Acc: 99.73, Valid Acc: 51.50
> Epoch 100: Train Loss 0.065246, Train Acc: 99.57, Valid Acc: 50.86
> Best Valid Acc: 52.86%
> ```

> **Problem 3.3** *(2 points)*
> Show that the cross entropy computed above is equivalent to the negative log likelihood of the probability distribution.

> **Solution 3.3**
> The negative log likelihood $\text{NLL}$ of the probability distribution $p$ given parameters $\theta$ is,
> $$
>     \begin{align*}
>         \text{NLL}(\theta | x)
>         &= -\frac{1}{N} \log \mathbb{P}_\theta(X=x) \\
>         &= -\frac{1}{N} \log \prod_{x\in\mathcal{X}} q_\theta(x)^{Np(x)} \\
>         &= -\frac{1}{N} \sum_{x\in\mathcal{X}} \log q_\theta(x)^{Np(x)} \\
>         &= -\sum_{x\in\mathcal{X}} p(x) \log q_\theta(x) \\
>         &= H(p, q)
>     \end{align*}
> $$
> where $H(p, q)$ is the cross entropy.
> So, the cross entropy is equivalent to the negative log likelihood.

> **Problem 3.4 (bonus)** *(1 points)*
> Why is it numerically unstable if you compute log on top of softmax?

> **Solution 3.4**
> The softmax is defined as:
> $$
>     \sigma(\textbf{z})_k = \frac{e^{z_k}}{\sum_{i=1}^K e^{z_i}} \qquad
>     \text{for } k = 1, 2, \cdots, K \text{ and } \textbf{z} = (z_1, \cdots, z_k) \in \mathbb{R}^K.
> $$
> Let $c$ is the maximum value of elements in $\mathbf{z}$, which means,
> $$
>     c = \max_{1 \le i \le K} z_i.
> $$
> Then, $\mathbf{z}$ can be written as:
> $$
>     \begin{align*}
>         \textbf{z}
>         &= (z_1, \cdots, z_k) \\
>         &= \left(c + (z_1 - c), \cdots, c + (z_k - c)\right).
>     \end{align*}
> $$
> The log of the softmax $\textbf{z}$ can be computed as:
> $$
>     \begin{align*}
>         \log\sigma(\textbf{z})
>         &= \log\left(\frac{e^{z_1}}{\sum_{i=1}^K e^{z_i}}, \cdots, \frac{e^{z_K}}{\sum_{i=1}^K e^{z_i}}\right) \\
>         &= \log\left(\frac{e^{z_1 - c}}{\sum_{i=1}^K e^{z_i - c}}, \cdots, \frac{e^{z_K - c}}{\sum_{i=1}^K e^{z_i - c}}\right) \\
>         &= \left(\log\frac{e^{z_1 - c}}{\sum_{i=1}^K e^{z_i - c}}, \cdots, \log\frac{e^{z_K - c}}{\sum_{i=1}^K e^{z_i - c}}\right).
>     \end{align*}
> $$
> Note that $\sum_{i=1}^K e^{z_i - c} \ge 1$ because there exists $e^{c-c} = e^0 = 1$ when $z_k = c$.
> Now, let $d$ is the minimum value of elements in $\mathbf{z}$, which means,
> $$
>     d = \min_{1 \le j \le K} z_j.
> $$
> Then, as $d - c \to -\infty$, $e^{d-c} \to 0$ and $\frac{e^{d-c}}{\sum_{i=1}^K e^{z_i - c}} \to 0$.
> This gives $\log\frac{e^{d-c}}{\sum_{i=1}^K e^{z_i - c}} \to -\infty$ in the result of the log of the softmax $\textbf{z}$, which makes numerically unstable.

## 4. Text Classification with LSTM and Dropout

Replace your RNN module with an LSTM module. See Lecture slides 05 and 06 for the formal definition of LSTMs.

You will also use Dropout, which randomly makes each dimension zero with the probability of `p` and scale it by `1/(1-p)` if it is not zero during training.
Put it either at the input or the output of the LSTM to prevent it from overfitting.

```python
class LSTMCell(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.Vf = nn.Linear(hidden_features, hidden_features, bias=False)
        self.Vi = nn.Linear(hidden_features, hidden_features, bias=False)
        self.Vo = nn.Linear(hidden_features, hidden_features, bias=False)
        self.Vc = nn.Linear(hidden_features, hidden_features, bias=False)
        self.Uf = nn.Linear(in_features, hidden_features)
        self.Ui = nn.Linear(in_features, hidden_features)
        self.Uo = nn.Linear(in_features, hidden_features)
        self.Uc = nn.Linear(in_features, hidden_features)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, h, c, x):
        f = self.sigmoid(self.Vf(h) + self.Uf(x))
        i = self.sigmoid(self.Vi(h) + self.Ui(x))
        o = self.sigmoid(self.Vo(h) + self.Uo(x))
        c_ = self.tanh(self.Vc(h) + self.Uc(x))
        c = f * c + i * c_
        h = o * self.tanh(c)
        return h, c
```

```python
class LSTM(nn.Module):
    def __init__(self, in_features, hidden_features, dropout=0, num_layers=1, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_features = hidden_features
        self.layers = nn.ModuleList([
            LSTMCell((hidden_features if i > 0 else in_features), hidden_features)
            for i in range(num_layers)
        ])
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            for _ in range(num_layers)
        ])
        self.W = nn.Linear(hidden_features * (2 if bidirectional else 1), hidden_features)

    def forward(self, x):
        x_seq = range(x.shape[1])
        hs = []
        for seq in [x_seq, reversed(x_seq)] if self.bidirectional else [x_seq]:
            h = [torch.zeros((x.shape[0], self.hidden_features), device=x.device)
                for _ in range(len(self.layers))]
            c = [torch.zeros((x.shape[0], self.hidden_features), device=x.device)
                for _ in range(len(self.layers))]
            for t in seq:
                xt = x[:, t, :]
                for i in range(len(self.layers)):
                    h_ = (self.dropouts[i - 1](h[i - 1]) if i > 0 else xt)
                    h[i], c[i] = self.layers[i](h[i], c[i], h_)
            hs.append(h[-1])
        h = torch.cat(hs, dim=1)
        o = self.W(self.dropouts[-1](h))
        return o
```

> **Problem 4.1** *(3 points)*
> Implement and use LSTM (without using PyTorch's LSTM module) instead of vanilla RNN.
> Report the accuracy on the dev data.

```python
class LSTMModel(ModelBase):
    def __init__(self, dim, embed=None):
        super().__init__(LSTM(dim, dim), dim, embed)
```

```python
torch.manual_seed(19)
model = LSTMModel(dim=64, embed=vocab).to(device)
Trainer(
    model, torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=7e-4),
).train(train_loader, valid_loader, epochs=100)
```

> ```text
> LSTMModel
> Epoch  10: Train Loss 0.690026, Train Acc: 50.33, Valid Acc: 49.32
> Epoch  20: Train Loss 0.688016, Train Acc: 50.56, Valid Acc: 50.59
> Epoch  30: Train Loss 0.690925, Train Acc: 51.67, Valid Acc: 50.05
> Epoch  40: Train Loss 0.685361, Train Acc: 52.86, Valid Acc: 49.68
> Epoch  50: Train Loss 0.643789, Train Acc: 53.71, Valid Acc: 50.68
> Epoch  60: Train Loss 0.235452, Train Acc: 94.10, Valid Acc: 72.84
> Epoch  70: Train Loss 0.127913, Train Acc: 97.20, Valid Acc: 71.75
> Epoch  80: Train Loss 0.101538, Train Acc: 98.01, Valid Acc: 71.03
> Epoch  90: Train Loss 0.057607, Train Acc: 98.30, Valid Acc: 70.66
> Epoch 100: Train Loss 0.085226, Train Acc: 98.48, Valid Acc: 70.48
> Best Valid Acc: 72.84%
> ```

> **Problem 4.2** *(2 points)*
> Use Dropout on LSTM (either at input or output).
> Report the accuracy on the dev data.

```python
class LSTMDropoutModel(ModelBase):
    def __init__(self, dim, dropout, embed=None):
        super().__init__(LSTM(dim, dim, dropout), dim, embed)
```

```python
torch.manual_seed(19)
model = LSTMDropoutModel(dim=64, dropout=0.3, embed=vocab).to(device)
Trainer(
    model, torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=7e-4),
).train(train_loader, valid_loader, epochs=100)
```

> ```text
> LSTMDropoutModel
> Epoch  10: Train Loss 0.691325, Train Acc: 50.33, Valid Acc: 49.32
> Epoch  20: Train Loss 0.687287, Train Acc: 50.33, Valid Acc: 49.32
> Epoch  30: Train Loss 0.691723, Train Acc: 51.40, Valid Acc: 50.41
> Epoch  40: Train Loss 0.687274, Train Acc: 52.52, Valid Acc: 50.23
> Epoch  50: Train Loss 0.660254, Train Acc: 55.26, Valid Acc: 50.95
> Epoch  60: Train Loss 0.144474, Train Acc: 95.88, Valid Acc: 70.75
> Epoch  70: Train Loss 0.089596, Train Acc: 98.13, Valid Acc: 72.30
> Epoch  80: Train Loss 0.099995, Train Acc: 98.64, Valid Acc: 70.30
> Epoch  90: Train Loss 0.104280, Train Acc: 98.77, Valid Acc: 69.57
> Epoch 100: Train Loss 0.061152, Train Acc: 98.34, Valid Acc: 69.85
> Best Valid Acc: 72.30%
> ```

> **Problem 4.3 (bonus)** *(2 points)*
> Consider implementing bidirectional LSTM and two layers of LSTM.
> Report your accuracy on dev data.

```python
class LSTMBidirectionalModel(ModelBase):
    def __init__(self, dim, embed=None):
        super().__init__(LSTM(dim, dim, bidirectional=True), dim, embed)
```

```python
torch.manual_seed(19)
model = LSTMBidirectionalModel(dim=64, embed=vocab).to(device)
Trainer(
    model, torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=7e-4),
).train(train_loader, valid_loader, epochs=100)
```

> ```text
> LSTMBidirectionalModel
> Epoch  10: Train Loss 0.448450, Train Acc: 81.02, Valid Acc: 68.94
> Epoch  20: Train Loss 0.242369, Train Acc: 93.64, Valid Acc: 69.48
> Epoch  30: Train Loss 0.054993, Train Acc: 98.77, Valid Acc: 69.12
> Epoch  40: Train Loss 0.042721, Train Acc: 99.65, Valid Acc: 71.21
> Epoch  50: Train Loss 0.047925, Train Acc: 99.18, Valid Acc: 71.48
> Epoch  60: Train Loss 0.032370, Train Acc: 99.11, Valid Acc: 69.94
> Epoch  70: Train Loss 0.036121, Train Acc: 98.96, Valid Acc: 70.84
> Epoch  80: Train Loss 0.021599, Train Acc: 99.59, Valid Acc: 70.21
> Epoch  90: Train Loss 0.008167, Train Acc: 99.94, Valid Acc: 69.66
> Epoch 100: Train Loss 0.023611, Train Acc: 99.73, Valid Acc: 70.75
> Best Valid Acc: 71.48%
> ```

```python
class LSTMTwoLayerModel(ModelBase):
    def __init__(self, dim, embed=None):
        super().__init__(LSTM(dim, dim, num_layers=2), dim, embed)
```

```python
torch.manual_seed(19)
model = LSTMTwoLayerModel(dim=64, embed=vocab).to(device)
Trainer(
    model, torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4),
).train(train_loader, valid_loader, epochs=100)
```

> ```text
> LSTMTwoLayerModel
> Epoch  10: Train Loss 0.677605, Train Acc: 54.88, Valid Acc: 51.50
> Epoch  20: Train Loss 0.257917, Train Acc: 94.29, Valid Acc: 71.48
> Epoch  30: Train Loss 0.086907, Train Acc: 97.07, Valid Acc: 71.30
> Epoch  40: Train Loss 0.075478, Train Acc: 97.78, Valid Acc: 71.84
> Epoch  50: Train Loss 0.036275, Train Acc: 98.82, Valid Acc: 68.39
> Epoch  60: Train Loss 0.010863, Train Acc: 99.66, Valid Acc: 68.94
> Epoch  70: Train Loss 0.001042, Train Acc: 99.89, Valid Acc: 70.12
> Epoch  80: Train Loss 0.003824, Train Acc: 99.78, Valid Acc: 70.30
> Epoch  90: Train Loss 0.001627, Train Acc: 99.95, Valid Acc: 70.57
> Epoch 100: Train Loss 0.088498, Train Acc: 99.23, Valid Acc: 69.57
> Best Valid Acc: 71.84%
> ```

## 5. Pretrained Word Vectors

The last step is to use pretrained vocabulary and word vectors.
The prebuilt vocabulary will replace the vocabulary you built with SST training data, and the word vectors will replace the embedding vectors.
You will observe the power of leveraging self-supservised pretrained models.

> **Problem 5.1 (bonus)** *(2 points)*
> Go to [GloVe project page](https://nlp.stanford.edu/projects/glove/) and download `glove.6B.zip`.
> Use these pretrained word vectors to replace word embeddings in your model from 4.2.
> Report the model's accuracy on the dev data.

```python
from urllib import request
import zipfile

if not os.path.exists("glove/glove.6B.100d.txt"):
    os.makedirs("glove", exist_ok=True)
    print("Downloading glove.6B.zip")
    request.urlretrieve("http://nlp.stanford.edu/data/glove.6B.zip", "glove/glove.6B.zip")
    with zipfile.ZipFile("glove/glove.6B.zip", "r") as zipf:
        zipf.extractall("glove")
```

```python
dim = 50

with open(f"glove/glove.6B.{dim}d.txt", "r") as glove:
    glove_map = map(lambda line: line.split(), glove.readlines())
    word2vec = {word: list(map(float, vec)) for word, *vec in glove_map}

unk_vec = word2vec["unk"]
x2vec = lambda x_item: torch.Tensor([[word2vec.get(token, unk_vec) for token in x_item]]).to(device)

glove_train_loader = get_dataloader(x_train, y_train, x2vec, y2long, batch_size, shuffle=True)
glove_valid_loader = get_dataloader(x_valid, y_valid, x2vec, y2long, batch_size, shuffle=False)
```

```python
torch.manual_seed(19)
model = MLPModel(dim=dim, length=length).to(device)
Trainer(
    model, torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=3e-3),
).train(glove_train_loader, glove_valid_loader, epochs=100)
```

> ```text
> MLPModel
> Epoch  10: Train Loss 0.403562, Train Acc: 78.80, Valid Acc: 65.67
> Epoch  20: Train Loss 0.378033, Train Acc: 86.39, Valid Acc: 68.30
> Epoch  30: Train Loss 0.322711, Train Acc: 91.42, Valid Acc: 66.12
> Epoch  40: Train Loss 0.189466, Train Acc: 94.22, Valid Acc: 68.66
> Epoch  50: Train Loss 0.143744, Train Acc: 96.76, Valid Acc: 66.85
> Epoch  60: Train Loss 0.094514, Train Acc: 98.22, Valid Acc: 67.67
> Epoch  70: Train Loss 0.103640, Train Acc: 98.78, Valid Acc: 67.57
> Epoch  80: Train Loss 0.119256, Train Acc: 98.82, Valid Acc: 65.30
> Epoch  90: Train Loss 0.072094, Train Acc: 99.47, Valid Acc: 67.03
> Epoch 100: Train Loss 0.056376, Train Acc: 99.53, Valid Acc: 67.57
> Best Valid Acc: 68.66%
> ```

```python
torch.manual_seed(19)
model = RNNModel(dim=dim).to(device)
Trainer(
    model, torch.optim.Adam(model.parameters(), lr=3e-4),
).train(glove_train_loader, glove_valid_loader, epochs=100)
```

> ```text
> RNNModel
> Epoch  10: Train Loss 0.694586, Train Acc: 51.58, Valid Acc: 50.23
> Epoch  20: Train Loss 0.613806, Train Acc: 69.49, Valid Acc: 68.76
> Epoch  30: Train Loss 0.546628, Train Acc: 70.58, Valid Acc: 68.66
> Epoch  40: Train Loss 0.602340, Train Acc: 71.98, Valid Acc: 69.12
> Epoch  50: Train Loss 0.546888, Train Acc: 73.07, Valid Acc: 69.12
> Epoch  60: Train Loss 0.502023, Train Acc: 70.24, Valid Acc: 65.85
> Epoch  70: Train Loss 0.498196, Train Acc: 73.69, Valid Acc: 68.94
> Epoch  80: Train Loss 0.582686, Train Acc: 75.63, Valid Acc: 71.75
> Epoch  90: Train Loss 0.461066, Train Acc: 76.87, Valid Acc: 70.84
> Epoch 100: Train Loss 0.533453, Train Acc: 77.00, Valid Acc: 72.39
> Best Valid Acc: 72.39%
> ```

```python
torch.manual_seed(19)
model = LSTMModel(dim=dim).to(device)
Trainer(
    model, torch.optim.Adam(model.parameters(), lr=3e-4),
).train(glove_train_loader, glove_valid_loader, epochs=100)
```

> ```text
> LSTMModel
> Epoch  10: Train Loss 0.596871, Train Acc: 69.62, Valid Acc: 70.84
> Epoch  20: Train Loss 0.503496, Train Acc: 72.62, Valid Acc: 72.30
> Epoch  30: Train Loss 0.538801, Train Acc: 75.77, Valid Acc: 74.30
> Epoch  40: Train Loss 0.448008, Train Acc: 77.65, Valid Acc: 72.84
> Epoch  50: Train Loss 0.470757, Train Acc: 78.75, Valid Acc: 73.75
> Epoch  60: Train Loss 0.416706, Train Acc: 80.62, Valid Acc: 73.84
> Epoch  70: Train Loss 0.299589, Train Acc: 81.62, Valid Acc: 74.30
> Epoch  80: Train Loss 0.280359, Train Acc: 83.11, Valid Acc: 74.02
> Epoch  90: Train Loss 0.385639, Train Acc: 80.11, Valid Acc: 73.84
> Epoch 100: Train Loss 0.383262, Train Acc: 85.32, Valid Acc: 74.93
> Best Valid Acc: 74.93%
> ```

```python
torch.manual_seed(19)
model = LSTMDropoutModel(dim=dim, dropout=0.3).to(device)
Trainer(
    model, torch.optim.Adam(model.parameters(), lr=3e-4),
).train(glove_train_loader, glove_valid_loader, epochs=100)
```

> ```text
> LSTMDropoutModel
> Epoch  10: Train Loss 0.639923, Train Acc: 68.16, Valid Acc: 69.03
> Epoch  20: Train Loss 0.520845, Train Acc: 72.12, Valid Acc: 73.30
> Epoch  30: Train Loss 0.544825, Train Acc: 75.71, Valid Acc: 73.93
> Epoch  40: Train Loss 0.438084, Train Acc: 74.73, Valid Acc: 70.84
> Epoch  50: Train Loss 0.454220, Train Acc: 78.78, Valid Acc: 73.84
> Epoch  60: Train Loss 0.437089, Train Acc: 80.37, Valid Acc: 73.30
> Epoch  70: Train Loss 0.314890, Train Acc: 81.62, Valid Acc: 73.48
> Epoch  80: Train Loss 0.331174, Train Acc: 82.96, Valid Acc: 73.84
> Epoch  90: Train Loss 0.346455, Train Acc: 82.95, Valid Acc: 74.30
> Epoch 100: Train Loss 0.358120, Train Acc: 85.81, Valid Acc: 72.93
> Best Valid Acc: 74.30%
> ```

```python
torch.manual_seed(19)
model = LSTMBidirectionalModel(dim=dim).to(device)
Trainer(
    model, torch.optim.Adam(model.parameters(), lr=3e-4),
).train(glove_train_loader, glove_valid_loader, epochs=100)
```

> ```text
> LSTMBidirectionalModel
> Epoch  10: Train Loss 0.658043, Train Acc: 71.49, Valid Acc: 68.48
> Epoch  20: Train Loss 0.486678, Train Acc: 75.80, Valid Acc: 73.02
> Epoch  30: Train Loss 0.438268, Train Acc: 78.29, Valid Acc: 75.75
> Epoch  40: Train Loss 0.420359, Train Acc: 80.09, Valid Acc: 75.57
> Epoch  50: Train Loss 0.390116, Train Acc: 81.68, Valid Acc: 75.02
> Epoch  60: Train Loss 0.371564, Train Acc: 84.12, Valid Acc: 75.02
> Epoch  70: Train Loss 0.311193, Train Acc: 85.12, Valid Acc: 73.57
> Epoch  80: Train Loss 0.270903, Train Acc: 87.91, Valid Acc: 73.66
> Epoch  90: Train Loss 0.262891, Train Acc: 89.35, Valid Acc: 73.30
> Epoch 100: Train Loss 0.236179, Train Acc: 91.03, Valid Acc: 72.93
> Best Valid Acc: 75.75%
> ```

```python
torch.manual_seed(19)
model = LSTMTwoLayerModel(dim=dim).to(device)
Trainer(
    model, torch.optim.Adam(model.parameters(), lr=3e-4),
).train(glove_train_loader, glove_valid_loader, epochs=100)
```

> ```text
> LSTMTwoLayerModel
> Epoch  10: Train Loss 0.584000, Train Acc: 70.61, Valid Acc: 71.93
> Epoch  20: Train Loss 0.486187, Train Acc: 74.50, Valid Acc: 72.12
> Epoch  30: Train Loss 0.440720, Train Acc: 76.72, Valid Acc: 74.11
> Epoch  40: Train Loss 0.497369, Train Acc: 78.12, Valid Acc: 73.84
> Epoch  50: Train Loss 0.392846, Train Acc: 80.58, Valid Acc: 74.75
> Epoch  60: Train Loss 0.403120, Train Acc: 81.87, Valid Acc: 74.48
> Epoch  70: Train Loss 0.311465, Train Acc: 80.63, Valid Acc: 73.21
> Epoch  80: Train Loss 0.365286, Train Acc: 84.59, Valid Acc: 74.93
> Epoch  90: Train Loss 0.363596, Train Acc: 84.13, Valid Acc: 73.57
> Epoch 100: Train Loss 0.358913, Train Acc: 83.87, Valid Acc: 73.84
> Best Valid Acc: 74.93%
> ```
