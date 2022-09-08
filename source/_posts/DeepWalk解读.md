---
title: DeepWalk解读
mathjax: true
toc: true
date: 2022-09-09 00:41:52
categories:
- Machine Learning
tags:
- DeepWalk
- Paper Reading
- PyTorch
---

转载自：[dsgiitr/graph_nets](https://github.com/dsgiitr/graph_nets/blob/master/DeepWalk/DeepWalk_Blog%2BCode.ipynb)

___

As a part of this blog series and continuing with the tradition of extracting useful graph features by considering the topology of the network graph using machine learning, this blog deals with Deep Walk. This is a simple unsupervised online learning approach, very similar to language modelling used in NLP, where the goal is to generate word embeddings. In this case, generalizing the same concept, it simply tries to learn latent representations of nodes/vertices of a given graph. These graph embeddings which capture neighborhood similarity and community membership can then be used for learning downstream tasks on the graph. 

<!--more-->


![Input Graph to Embdeddings](./DeepWalk解读/karate_to_embedding.jpg)


## Motivation

Assume a setting, given a graph G where you wish to convert the nodes into embedding vectors and the only information about a node are the indices of the nodes to which it is connected (adjacency matrix). Since there is no initial feature matrix corresponding to the data, we will construct a feature matrix which will have all the randomly selected nodes. There can be multiple methods to select these but here we will be assuming that they are normally sampled (though it won't make much of a difference even if they are taken from some other distribution).


## Random Walks

Random walk rooted at vertex $v_i$ as $W_{v_i}$. It is a stochastic process with random variables ${W^1}_{v_i}$, ${W^2}_{v_i}$, $. . .$, ${W^k}_{v_i}$ such that ${W^{k+1}}{v_i}$ is a vertex chosen at random from the neighbors of vertex $v_k$. Random Walk distances are good features for many problems. We'll be discussing how these short random walks are analogous to the sentences in the language modelling setting and how we can carry the concept of context windows to graphs as well.


## What is Power Law?

A scale-free network is a network whose degree distribution follows a power law, at least asymptotically. That is, the fraction $P(k)$ of nodes in the network having $k$ connections to other nodes goes for large values of $k$ as
$P(k) \sim k^{-\gamma}$ where $k=2,3$ etc.

![Power Law Graph](./DeepWalk解读/Power_Law_Graph.gif)

The network of global banking activity with nodes representing the absolute size of assets booked in the respective jurisdiction and the edges between them the exchange of financial assets, with data taken from the IMF is a scale free network and follows Power Law. We can then see clearly how a very few core nodes dominate this network, there are approximately 200 countries in the world but these 19 largest jurisdictions in terms of capital together are responsible for over 90% of the assets.

<img src="DeepWalk解读/Power_Law_Example.jpg" alt="Input Graph to Embdeddings" width="600"/>

These highly centralized networks are more formally called scale free or power law networks, that describe a power or exponential relationship between the degree of connectivity a node has and the frequency of its occurrence. [More](https://www.youtube.com/watch?v=qmCrtuS9vtU) about centralized networks and power law.

### Why is it important here?

Social networks, including collaboration networks, computer networks, financial networks and Protein-protein interaction networks are some examples of networks claimed to be scale-free.

According to the authors, "If the degree distribution of a connected graph follows a power law (i.e. scale-free), we observe that the frequency which vertices appear in the short random walks will also follow a power-law distribution. Word frequency in natural language follows a similar distribution, and techniques from language modeling account for this distributional behavior."

![NLP vs Graph Random Walks Power Law D](DeepWalk解读/NLP_vs_Graph.jpg)
*$(a)$ comes from a series of short random walks on a scale-free graph, and $(b)$ comes from the text of 100,000 articles from the English Wikipedia.*


## Intuition with SkipGram

Think about the below unrelated problem for now:-

Given, some english sentences (could be any other language, doesn't matter) you need to find a vector corresponding to each word appearing at least once in the sentence such that the words having similar meaning appear close to each other in their vector space, and the opposite must hold for words which are dissimilar.

Suppose the sentences are
1. Hi, I am Bert.
2. Hello, this is Bert.

From the above sentences you can see that 1 and 2 are related to each other, so even if someone does'nt know the language, one can make out that the words 'Hi' and 'Hello' have roughly the same meaning. We will be using a technique similar to what a human uses while trying to find out related words. Yes! We'll be guessing the meaning based on the words which are common between the sentences. Mathematically, learning a representation in word-2-vec means learning a mapping function from the word co-occurences, and that is exactly what we are heading for.

#### But, How?

First lets git rid of the punctuations and assign a random vector to each word. Now since these vectors are assigned randomly, it implies the current representation is useless. We'll use our good old friend, *probability*, to convert these into meaningful representations. The idea is to maximize the probability of the appearence of a word, given the words that appear around it. Let's assume the probability is given by $P(x|y)$ where $y$ is the set of words that appear in the same sentence in which $x$ occurs. Remember we are only taking one sentence at a time, so first we'll maximize the probability of 'Hi' given {'I', 'am', 'Bert'} , then we'll maximize the probability of 'I' given {'Hi', 'am', 'Bert'}. We will do it for each word in the first sentence, and then for the second sentence. Repeat this procedure for all the sentences over and over again until the feature vectors have converged. 

One question that may arise now is, 'How do these feature vectors relate with the probability?'. The answer is that in the probability function we'll utilize the word vectors assinged to them. But, aren't those vectors random? Ahh, they are at the start, but we promise you by the end of the blog they would have converged to the values which really gives some meaning to those seamingly random numbers.

#### So, What exactly the probability function helps us with?

What does it mean to find the probability of a vector given other vectors? This actually is a simple question with a pretty simple answer, take it as a fill in the blank problem that you may have dealt with in the primary school,

Roses ____ red.

What is the most likely guess? Most people will fill it with an 'are'. (Unless, you are pretending to be oversmart in an attempt to prove how cool you are). You were able to fill that, because, you've seen some examples of the word 'are' previously in life which help you with the context. The probability function is also trying to do the same, it is finding out the word which is most likely to occur given the words that are surrounding it.


#### But but this still doesn't explain how it's gonna do that.

In case you guessed 'Neural Network', you are correct. In this blog we'll be using neural nets (feeling sleepy now, so let's wrap this up)

It is not necesary to use neural nets to estimate the probability funciton but it works and looks cool :P, frankly, the authors used it, so we'll follow them.

The input layer will have |V| neurons, where |V| is the number of words that are interesting to us. We will be using only one hidden layer for simplicity. It can have as many neurons as you want, but it is suggested to keep a number that is less than the number of words in the vocabulary. The output layer will also have the |V| neurons.

Now let's move on to the interpretation of input and output layers (don't care about the hidden layer).
Lets suppose the words in the vocabulary are $V_1$, $V_2$, $...$ $V_i$, $....$ $V_n$. Assume that out of these V4,V7, V9 appears along with the word whose probability we are tying to maximise. so the input layers will have the 4th, 7th, and the 9th neuron with value 1 and all other will have the value 0. The hidden layer will then have some function of these values. The hidden layer have no non linear acitvation. The |V| neuron in the output layer will have a score, the higher it is ,the higher the chances of that word appearning along with the surrounding words. Apply Sigmoid, boom! we got the probabilities. 

So a simple neural network will help us solve the fill in the blank problem.


## Deep Walk = SkipGram Analogy + Random Walks

These random walks can be thought of as short sentences and phrases in a special language; the direct analog is to estimate the likelihood of observing vertex $v_i$ given all the previous vertices visited so far in the random walk, i.e. Our goal is to learn a latent representation, not only a probability distribution of node co-occurrences, and so we introduce a mapping function $ Φ: v ∈ V→R^{|V|×d} $. This mapping $Φ$ represents the latent social representation associated with each vertex $v$ in the graph. (In practice, we represent $Φ$ by a $|V|×d$ matrix of free parameters, which will serve later on as our $X_E$).

The problem then, is to estimate the likelihood: $ Pr ({v}_{i} | Φ(v1), Φ(v2), · · · , Φ(vi−1))) $

In simple words *DeepWalk* algorithm uses the notion of Random Walks to get the surrounding nodes(words) and ultimately calulate the probability given the context nodes. In simple words we use random walk to start at a node, finds out all the nodes which have and edge connecting with this start node and randomly select one out of them, then consider this new node as the start node and repeat the procedue after n iterations you will have traversed n nodes (some of them might repeat, but it does not matter as is the case of words in a sentence which may repeat as well). We will take n nodes as the surrounding nodes for the original node and will try to maximize probability with respect to those using the probability function estimate. 

*So, that is for you Ladies and Gentlemen , the <b>'DeepWalk'</b> model.*

Mathematically the Deep Walk algorithm is defined as follows,

![Deep Walk Algorithm](DeepWalk解读/DeepWalk_Algo.jpg)

## PyTorch Implementation of DeepWalk

Here we will use using the following graph as an example to implement Deep Walk on,
![Example Graph](./DeepWalk解读/graph.png)

As you can see there are two connected components, so we can expect than when we create the vectors for each node, the vectors of [1 , 2, 3, 7] should be close and similarly that of [4, 5, 6] should be close. Also if  any two vectors are from different group then their vectors should also be far away.

Here we will represent the graph using the adjacency list representation. Make sure that you are able to understand that the given graph and this adjacency list are equivalent.


```python
adj_list = [[1,2,3], [0,2,3], [0, 1, 3], [0, 1, 2], [5, 6], [4,6], [4, 5], [1, 3]]
size_vertex = len(adj_list)  # number of vertices
```

## Imports


```python
import torch
import torch.nn as nn
import random
from tqdm import tqdm
```

## Hyperparameters


```python
w=3            # window size
d=2            # embedding size
y=200          # walks per vertex
t=6            # walk length 
lr=0.025       # learning rate
```


```python
v=[0,1,2,3,4,5,6,7] #labels of available vertices
```

## Random Walk


```python
def RandomWalk(node,t):
    walk = [node]        # Walk starts from this node
    
    for i in range(t-1):
        node = adj_list[node][random.randint(0,len(adj_list[node])-1)]
        walk.append(node)

    return walk
```

## Skipgram

The skipgram model is closely related to the CBOW model that we just covered. In the CBOW model we have to maximise the probability of the word given its surrounding word using a neural network. And when the probability is maximised, the weights learnt from the input to hidden layer are the word vectors of the given words. In the skipgram word we will be using a using single word to maximise the probability of the surrounding words. This can be done by using a neural network that looks like the mirror image of the network that we used for the CBOW. And in the end the weights of the input to hidden layer will be the corresponding word vectors.

Now let's analyze the complexity.
There are |V| words in the vocabulary so for each iteration we will be modifying a total of |V| vectors. This is very complex, usually the vocabulary size is in million and since we usually need millions of iteration before convergence, this can take a long long time to run.

We will soon be discussing some methods like Hierarchical Softmax or negative sampling to reduce this complexity. But, first we'll code for a simple skipgram model. The class defines the model, whereas the function 'skip_gram' takes care of the training loop.


```python
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.phi  = nn.Parameter(torch.rand((size_vertex, d), requires_grad=True))    
        self.phi2 = nn.Parameter(torch.rand((d, size_vertex), requires_grad=True))
        
        
    def forward(self, one_hot):
        hidden = torch.matmul(one_hot, self.phi)
        out    = torch.matmul(hidden, self.phi2)
        return out
```


```python
model = Model()
```

when predicting the $k$-th vertex, the loss is as follows:
$$
\begin{aligned}
loss_k &= - log \frac{e^{v_k}} {\sum_{i=1}^n e^{v_i}} \\
&= log {\sum_{i=1}^n e^{v_i}} - v_k
\end{aligned}
$$


```python
def skip_gram(wvi,  w):
    for j in range(len(wvi)):
        for k in range(max(0,j-w) , min(j+w, len(wvi))):
            
            #generate one hot vector
            one_hot          = torch.zeros(size_vertex)
            one_hot[wvi[j]]  = 1    # 当前中心节点为1
            
            out              = model(one_hot)
            loss             = torch.log(torch.sum(torch.exp(out))) - out[wvi[k]]
            loss.backward()
            
            for param in model.parameters():
                param.data.sub_(lr*param.grad)
                param.grad.data.zero_()
```


```python
for i in tqdm(range(y)):
    random.shuffle(v)
    for vi in v:
        path=RandomWalk(vi,t)    # 生成一条随机游走的路径
        skip_gram(path, w)
```

    100%|██████████| 200/200 [00:03<00:00, 62.37it/s]


i'th row of the model.phi corresponds to vector of the i'th node. As you can see the vectors of [0, 1, 2,3 , 7] are very close, whereas their vector are much different from the vectors corresponding to [4, 5, 6].


```python
print(model.phi)
```

    Parameter containing:
    tensor([[ 0.8624,  0.7754],
            [ 0.2209,  1.0904],
            [ 0.6065,  0.7289],
            [ 0.3771,  1.1625],
            [-0.0250, -1.2358],
            [ 0.0542, -1.2929],
            [-0.6126, -1.1073],
            [-0.9810,  0.9992]], requires_grad=True)


Now we will be discussing a variant of the above using Hierarchical softmax.

## Hierarchical Softmax

As we have seen in the skip-gram model that the probability of any outcome depends on the total outcomes of our model. If you haven't noticed this yet, let us explain you how!

When we calculate the probability of an outcome using softmax, this probability depends on the number of model parameters via the normalisation constant(denominator term) in the softmax.

$\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$

And the number of such parameters are linear in the total number of outcomes. It means if we are dealing with a very large graphical structure, it can be computationally very expensive and taking a lot of time.

### Can we somehow overcome this challenge?
Obviously, Yes! (because we're asking at this stage). 

\*Drum roll please\*

<b>Enter "Hierarchical Softmax(hs)"</b>.

Basically, hs is an alternative approximation to the softmax in which the probability of any one outcome depends on a number of model parameters that is only logarithmic in the total number of outcomes.

Hierarchical softmax uses a binary tree to represent all the words(nodes) in the vocabulary. Each leaf of the tree is a node of our graph, and there is a unique path from root to the leaf. Each intermediate node of tree explicitly represents the relative probabilities of its child nodes. So these nodes are associated to different vectors which our model is going to learn.

The idea behind decomposing the output layer into binary tree is to reduce the time complexity to obtain 
probability distribution from $O(V)$ to $O(log(V))$

Let us understand the process with an example.

![binary tree](DeepWalk解读/tree.png)

In this example, leaf nodes represent the original nodes of our graph. The highlighted nodes and edges make a path from root to an example leaf node $w_2$.

Here, length of the path $L(w_{2}) = 4$.

$n(w, j)$ means the $j^{th}$ node on the path from root to a leaf node $w$.

Now, view this tree as a decision process, or a random walk, that begins at the root of the tree and descents towards the leaf nodes at each step. It turns out that the probability of each outcome in the original distribution uniquely determines the transition probabilities of this random walk. If you want to go from root node to $w_2$(say), first you have to take a left turn, again left turn and then right turn. 

Let's denote the probability of going left at an intermediate node $n$ as $p(n,left)$ and probability of going right as $p(n,right)$. So we can define the probabilty of going to $w_2$ as follows.

<b> $P(w2|wi) = p(n(w_{2}, 1), left) . p(n(w_{2}, 2),left) . p(n(w_{2}, 3), right)$ </b>

Above process implies that the cost for computing the loss function and its gradient will be proportional to the number of nodes $(V)$ in the intermediate path between root node and the output node, which on average is no greater than $log(V)$. That's nice! Isn't it? In the case where we deal with a large number of outcomes, there will be a huge difference in the computational cost of 'vanilla' softmax and hierarchical softmax.

Implementation remains similar to the vanilla, except that we will only need to change the Model class by HierarchicalModel class, which is defined below.


```python
def func_L(w):
    """
    Parameters
    ----------
    w: Leaf node.
    
    Returns
    -------
    count: The length of path from the root node to the given vertex.
    """
    count=1
    while(w!=1):
        count+=1
        w//=2

    return count
```


```python
# func_n returns the nth node in the path from the root node to the given vertex
def func_n(w, j):
    li=[w]
    while(w!=1):
        w = w//2
        li.append(w)

    li.reverse()
    
    return li[j]
```


```python
def sigmoid(x):
    out = 1/(1+torch.exp(-x))
    return out
```


```python
class HierarchicalModel(torch.nn.Module):
    
    def __init__(self):
        super(HierarchicalModel, self).__init__()
        self.phi         = nn.Parameter(torch.rand((size_vertex, d), requires_grad=True))   
        self.prob_tensor = nn.Parameter(torch.rand((2*size_vertex, d), requires_grad=True))
    
    def forward(self, wi, wo):
        one_hot     = torch.zeros(size_vertex)
        one_hot[wi] = 1
        w = size_vertex + wo
        h = torch.matmul(one_hot,self.phi)
        p = torch.tensor([1.0])
        for j in range(1, func_L(w)-1):
            mult = -1
            if(func_n(w, j+1)==2*func_n(w, j)): # Left child
                mult = 1
        
            p = p*sigmoid(mult*torch.matmul(self.prob_tensor[func_n(w,j)], h))
        
        return p
```

The input to hidden weight vector no longer represents the vector corresponding to each vector , so directly trying to read it will not provide any valuable insight, a better option is to predict the probability of different vectors against each other to figure out the likelihood of coexistance of the nodes.


```python
hierarchicalModel = HierarchicalModel()
```


```python
def HierarchicalSkipGram(wvi,  w):
   
    for j in range(len(wvi)):
        for k in range(max(0,j-w) , min(j+w, len(wvi))):
            #generate one hot vector
       
            prob = hierarchicalModel(wvi[j], wvi[k])
            loss = - torch.log(prob)
            loss.backward()
            for param in hierarchicalModel.parameters():
                param.data.sub_(lr*param.grad)
                param.grad.data.zero_()
```


```python
for i in range(y):
    random.shuffle(v)
    for vi in v:
        wvi = RandomWalk(vi,t)
        HierarchicalSkipGram(wvi, w)
```


```python
for i in range(8):
    for j in range(8):
        print((hierarchicalModel(i,j).item()*100)//1, end=' ')
    print(end = '\n')
```

    30.0 23.0 18.0 27.0 23.0 19.0 4.0 52.0 
    21.0 30.0 21.0 26.0 10.0 16.0 0.0 72.0 
    20.0 29.0 25.0 24.0 14.0 21.0 4.0 59.0 
    24.0 29.0 18.0 27.0 11.0 15.0 0.0 72.0 
    27.0 20.0 28.0 23.0 42.0 29.0 28.0 0.0 
    20.0 22.0 37.0 19.0 37.0 35.0 27.0 0.0 
    23.0 23.0 31.0 22.0 33.0 31.0 34.0 0.0 
    20.0 32.0 20.0 26.0 7.0 13.0 0.0 78.0 


<h3>References</h3>

- [DeepWalk: Online Learning of Social Representations](http://www.perozzi.net/publications/14_kdd_deepwalk.pdf)

- [An Illustrated Explanation of Using SkipGram To Encode The Structure of A Graph (DeepWalk)](https://medium.com/@_init_/an-illustrated-explanation-of-using-skipgram-to-encode-the-structure-of-a-graph-deepwalk-6220e304d71b?source=---------13------------------)

- [Word Embedding](https://medium.com/data-science-group-iitr/word-embedding-2d05d270b285)

- [Centralized & Scale Free Networks](https://www.youtube.com/watch?v=qmCrtuS9vtU)


- Beautiful explanations by Chris McCormick:
    - [Hieararchical Softmax](https://youtu.be/pzyIWCelt_E)
    - [word2vec](http://mccormickml.com/2019/03/12/the-inner-workings-of-word2vec/)
    - [Negative Sampling](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/)
    - [skip-gram](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
