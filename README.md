# Machine Learning Models Implementation

In this repository, I will be implementing various machine learning models that I learned from a series of tutorial videos by Berkeley. You can find the complete series here: [Berkeley ML Tutorial Series - Video Link](https://www.youtube.com/watch?v=Q3fqoJ41g6U&list=PLzWRmD0Vi2KVsrCqA4VnztE4t71KnTnP5)
### Topics
1. Generative Models
2. Transformers
3. Vision Transformers
4. Contrastive Learning
5. NeRF
   
## Lecture 9-12 GENERATIVE MODELS

Generative models learn from the existing datasets and generate new
instances that are similar to original data. They do this by capturing
or learning the probability of the dataset. Once the model has learned
distribution it will generate new data points by sampling the learnt
probability distribution.

Types of Generative Models:

1.  **Variational Auto Encoders (VAEs):**

    VAEs extends the traditional autoencoders which has an encoder-decoder
    network where the encoder compresses the input data into a latent space,
    and the decoder reconstructs the input from this latent representation.
    In contrast to vanilla autoencoders which maps input to a fixed point in
    the latent space, VAEs uses the probabilistic encoders (figure 1). In
    VAEs, encoders outputs probabilistic distribution (mean and variance) to
    the latent space (gaussian distribution) and decoder takes the sampled
    latent variables from this distribution and produces the reconstructed
    data.
<p align="center">
    <img align="center" width="318"  alt="image" src="https://github.com/user-attachments/assets/cd14f4c1-79e9-49c9-b100-c1d629e83780">
    <p align="center">Figure 1: Difference between autoencoder and variational encoder</p>
</p>

<p align="center">
    <img width="248" alt="image" src="https://github.com/user-attachments/assets/db382beb-de33-47ee-87b1-e62ff6f355e2">
    <p align="center">Figure 2: VAEs Loss Function</p>
</p>
    The loss function (figure 2) contains 2 terms, reconstruction loss (to
    make the output close to the original input) and a regularization term
    (to make the learned latent variables to be close to standard normal
    distribution)

2.  **Generative Adversarial networks (GANs):**

    GANs works on the concept of adversarial learning where two networks in
    the system compete which other. It composes of 2 neural networks
    <p align="center">
        <img width="238" alt="image" src="https://github.com/user-attachments/assets/f326040a-8f24-43b2-beea-1035370a994e">
    </p>

    -   **Generator:** It generates fake images by sampling from an input
        random noise to fool the discriminator.

    -   **Discriminator:** It tries to classify the images as real or fake.
        Discriminator is fed with fake images from generator and the real
        images.
        <p align="center">
            <img width="360" alt="image" src="https://github.com/user-attachments/assets/696a5191-7be4-4b3a-9bb9-9c4a81186080">
            <p align="center">Figure 3: Objective function of GANs</p>
        </p>

    GANs trained on minmax objective function (figure 3). The goal of the
    discriminator **(D)** is to correctly classify real and generated
    images. The discriminator wants to maximize the function V(D,G) as it
    tries to correctly classify real images x and generated images G(z). The
    generator, on the other hand, tries to minimize the same function V(D,G)
    by generating realistic images so that the discriminator fails to
    classify them correctly. Additionally, Conditional GANs an extension of
    GAN models takes some information such as class labels to generate
    desired output.

4.  **Diffusion Models:**

    Diffusion Models are generative models, used to generate data similar to
    the data on which they are trained. Diffusion Models work by destroying
    training data through Markov chain of diffusion steps which add Gaussian
    noise to the data, and then learning to reverse the diffusion process to
    construct desired data samples from the noise. Diffusion process is
    split into two processes

    -   The **Forward** diffusion is a process of turning an image into
        noise in a sequence of steps. Each step adds a small amount of noise
        to the data (figure 4). Figure 5 shows the mathematical formulation
        of the process x0, x1, x2,..., xT are the noisy samples and the xt
        depends on the previous step xt-1. βt​ is the variance schedule that
        controls amount of noise added at each time step t.
        <p align="center">
            <img width="338" alt="image" src="https://github.com/user-attachments/assets/1e24ebbc-c664-47b9-a126-0965399c936d">
            <p align="center">Figure 4: The process of adding noise in forward process in 10 steps.</p>
        </p>

    <p align="center">
        <img width="229" alt="image" src="https://github.com/user-attachments/assets/e0c30285-0abb-4110-be9d-75f534de75a7">
        <p>Figure 5: Mathematical formulation of forward process.</p> 
    </p>

    -   The **Reverse** diffusion is supposed to turn that noise into the
        image again. Starting from the noisy data xT​, the model attempts to
        remove the noise in a step-by-step manner to recover the original
        data x0. Figure 6 shows the reverse process of diffusion model.
        <p align = "center">
            <img width="227" alt="image" src="https://github.com/user-attachments/assets/a1737496-ef84-4e5f-ba06-deb6fe9ebc13">
            <p align="center"> Figure 6: Mathematical formulation of reverse diffusion process.</p>
        </p>
## Lecture 14 - Transformers and Attention
#### Background
<ul>
<li>Recurrent neural networks (RNN), long short-term memory (LSTM) and gated recurrent neural networks (GRU) have been the standard for language modeling and machine translation, but they have some limitations:</li>
   <ul>
      <li>Cannot handle very long-term dependencies </li>
      <li>In Seq-Seq models, decoder only accesses last hidden state. </li>
      <li>Early information in sentence can be lost. </li>
      <li>Not parallelizable</li>
   </ul>
<li>he Transformer model was proposed as a new architecture that eliminates recurrence and relies entirely on attention mechanisms to model global dependencies in sequences.</li>
<li>This design allows for parallelization during training, leading to significant improvements in computational efficiency and achieving state-of-the-art performance in tasks like machine translation.</li>
</ul>

#### Attention Mechanism
 <p align="center">
        <img width="229" alt="image" src="https://github.com/user-attachments/assets/6bc74125-dfe1-440c-a4cf-8d7688e2d960">
        <p align="center">Image source: : https://jalammar.github.io/illustrated-transformer</p> 
</p>

Attention is a mechanism used in “Transformer” models allows the position in the input to focus its relevance to different parts of the input sequence when processing each word or token. It helps the model understand which other words in the sentence and generate a representation of the input sequence.
The mechanism consist of 3 vectors:
Query (Q)
Key (K)
Value (V)
Each element becomes query, key, and value from the input embeddings matrix X by multiplying by a weight matrix W
 <p align="center">
        <img width="229" alt="image" src="https://github.com/user-attachments/assets/acfbb007-4308-4f0e-b1d5-3edebe04fc77">
<!--         <p>Image source: : https://jalammar.github.io/illustrated-transformer</p>  -->
</p>

 <p align="center">
        <img width="600" alt="image" height="400px" src="https://github.com/user-attachments/assets/ff8632f7-349d-4419-b3ee-d2ddb8950434">
        <p align="center">Illustration of Attention mechanism(image source:(https://www.analyticsvidhya.com/blog/2019/06/understanding-transformers-nlp-state-of-the-art-models)
</p> 
</p>

#### Transformer Model Architecture
 <p align="center">
        <img width="229" alt="image" src="https://github.com/user-attachments/assets/af98ad84-a5e0-476f-b32a-970b6e052160">
</p>
The fundamental building blocks of the Transformer model are the encoder and decoder.
Both encoder and decoder layers are stacked 6 times in the Transformer model.
Input sequences are passed to the model as embeddings along with positional encodings.
Encoder and decoder use multi-head attention and masked multi-head attention.
The decoder looks at the output of the encoder (and the previously generated words).
<!-- ![image](https://github.com/user-attachments/assets/af98ad84-a5e0-476f-b32a-970b6e052160) -->

#### Scaled Dot Product Attention
<p align="center">
        <img width="229" alt="image" src="https://github.com/user-attachments/assets/5211a051-0fdc-4288-b03e-cf39d221adce">
</p>

In the paper, attention mechanism is described as “Scaled Dot Product Attention” and it can be defined as

Input to this method consists of:
<p align="center">
        <img width="229" alt="image" src="https://github.com/user-attachments/assets/06e7bd5a-3b3e-497e-94af-690d63ed31cb">
</p>
<!-- ![image]() -->

Query (Q) (dimension dk = 64) 
Key (K) (dimension dk = 64) 
Value (V) (dimension dv = 64)			
Dot product of query and key computed and scaled. Scaling is performed by dividing dot product by        which is 8. (the logic behind the scaling: Dot product grows too large in magnitude for large number of dimensions and result in vanishing gradient problems)
SoftMax is applied to normalize the values between 0 and 1. The result of this step is attention weight matrix.
The softMax score is multiplied by value matrix.

#### Multi-Head Attention
<p align="center">
        <img width="229" alt="image" src="https://github.com/user-attachments/assets/3c07dbef-ef92-470a-b6a1-af9a3556a5d9">
</p>
<p align="center">
        <img width="229" alt="image" src="https://github.com/user-attachments/assets/fcc9e66d-4845-4907-a274-24c4baa33223">
</p>
<!-- ![image]()
![image]() -->
<ul>
   <li>Instead of using a single attention mechanism with dmodel dimension the model linearly projects query, keys and values them h times into lower dimensions dk,dk and dv.

</li>
   <li>Each projection involves different learned linear transformations for queries, keys, and values. The attention function is performed separately on each of these projected versions (heads) in parallel.
</li>
   <li>These outputs are then concatenated and projected again to form the final output.
</li>
   <li>Allows the model to attend to information from different subspaces simultaneously.
</li>
   <li><Typically, h=8 parallel attention heads are used. Each head has reduced dimensions dk, dv = dmodel /8 = 64 maintaining computational efficiency similar to single-head attention with full dimensions
/li>
</ul>

#### Tokenization and Input Embeddings
<p align="center">
        <img width="600" alt="image" height="400px" src="https://github.com/user-attachments/assets/8f96285c-1d16-4dbc-bf7c-80322a5ede26">
        <p align="center">Image Source: https://learning.oreilly.com/library/view/build-a-large/9781633437166/OEBPS/Text/chapter-2.html#p248</p>
</p> 
<!-- ![image]() -->

#### Positional Encoding
<p align="center">
        <img width="600" alt="image" height="400px" src="https://github.com/user-attachments/assets/e931a043-194d-4932-9880-fa99d12b1d21">
<!--         <p align="center">Image Source: https://learning.oreilly.com/library/view/build-a-large/9781633437166/OEBPS/Text/chapter-2.html#p248</p> -->
</p> 
<!-- ![image]() -->

<ul>
   <li>Attention mechanism doesn’t have a notion of position or order for the tokens within a sequence.
</li>
   <li>For the transformer to understand the relative or absolute positions of tokens in a sequence, a ‘positional encoding’ must be added to the input 
</li>
   <li>In order to differentiate between different positions in the input sequence, the ‘positional encoding’ uses sine and cosine functions of different frequencies. embeddings at the bottoms of the encoder and decoder stacks. 
      <p align="center">
        <img width="600" alt="image" height="400px" src="https://github.com/user-attachments/assets/02761a53-11a3-4d99-bb4c-0fec7af01b5e">
<!--         <p align="center">Image Source: https://learning.oreilly.com/library/view/build-a-large/9781633437166/OEBPS/Text/chapter-2.html#p248</p> -->
</p> 
<!-- ![image]() -->

</li>
   <li>Where pos is position in the sequence and i is the dimension or index of the positional encoding vector. For every even embedding index sine formula is applied and cosine is applied to odd index
</li>
</ul>

#### Transfomer Encoder Network
<p align="center">
        <img width="400" alt="image" height="400px" src="https://github.com/user-attachments/assets/df60d5d8-f5fb-4e36-b075-5c0eb48c5cd2">
<!--         <p align="center">Image Source: https://learning.oreilly.com/library/view/build-a-large/9781633437166/OEBPS/Text/chapter-2.html#p248</p> -->
</p> 

-   The encoder used in this paper is composed of **N = 6** identical stacked layers.
-   Each layer is divided into two sub-layers:
      1. Multi-Head attention layer
      2.  Feed-Forward network layer
-   Each encoder is encapsulated by a **residual connection** which is then followed by layer normalization.
-   Residual connections: Adds the input value x to the sub-layer’s output f(x) through a separate path.
-   The final output of each sub-layer can be described as:
      - **LayerNorm(x + f(x))**, where f(x) is the function used in the corresponding sub-layer.

**Position-wise Feed-Forward Networks**
![image](https://github.com/user-attachments/assets/2d9df99f-a344-4545-a144-05ad801711ba)
-   This layer is applied independently and parallel to each position of the input sequence.
-   This FFN consists of **two linear transformations** with a **ReLU activation function** placed in between.
-   The first linear transformation projects the vector into a higher dimensional space. In this paper, the input was transformed from **dₖ = 512 to 2048**.
-   ReLU is applied before being put through the second linear transformation, which reduces the matrix back to its original dimensionality.


#### Transformer Decoder Network
<p align="center">
        <img width="400" alt="image" height="400px" src="https://github.com/user-attachments/assets/b5b2192f-5e85-4ece-909b-845d88bafedc">
<!--         <p align="center">Image Source: https://learning.oreilly.com/library/view/build-a-large/9781633437166/OEBPS/Text/chapter-2.html#p248</p> -->
</p> 


-   The decoder is comprised of **N = 6** identical layers. 
-   It has 3 sub-layers
      1. Masked Multi-Head attention layer
      2. Multi-Head attention layer
      3. Feed-forward network layer.
-   Similar to the encoder, residual connections are employed around each sub-layer in the decoder, followed by layer normalization.
-   To make sure that the model is auto-regressive, that is, with only previous positions influencing the present position, the first sub-layer is modified with a masking mechanism that prevents each position from attending to future positions (i.e., it only looks at the current and previous tokens)
-   The output from the final encoder layer is converted to a key and value vector and passed to the decoder.
**Final Linear and Softmax Layer**
-   The Linear layer is a simple fully connected neural network that projects the vector produced by the stack of decoders, into a much, much larger vector called a logits vector.
-   The SoftMax layer then turns those scores into probabilities (all positive, all add up to 1.0). The cell with the highest probability is chosen, and the word associated with it is produced as the output for this time step.















