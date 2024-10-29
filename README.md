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










