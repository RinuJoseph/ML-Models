# Machine Learning Models Implementation

In this repository, I will be implementing various machine learning models that I learned from a series of tutorial videos by Berkeley. You can find the complete series here: [Berkeley ML Tutorial Series - Video Link](https://www.youtube.com/watch?v=Q3fqoJ41g6U&list=PLzWRmD0Vi2KVsrCqA4VnztE4t71KnTnP5)

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

<center><img width="318" alt="image" src="https://github.com/user-attachments/assets/cd14f4c1-79e9-49c9-b100-c1d629e83780">
<p>Difference between autoencoder and variational encoder</p></center>



<figure>
<img src="media/image2.png" style="width:3.90592in;height:0.30421in" />
<figcaption><p>: VAEs Loss Function</p></figcaption>
</figure>
    The loss function (figure 2) contains 2 terms, reconstruction loss (to
    make the output close to the original input) and a regularization term
    (to make the learned latent variables to be close to standard normal
    distribution)

2.  **Generative Adversarial networks (GANs):**

    GANs works on the concept of adversarial learning where two networks in
    the system compete which other. It composes of 2 neural networks

    ![Figure 2: Roles of the generator and the discriminator. Source:
    Stanford CS231n \[2\].](media/image3.png){width="3.2980785214348205in"
    height="1.225in"}

    -   **Generator:** It generates fake images by sampling from an input
        random noise to fool the discriminator.

    -   **Discriminator:** It tries to classify the images as real or fake.
        Discriminator is fed with fake images from generator and the real
        images.

    : Objective function of GANs

    GANs trained on minmax objective function (figure 3). The goal of the
    discriminator **(D)** is to correctly classify real and generated
    images. The discriminator wants to maximize the function V(D,G) as it
    tries to correctly classify real images x and generated images G(z). The
    generator, on the other hand, tries to minimize the same function V(D,G)
    by generating realistic images so that the discriminator fails to
    classify them correctly. Additionally, Conditional GANs an extension of
    GAN models takes some information such as class labels to generate
    desired output.

3.  **Diffusion Models:**

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

    : The process of adding noise in forward process in 10 steps.

    <figure>
    <img src="media/image6.png" style="width:3.30086in;height:0.37in" />
    <figcaption><p>: Mathematical formulation of forward
    process.</p></figcaption>
    </figure>

    -   The **Reverse** diffusion is supposed to turn that noise into the
        image again. Starting from the noisy data xT​, the model attempts to
        remove the noise in a step-by-step manner to recover the original
        data x0. Figure 6 shows the reverse process of diffusion model.

    : Mathematical formulation of reverse diffusion process.
