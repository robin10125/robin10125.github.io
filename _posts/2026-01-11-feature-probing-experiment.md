---
layout: post
title: "Can a model trained to predict letters learn to represent words?"
date: 2026-01-11
---
<nav>
  <a href="{{ site.baseurl }}/">← Back to all posts</a>
</nav>

<hr>
I recently did a fun weekend experiment to test out the use of sparse autoencoders for feature detection.  Specifically, I train a model to predict individual characters from the TinyStories dataset, and then use feature detection to test if the model has learned to represent individual words from this simple objective.  The purpose of this experiment is to build an intuition on the development of features, and more importantly for me, to start experimenting in the space of emergent language abilities.  That is to say, it is a first and smallest step in the direction of having models learn language as a higher order abstraction of environmental features.  

I am interested in this because I am interested in how we (as humans) can learn distinct concepts (that we can put words to) from raw enviromental data -to the extent that language emerges naturally from this process.  I think this is an interesting metric to pose for models -that is, if a model can produce language as an emergent property of modeling environmental data, operating as a agent in that enviromemnt, and performing social functions with other agents, then it surely has achieved impressive levels of intelligence.  And this leads to the question that truly interests me, which is how we can structure models to do this.  An interesting aside: did you know that 2 human babies can develop their own language to communicate with each other absent any other exposure to language?  Imagine if two artificial models could do this also!

The scope of this experiment does not reach this yet, as I said, this is a first and smallest step in my exploration of this idea.

Going into this experiment, I fully expected to the model to learn to represent words, as doing so seems to be the most natural way to properly predict characters -if you have an internal model of words and their composed characters, predicting next characters is a trivial task. 

# Experimental Design

## Transformer Model
I trained a simple GPT-2 style model to predict character level (byte) tokens from the TinyStories dataset.  I trained the model for 6000 steps, with a batch size of 32, and context length of 512, totalling around 100 000 000 tokens.  The model was trained on a single Tesla T4 GPU for a grand total of a couple dollars worth of compute credits.

Model hyperparameters table below:

| Hyperparameter | Value |
| --- | --- |
| Vocab Size | 256 |
| Context Length | 512 |
| Embedding Dimension | 512 |
| Attention Heads | 8 |
| Layers | 8 |
| FFN Hidden Dimension | 2048 |
| Dropout | 0.1 |
| Normalization | RMSNorm | 
| Positional Embedding | RoPE 
| Optimizer | AdamW |
| Batch Size | 32 |
| Training Steps | 6000 |
| Gradient Accumulation Steps | 4 |
| Learning Rate | 1e-3 |
| Weight Decay | 0.1 |
| Warmup Steps | 300 |
| Learning Rate Scheduler | Cosine |

The model reached a final validation loss of 0.5750 after training, which I figured would be sufficient for the task at hand.  The text generation is thematically relevant to the prompt but kind of incoherent as well, but notable the byte level generation produces real words, spelt correctly and relevant to the context.  
Eg:
>Prompt: In a world where artificial intelligence
>Output: In a world where artificial intelligence and a whale wanted to be saved!
>
>The fish and the whale worked together to build a river. Soon the whale was sailing around the river with soap!

This definitly has that delightful creativity that only small models seem to produce.  

<figure>
  <img src="/images/char-feature-model-val-loss.png" alt="Loss curve for character prediction model.">
</figure>

## Sparse Autoencoder
I then trained the SAE.  I used <a href="https://cdn.openai.com/papers/sparse-autoencoders.pdf">top-k activation selection to enforce sparcity</a>. I used a 32x expansion factor, for a hidden dimension of 16384, and a k value of 32.

SAE hyperparameters table below:

| Hyperparameter | Value |
| --- | --- |
| Expansion Factor | 32 |
| Hidden Dimension | 16384 |
| k | 32 |
| Learning Rate | 1e-4 |
| Batch Size | 4096 |
| Training Steps | 30 000 |
| Warmup Steps | 100 |
| Resample Dead Neurons | True |
| Dead Neuron Threshold | 10000 |

One mistake I made was only collecting 250 000 activations for feature detection, which is probably 100x to few given the number of training steps.  I don't think this really affects my results for the purposes of this experiment, but it is something I will be more careful about next time.
Anyways, the SAE model reached a final reconstruction loss of 0.6197, an aux top-k loss of 0.4488.

<figure>
  <img src="/images/SAE-recon-loss.png" alt="Training stats for SAE model.">
  <figcaption>Graphs of training stats for SAE model. These graphs reflect performance on the training set</figcaption>
</figure>

I then did a basic analysis of the activations, finding:

>Total features: 16384
>Dead features (never activated): 2844
>Rare features (<0.1%): 10311
>Mean activation frequency: 0.0020
>Median activation frequency: 0.0007

I then look to find which features activate very strongly to particular words, and which features activate exvlusively to particular words.  I look at the most common words in my dataset, and find the features that activate for tokens towards the end of the word. This roughly corresponds to the presence of features that can detect the end of a word.  A second exmperiment I am currently running looks at the middle of the word and examines the features produced there.  I will update this post with the results of that experiment when I am finished with it.
<figure>
  <img src="/images/feature-activation-frequency.png" alt="Distribution of feature activations and their frequency.">
</figure>
# Results and Discussion
Overall I found 262 words that have unique highly selective features active strongly for them.  My analysis so far isn't super deep, but superficially these results are seem pretty strong wrt to my hypothesis.  I am quite impressed by how many words have dedicated highly selective features.  

We already knew this, but this helps confirm the idea that abstract prediction targets can emerge organically from training on low level prediction tasks.  This is directly analogous to the traditional mecahnistic interpretability tests of finding high order concepts from a model traiend to predict word level tokens.  

## Next Step 

Simply finding features that correspond uniquely to particular words at the end of the word is not enough to show that the model is really using word level abstractions.  These features could simply be features corresponding to sequences of characters that tell the model to impement a space and thus end a word.  I am looking to show the model developing a word level abstraction of the data, and to do that I must show that the model uses word level features to make word level predictions.  To do this I decided to also detect selective features that are active for particular words in the middle of the word.  Finding these less straighforward that finding features that correspond to the end of a word, so I implemented another strategy to find these.  Rather than searching for features that are active for tokens at the end of a word, I did a search over all the features and found positions in the text where these features are active.  I then limited the search to features that are selectively active for particular words.  I found 3890 highly selective features out of 15997.  Not surprisingly, the word "and" has the most highly selective features, with 131 of them, but surprisingly, the word "little" was very well represented at number 2 with 112 highly selective features.  For the next stage of this project, I focused on some of these features that correspond strongly to "little" mid word activations.

## Feature Injection and Steering

To more robustly test the hypothesis that these features are useful word level abstractions, I tried a simple feature injection experiment.  I took several highly selective features, and amplified their activation in the forward pass of the GPT model to see if this would bias the outputs to the the correlated words.  
Overall the results are noisy due to the small scale of this experiment, and somewhat qualitative, however an interesting result came from amplifying feature 7149, highly selective for the word "little".  At no amplification, the text generation does not make use of the word "little"; however, at higher levels of apmlification, we see "little" start appearing from the same initial prompts.  For example, "once upon a time, there was a " → "grumpy old lady. She was very restless a..." became "once upon a time, there was a " → "little boy who went to the beach with hi..." at higher levels of amplification.  As the amplification strength increases, the rate of usage of "little" remains above the baseline usage.  


[Check out the notebook for more details and the implementation!](https://github.com/robin10125/robin10125.github.io/blob/main/code/char_gpt_sae_training.ipynb)
