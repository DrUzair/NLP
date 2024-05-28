# **Word Sense Induction with EM**

**Introduction:**
Understanding the various meanings of words within different contexts is a fundamental challenge in natural language understanding. Words often exhibit multiple senses, and disambiguating these senses is crucial for tasks like machine translation, information retrieval, and sentiment analysis. Word Sense Induction (WSI) aims to automatically discover these underlying senses from large text corpora. 

**Defining Word Sense Induction:**
Word Sense Induction is the process of automatically identifying and clustering different senses or meanings of a word within a given context or corpus without prior knowledge of the senses. Unlike Word Sense Disambiguation, which selects the most appropriate sense from a predefined set, WSI discovers senses autonomously based on distributional patterns in the data.

**Intuition behind EM Algorithm:**
The EM algorithm is a powerful tool for solving problems with latent variables, where some variables are unobserved or hidden. In the context of WSI, the latent variables represent the underlying senses of words, while the observed variables are the instances of words in the corpus. EM iteratively estimates the parameters of a probabilistic model using a two-step process: the Expectation step (E-step) and the Maximization step (M-step).

**E-step:**
In the E-step, the algorithm computes the expected values of the latent variables given the observed data and the current parameter estimates. For WSI, this involves estimating the probability that each word instance belongs to each sense cluster based on the current sense assignments and sense probabilities.

**M-step:**
In the M-step, the algorithm updates the parameters to maximize the expected log-likelihood obtained from the E-step. For WSI, this entails updating the sense probabilities and sense assignments based on the estimated probabilities from the E-step.

**Examples and Equations:**
Consider a corpus containing sentences like:

- "The bank raised interest rates."
- "She sat by the river bank."

In this context, the word "bank" exhibits two senses: financial institution and riverbank. Using EM, we initialize sense probabilities and iteratively refine them to reflect the distributional patterns of the word "bank" in the corpus.

The equations for the E-step and M-step in the EM algorithm are as follows:

**E-step:**
$P(z_i = j | w_i) = \frac{{P(w_i | z_i = j)P(z_i = j)}}{{\sum_{k=1}^{K}P(w_i | z_i = k)P(z_i = k)}}$

**M-step:**
$P(w | z = j) = \frac{{\sum_{i=1}^{N} I(z_i = j \text{ and } w_i = w)}}{{\sum_{i=1}^{N} I(z_i = j)}}$

Where:

- $P(z_i = j | w_i)$ represents the probability that word $w_i$ belongs to sense $j$,
- $P(w_i | z_i = j) $ represents the probability of observing word $w_i$ given sense $j$,
- $P(z_i = j) $ represents the prior probability of sense $j$,
- $K $ is the total number of senses,
- $N $ is the total number of word instances,
- $I(\cdot) $ is the indicator function.

**Finally:**
Word Sense Induction using the EM algorithm offers a promising approach to uncovering the nuanced meanings of words in natural language text. By leveraging distributional patterns and latent variables, EM enables the automatic discovery of word senses without relying on predefined sense inventories. As NLP applications continue to advance, the insights gained from WSI contribute to more robust and accurate language understanding systems.

[Lab code notebook](https://github.com/DrUzair/NLP/blob/master/WordSenseInduction/EM_WordSenseInduction.ipynb)
