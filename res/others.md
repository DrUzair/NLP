* [CNN for NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp)

# Recurrent Neural Networks
## Architectures
* Many to One: Sentiment classification
* Many to Many: 

** Named Entity Recognition
** Translation

* One to Many: Music Generation
* One to one: Standard NN
* [The unreasonable effectiveness of RNN](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

## Issues
### Vanishing Gradients
Difficulty in capturing long range dependencies:

*The trip* to Niagara Falls was a pleasent one *despite* cold weather.

*The trip* to Niagara Falls was a pleasent one *due to* cold weather.

*despite* or *due to* are related to *The trip* but for a model (i.e. RNN) it is hard to produce probability of such a distant dependency. 
 

Technically, The effect of deeper (later) layers on change in the weights of shallower (earlier) layers is negligible or ***vanishing*
