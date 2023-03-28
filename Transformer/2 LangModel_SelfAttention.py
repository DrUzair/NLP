import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

''' Look at all previous words to generate next
    @Author: Uzair Ahmad
    2022
'''


class SelfAttentionLanguageModel(nn.Module):
    class SelfAttentionHead(nn.Module):
        def __init__(self, in_size, out_size):
            '''
            in_size is embed_size
            out_size is head_size
            '''
            super().__init__()
            self.head_size = out_size
            self.K = nn.Linear(in_size, self.head_size, bias=False)
            self.Q = nn.Linear(in_size, self.head_size, bias=False)
            self.V = nn.Linear(in_size, self.head_size, bias=False)

        def forward(self, x):
            keys = self.K(x)
            queries = self.Q(x)
            # affinities :
            # all the queries will dot-product with all the keys
            # transpose (swap) second dimension (input_length) with third (head_size)
            keys_t = keys.transpose(1, 2)
            autocorrs = (queries @ keys_t) * (self.head_size ** -0.5)  # (batch_size x input_length x input_length)
            '''
            (batch_size x input_length x embed_size) @ (batch_size x embed_size x input_length) ----> (batch_size x input_length x input_length)
            '''
            autocorrs = torch.tril(autocorrs)
            autocorrs = autocorrs.masked_fill(autocorrs == 0, float('-inf'))
            autocorrs = torch.softmax(autocorrs, dim=-1)
            values = self.V(x)  # (batch_size x input_length x head_size)
            out = autocorrs @ values
            return out

    def __init__(self, batch_size=4,
                 input_length=8,
                 embed_size=16,
                 sa_head_size=8):
        super().__init__()
        self.val_data = None
        self.train_data = None
        self.sa_head = None
        self.val_text = None
        self.train_text = None
        self.K = None
        self.linear_sahead_to_vocab = None
        self.vocab = None
        self.token_embeddings_table = None
        self.vocab_size = None
        self.encoder = None
        self.decoder = None
        self.vocab_size: int
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # input_length = how many consecutive tokens/chars in one input
        self.input_length = input_length
        # batch_size = how many inputs are going to be processed in-parallel (on GPU)
        self.batch_size = batch_size
        # embed_size = embedding size
        self.embed_size = embed_size
        # sa_head_size head_size of self-attention module
        self.sa_head_size = sa_head_size
        self.lm_head = None
        self.position_embeddings_table = None

    def forward(self, in_ids, target=None):
        in_ids_emb = self.token_embeddings_table(in_ids)  # batch_size x embed_size
        sa_head_outputs = self.sa_head(in_ids_emb)  # communicate
        logits = self.linear_sahead_to_vocab(sa_head_outputs)  # compute
        if target is None:
            ce_loss = None
        else:
            batch_size, input_length, vocab_size = logits.shape
            logits_ = logits.view(batch_size * input_length, vocab_size)
            targets = target.view(batch_size * input_length)
            ce_loss = F.cross_entropy(logits_, targets)
        return logits, ce_loss

    def fit(self, train_iters=100, eval_iters=10, lr=0.0001):
        """
        train_iters = how many training iterations
        eval_iters = how many batches to evaluate to get average performance
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for iteration in range(train_iters):
            if iteration % eval_iters == 0:
                avg_loss = self.eval_loss(eval_iters)
                print(f"iter {iteration}: train {avg_loss['train']} val {avg_loss['eval']}")
            inputs, targets = self.get_batch(split='train')
            _, ce_loss = self(inputs, targets)
            optimizer.zero_grad(set_to_none=True)  # clear gradients of previous step
            ce_loss.backward()  # propagate loss back to each unit in the network
            optimizer.step()  # update network parameters w.r.t the loss

    def generate(self, context_tokens, max_new_tokens):
        for _ in range(max_new_tokens):
            token_rep, _ = self(context_tokens)
            last_token_rep = token_rep[:, -1, :]
            probs = F.softmax(last_token_rep, dim=1)
            next_token = torch.multinomial(probs, num_samples=1)
            context_tokens = torch.cat((context_tokens, next_token), dim=1)
        output_text = self.decoder(context_tokens[0].tolist())
        return output_text

    @torch.no_grad()  # tell torch not to prepare for back-propagation (context manager)
    def eval_loss(self, eval_iters):
        perf = {}
        # set dropout and batch normalization layers to evaluation mode before running inference.
        self.eval()
        for split in ['train', 'eval']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                tokens, targets = self.get_batch(split)  # get random batch of inputs and targete
                _, ce_loss = self(tokens, targets)  # forward pass
                losses[k] = ce_loss.item()  # the value of loss tensor as a standard Python number
            perf[split] = losses.mean()
        self.train()  # turn-on training mode-
        return perf

    def prep(self, corpus):
        self.vocab = sorted(list(set(corpus)))
        self.vocab_size = len(self.vocab)
        c2i = {c: i for i, c in
               enumerate(self.vocab)}  # char c to integer i map. assign value i for every word in vocab
        i2c = {i: c for c, i in c2i.items()}  # integer i to char c map

        self.encoder = lambda doc: [c2i[c] for c in doc]
        self.decoder = lambda nums: ''.join([i2c[i] for i in nums])

        n = len(text)
        self.train_text = text[:int(n * 0.9)]
        self.val_text = text[int(n * 0.9):]

        self.train_data = torch.tensor(self.encoder(self.train_text), dtype=torch.long)
        self.val_data = torch.tensor(self.encoder(self.val_text), dtype=torch.long)

        # look-up table for embeddings (vocab_size x embed_size)
        # it will be mapping each token id to a vector of embed_size
        # a wrapper to store vector representations of each token
        self.token_embeddings_table = \
            nn.Embedding(self.vocab_size, self.embed_size)
        # self-attention head
        self.sa_head = self.SelfAttentionHead(in_size=self.embed_size,
                                              out_size=self.sa_head_size)
        # linear projection of sa_head output to vocabulary
        self.linear_sahead_to_vocab = nn.Linear(self.sa_head_size, self.vocab_size)

    def get_batch(self, split='train'):
        data = self.train_data if split == 'train' else self.val_data
        # get random chunks of length batch_size from data
        ix = torch.randint(len(data) - self.input_length,
                           (self.batch_size,))
        inputs_batch = torch.stack([data[i:i + self.input_length] for i in ix])
        targets_batch = torch.stack([data[i + 1:i + self.input_length + 1] for i in ix])
        inputs_batch = inputs_batch.to(self.device)
        targets_batch = targets_batch.to(self.device)
        # inputs_batch is
        return inputs_batch, targets_batch


# with open('./tiny_shakespear.txt', 'r') as f:
#     text = f.read()

text = 'a quick brown fox jumps over the lazy dog.\n ' \
       'lazy dog and a quick brown fox.\n' \
       'the dog is lazy and the fox jumps quickly.\n' \
       'a fox jumps over the dog because he is lazy.\n' \
       'dog is lazy and fox is brown. she quickly jumps over the lazy dog.'

# text = 'go long ago longer a go go go go long ago longer go go go go long ago longer go go go'

model = SelfAttentionLanguageModel(batch_size=32,
                                   input_length=8,
                                   embed_size=32,
                                   sa_head_size=32)
model = model.to(model.device)
model.prep(text)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
print(f'params {sum([np.prod(p.size()) for p in model_parameters])}')
input_batch, output_batch = model.get_batch(split='train')
_, _ = model(input_batch, output_batch)
model.fit(train_iters=2500, eval_iters=500, lr=1e-3)
outputs = model.generate(context_tokens=torch.zeros((1, 1),
                         dtype=torch.long,
                         device=model.device),
                         max_new_tokens=100)
print(outputs)


def look_inside():
    import numpy as np
    for token in model.vocab:
        token_code = model.encoder(token)
        # print('token ', token,'token code ', token_code)
        token_rep = model.token_embeddings_table(torch.tensor(token_code))
        tr = token_rep.detach().numpy()[0].tolist()
        # print('token_rep', tr)
        next_token_prob = np.round(F.softmax(token_rep, dim=1).detach().numpy()[0], 2).tolist()
        # print(next_token_prob)
        d = [(t, round(next_token_prob[i], 2)) for i, t in enumerate(model.vocab)]
        print(token, d)
