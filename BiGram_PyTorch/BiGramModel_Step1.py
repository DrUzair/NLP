class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab = None
        self.token_embeddings_table = None
        self.vocab_size = None
        self.encoder = None
        self.decoder = None
        self.vocab_size: int
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # input_length = how many consecutive tokens/chars in one input
        self.input_length = None
        # batch_size = how many inputs are going to be processed in-parallel (on GPU)
        self.batch_size = None

    def forward(self, in_ids, target=None):
        in_ids_emb = self.token_embeddings_table(in_ids) # batch_size x vocab_size
        if target is None:
            ce_loss = None
        else:
            batch_size, input_length, vocab_size = in_ids_emb.shape
            token_rep = in_ids_emb.view(batch_size * input_length, vocab_size)
            targets = target.view(batch_size * input_length)
            ce_loss = F.cross_entropy(token_rep, targets)
        return in_ids_emb, ce_loss

    def fit(self, train_iters=100, eval_iters=10, lr=0.001):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for iteration in range(train_iters):
            if iteration % eval_iters == 0:
                avg_loss = self.eval_loss(eval_iters)
                print(f"iter {iteration} train {avg_loss['train']} val {avg_loss['eval']}")
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

    def prep(self, text):
        self.vocab = sorted(list(set(text)))
        self.vocab_size = len(self.vocab)
        ctoi = {c: i for i, c in
                enumerate(self.vocab)}  # char c to integer i map. assign value i for every word in vocab
        itoc = {i: c for c, i in ctoi.items()}  # integer i to char c map

        self.encoder = lambda text: [ctoi[c] for c in text]
        self.decoder = lambda nums: ''.join([itoc[i] for i in nums])

        n = len(text)
        self.train_text = text[:int(n * 0.9)]
        self.val_text = text[int(n * 0.9):]

        self.train_data = torch.tensor(self.encoder(self.train_text), dtype=torch.long)
        self.val_data = torch.tensor(self.encoder(self.val_text), dtype=torch.long)

        # look-up table for embeddings (vocab_size x vocab_size)
        # the model will turning each input token into a vector of size vocab_size
        # a wrapper to store vector representations of each token
        self.token_embeddings_table = \
            nn.Embedding(self.vocab_size, self.vocab_size)

    def get_batch(self, split='train', input_length=8, batch_size=4):
        data = self.train_data if split == 'train' else self.val_data
        # get random chunks of length batch_size from data
        ix = torch.randint(len(data) - input_length, (batch_size,))
        inputs_batch = torch.stack([data[i:i + input_length] for i in ix])
        targets_batch = torch.stack([data[i + 1:i + input_length + 1] for i in ix])
        inputs_batch = inputs_batch.to(self.device)
        targets_batch = targets_batch.to(self.device)
        # inputs_batch is
        return inputs_batch, targets_batch

text = 'a quick brown fox jumps over the lazy dog.\n ' \
       'lazy dog and a quick brown fox.\n' \
       'a dog is lazy and fox is quick.\n' \
       'fox jumps and dog is lazy.\n' \
       'dog is lazy and fox is brown.'

model = BigramLanguageModel()
model = model.to(model.device)
model.prep(text)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
print(f'params {sum([np.prod(p.size()) for p in model_parameters])}')
input_batch, output_batch = model.get_batch(split='train')
_, _ = model(input_batch, output_batch)
model.fit(train_iters=10000, eval_iters=500, lr=0.001)

outputs = model.generate(context_tokens=torch.zeros((1, 1), dtype=torch.long,
                         device=model.device), max_new_tokens=100)
print(outputs)
