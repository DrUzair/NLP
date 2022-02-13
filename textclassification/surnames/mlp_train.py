from torch import nn
from data import *
from mlp_model import MultilayerPerceptron
import random
import time
import math
from torch.autograd import Variable
import matplotlib.pyplot as plt

n_hidden = 128
n_epochs = 100000
print_every = 5000
plot_every = 1000
learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return all_categories[category_i], category_i

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(lineToTensor(line))
    return category, line, category_tensor, line_tensor


mlp = MultilayerPerceptron(input_dim=n_letters, hidden_dim=100, output_dim=n_categories)
optimizer = torch.optim.SGD(mlp.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()


def train(category_tensor, line_tensor):
    # step 1. zero the gradients
    optimizer.zero_grad()

    # step 2. compute the output
    output = mlp(line_tensor)

    # step 3. compute the loss
    loss = criterion(output, category_tensor)

    # step 4. use loss to produce gradients
    loss.backward()

    # step 5. use optimizer to take gradient step
    optimizer.step()

    return output, loss.item()

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for epoch in range(1, n_epochs + 1):
    category, line, category_tensor, line_tensor = randomTrainingPair()
    tensor_1d = line_tensor.sum(0)
    output, loss = train(category_tensor, tensor_1d)
    current_loss += loss

    # Print epoch number, loss, name and guess
    if epoch % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0


torch.save(mlp, 'mlp-classification.pt')

print(all_losses)
plt.figure()
plt.plot(all_losses)