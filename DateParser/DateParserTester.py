from keras.layers import Bidirectional, Concatenate, Dot, Input, LSTM
from keras.layers import RepeatVector, Dense, Activation
from keras.models import Model
from nmt_utils import *
import json
from nmt_utils import format_date_x
import dateparser

human_vocab = json.loads(open('human_vocab.json').read())
machine_vocab = json.loads(open('machine_vocab.json').read())
inv_machine_vocab = json.loads(open('inv_machine_vocab.json').read())

Tx = int(30)
Ty = int(10)

repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor1 = Dense(10, activation = "tanh")
densor2 = Dense(1, activation = "relu")
activator = Activation(softmax, name='attention_weights') # custom softmax
dotor = Dot(axes = 1)


def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.

    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)

    Returns:
    context -- context vector, input of the next (post-attetion) LSTM cell
    """

    # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a"
    s_prev = repeator(s_prev)
    # Use concatenator to concatenate a and s_prev on the last axis
    concat = concatenator([a, s_prev])
    # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e.
    e = densor1(concat)
    # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies.
    energies = densor2(e)
    # Use "activator" on "energies" to compute the attention weights "alphas"
    alphas = activator(energies)
    # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell
    context = dotor([alphas, a])
    return context

n_a = 64
n_s = 128
post_activation_LSTM_cell = LSTM(n_s, return_state=True)
output_layer = Dense(len(machine_vocab), activation=softmax)


def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """

    # Define the inputs of your model with a shape (Tx,)
    # Define s0 and c0, initial hidden state for the decoder LSTM of shape (n_s,)
    X = Input(shape=(Tx, human_vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0

    # Initialize empty list of outputs
    outputs = []

    # pre-attention Bi-LSTM.
    a = Bidirectional(LSTM(units=n_a, input_shape=X.shape, return_sequences=True))(X)

    # Iterate for Ty steps
    for t in range(Ty):
        # one step of the attention mechanism --> get the context vector at step t
        context = one_step_attention(a, s)

        # apply the post-attention LSTM cell to the "context" vector.
        s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])

        # apply Dense layer to the hidden state output of the post-attention LSTM
        out = output_layer(s)

        # append "out" to the "outputs" list
        outputs.append(out)

    # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
    model = Model(inputs=[X, s0, c0], outputs=outputs)

    return model


if __name__ == "__main__":
    s0 = np.zeros((1, n_s))
    c0 = np.zeros((1, n_s))
    print(len(human_vocab), len(machine_vocab))
    model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
    model.summary()
    #model.load_weights("model.h5")
    model.load_weights("model_64_128.h5")

    faker = Faker()
    for year in range(1900, 2100):
        for month in range(1, 12):
            for day in range(1,28):
                dt = date(year, month, day)
                date_str = format_date_x(dt).lower()
                source = string_to_int(date_str, Tx, human_vocab)
                source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source)))
                source = np.expand_dims(source, axis=0)
                date_val = model.predict([source, s0, c0])
                date_val = np.argmax(date_val, axis=-1)
                date_val = [inv_machine_vocab[str(int(i))] for i in date_val]
                date_val = ''.join(date_val)
                date_parser_output = dateparser.parse(date_str).date()
                if date_val != dt.isoformat():
                    print(date_str, dt.isoformat(), 'model estimate', date_val, 'dateparser output', date_parser_output)
    print('all set')

