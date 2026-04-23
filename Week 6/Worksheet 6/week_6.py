import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Audio Machine Learning - Workshop Week 6 - Recurrent Neural Networks
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0 - Import Libraries
    """)
    return


@app.cell
def _():
    # '%pip install librosa' command supported automatically in marimo
    import librosa
    import IPython
    import numpy as np
    import torch
    import scipy.signal
    from torch.utils.data import Dataset, DataLoader
    from tqdm import tqdm

    return Dataset, IPython, librosa, np, scipy, torch, tqdm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 - Recurrent Layer in PyTorch
    """)
    return


@app.cell
def _(torch):
    # This create a basic RNN layer in PyTorch
    rec_layer = torch.nn.RNN(input_size=1, hidden_size=8, batch_first=True)
    return (rec_layer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this case:\
    'input_size' is the number of chanels in the Recurrent Layer input\
    'hidden_size' is the number of elements in the Recurrent Layer hidden state
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### ✏️✏️ Exercise: Using a Recurrent Layer  ✏️✏️
    ---
    Referring to the documentation:

        https://pytorch.org/docs/stable/generated/torch.nn.RNN.html

    Create a random input to be processed by the recurrent layer. It should have batch_size, $N$ = 4, and sequence length, $L=100$. Create this using torch.randn() and process it with the recurrent layer. Note that the recurrent layer above was called with 'batch_first=True'.
    """)
    return


@app.cell
def _(torch):
    ##TODO
    inp = torch.randn([4,100,1])
    return (inp,)


@app.cell
def _(inp, rec_layer, torch):
    # Check your answer
    assert rec_layer(inp)[0].shape == torch.Size([4, 100, 8])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now process the input with the recurrent layer. Use the documentation to see what will be returned by the recurrent layer, and save this to appropriately named variables
    """)
    return


@app.cell
def _(inp, rec_layer):
    ##TODO process input with recurrent layer

    output, h_n = rec_layer(inp)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *NOTE*
    PyTorch 1D-Convolutional layers assume inputs have dimensions in the order:
    - (N, C, L), i.e. (Batch, Channel, Length)

    PyTorch Recurrent layers assume inputs have dimensions in the order:
    - (L,N,H), i.e. (Length, Batch, CHannel)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 - Implementing your own recurrent layer

    The state update equation shown on the PyTorch RNN documentation can be implemented by applying a linear layer to the input, and a linear layer to the hidden state, summing the result and applying the tanh() nonlinearity.

    The RNN needs an initial hidden state before it can process inputs. This should be a tensor of zeros, with the shape depending on the hidden size of the RNN and the batch size. Complete the 'init_state' method such that it initialises the hidden state.

    The RNN forward method should take an input sequence of shape (N, L, 1), where N is the batch size and L is the length of the sequence. It should then iterate over the sequence, applying the RNN state update equation.

    When referring to the shapes of tensors in the RNN documentation, for this exercise $D=1$ and $num\_layers=1$.

    ---
    ### ✏️✏️ Exercise: Complete the Recurrent Layer Class ✏️✏️
    ---
    Complete this template for your own implementation of the recurrent layer.
    """)
    return


app._unparsable_cell(
    r"""
    class myRNN(torch.nn.Module):
        def __init__(self, input_channels, hidden_size):
            super(myRNN, self).__init__()
            self.inp_lin = torch.nn.Linear(input_channels) ##TODO  Set the input/output size of the input layer
            self.rec_lin = torch.nn.Linear(hidden_size) ##TODO Set the input/output size of the hidden-to-hidden layer
            self.non_lin = torch.nn.Tanh() ## Do Not Edit!
            self.hs = hidden_size          ## Do Not Edit!

        def init_state(self, batch_size):
            self.state = zeros()##TODO - Initialise the state with 0s, according to the shape of the hidden state
                         # 'hx' specified on the RNN documentation page

        ## x is the RNN input, of shape (N, L, 1)
        def forward(self, x):
            output = [] #  Do Not Edit! - For each time step, the updated hidden state should be appended to this list
            for n in range(x.shape[1]): #  Do Not Edit! - Create a loop over the time dimension of the sequence
                inp_step = x[:, n, :]   #  Do Not Edit! - Get a single time step from the input batch 

                self.state  = ## TODO implement the state update from the torch RNN page, using
                                # the linear layers and tanh layer defined in the constructor

                output.append(self.state.squeeze()) # Do Not Edit! - append the state to the list of hidden states

            return torch.stack(output, dim=1), self.state #  Do Not Edit!
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we can test if your RNN implementation gives the same result as the PyTorch one!

    First, create a PyTorch RNN and an RNN from the template defined above:
    """)
    return


@app.cell
def _(myRNN, torch):
    model = myRNN(1, 8) # Create RNN from Class template
    rnn = torch.nn.RNN(input_size=1, hidden_size=8, batch_first=True) # Create PyTorch RNN
    return model, rnn


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The Parameters of the RNNs are set randomly, when the RNN is created. The below code copies the parameters from the PyTorch RNN, to the RNN model you defined.
    """)
    return


@app.cell
def _(model, rnn):
    model.inp_lin.bias.data = rnn.bias_ih_l0.data
    model.inp_lin.weight.data = rnn.weight_ih_l0.data
    model.rec_lin.bias.data = rnn.bias_hh_l0.data
    model.rec_lin.weight.data = rnn.weight_hh_l0.data
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now the parameters of the two RNNs are the same, when processing the same input, they should produce the same output:
    """)
    return


@app.cell
def _(inp, model, rnn):
    # inp = torch.randn(4, 20, 1) # Create input to process with RNN

    model.init_state(4)   # Initialise hidden state
    states, h_t = model(inp) # Process inp with your RNN class

    states_pt, h_t_pt = rnn(inp) # Process inp with PyTorch RNN class
    return h_t, h_t_pt, states, states_pt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First, we will check the shapes of the returns:
    """)
    return


@app.cell
def _(states, states_pt):
    assert states_pt.shape == states.shape, f'PyTorch RNN states over time has shape {states_pt.shape}, whereas states over time returned by myRNN has shape {states.shape}'
    return


@app.cell
def _(h_t, h_t_pt):
    assert h_t.shape == h_t_pt.shape, f'PyTorch RNN final state {h_t_pt.shape}, whereas final state returned by myRNN has shape {h_t.shape}'
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If either of these assertions fail, check your RNN class, to see if the shapes of the various tensors are as expected.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Second, we will check the values of the tensors in the returns:
    """)
    return


@app.cell
def _(h_t, h_t_pt, states, states_pt, torch):
    assert torch.max(torch.abs(states_pt - states)) < 1e-6
    assert torch.max(torch.abs(h_t_pt - h_t)) < 1e-6
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3 - Compare speed of RNN implementations
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we will do a simple comparison of the speeds of both RNN implementations, using 'tqdm'.

    tqdm creates a handy progress bar, and also tells you the amount of time per iteration it takes to complete a for loop.
    """)
    return


@app.cell
def _(rnn, torch, tqdm):
    inp_1 = torch.randn(10, 10000, 1)
    tgt = torch.randn(10, 10000, 8)
    print('Processing input with PyTorch RNN layer')
    for n in tqdm(range(10)):
        y_hat, _ = rnn(inp_1)
        l = torch.mean(torch.abs(y_hat - tgt))
        l.backward()
    return inp_1, tgt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You should see a progress bar, then a number followed by either 'it/s' or by 's/it'

    'it/s' means the number of iterations of the for-loop completed per second, as 's/it' means the number of seconds taken to complete a single iteration of the for loop.

    For the above, with the PyTorch RNN, it should be completing around 1 s/it. That means it processes an entire batch of ten 10000 length inputs, then backpropagates the loss through this in about one second.

    Now compare this with the custom implementation:
    """)
    return


@app.cell
def _(inp_1, model, tgt, torch, tqdm):
    print('Processing input with myRNN layer')
    for n_1 in tqdm(range(10)):
        model.init_state(10)
        y_hat_1, _ = model(inp_1)
        l_1 = torch.mean(torch.abs(y_hat_1 - tgt))
        l_1.backward()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This should take about 3 second per iteration. Much slower!

    A major reason for this is that when PyTorch does calculations, it actually calls a C++ library called 'torch' to do the maths. When using the PyTorch RNN class, PyTorch passes the entire sequence of inputs to the underlying C++ function.

    In the myRNN library, the C++ torch library is still being called to calculate the linear layer outputs, but now it is being called once every time-step in the input sequence! There is some overhead everytime Python calls C++, so this results in it being much slower compared to the PyTorch RNN, which only has to call the C++ library once for the entire sequence.

    To create your own custom PyTorch modules that include recursions (like the RNN), implementing the recursion in C++ will greatly improve processing speed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 4 - Create a Distortion Effect Dataset

    For this section, we will create a distortion effect dataset, using clean guitar.
    """)
    return


@app.cell
def _(librosa):
    inp_2, fs = librosa.load('guitar-input.wav')  # Load the input guitar signal
    return fs, inp_2


@app.cell
def _(np, scipy):
    # Basic Distortion Effect - Low-Pass filter followed by saturating nonlinearity
    def Distfx(in_sig, gain):
        b = np.array([0.00460399444634034, 0.00920798889268068, 0.00460399444634034])
        a = np.array([1.0, -1.7990948352036205, 0.8175108129889816])
        out = scipy.signal.lfilter(b, a, in_sig)
        dist_out = np.tanh(10**(gain/20)*out)
        return dist_out

    return (Distfx,)


@app.cell
def _(Distfx, inp_2):
    tgt_1 = Distfx(inp_2, 40)  # Create the target distorted guitar audio
    return (tgt_1,)


@app.cell
def _(IPython, fs, inp_2):
    IPython.display.Audio(data=inp_2[207 * fs:212 * fs], rate=fs)
    return


@app.cell
def _(IPython, fs, tgt_1):
    IPython.display.Audio(data=tgt_1[207 * fs:212 * fs], rate=fs)
    return


@app.cell
def _(fs, inp_2, tgt_1, torch):
    # Split into train, validation and test set.
    inp_tensor = torch.tensor(inp_2).unsqueeze(0).float()
    tgt_tensor = torch.tensor(tgt_1).unsqueeze(0).float()
    inp_train = inp_tensor[:, 2 * fs:242 * fs].T
    tgt_train = tgt_tensor[:, 2 * fs:242 * fs].T  # 4 minutes of train data
    inp_val = inp_tensor[:, 242 * fs:272 * fs].T
    tgt_val = tgt_tensor[:, 242 * fs:272 * fs].T
    inp_test = inp_tensor[:, 272 * fs:302 * fs].T  # 30 seconds of validation data
    tgt_test = tgt_tensor[:, 272 * fs:302 * fs].T  # 30 seconds of test data
    return inp_train, tgt_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Below is the dataset class:
    """)
    return


@app.cell
def _(Dataset):
    class AudioDataSet(Dataset): 
        def __init__(self, input_audio, target_audio, len_samp):
            # Our class constructor takes the input and target audio streams as arguments, and saves them
            self.input_audio = input_audio
            self.target_audio = target_audio
            self.len_samp = len_samp

        def __getitem__(self, i):
            # The getitem method returns a segment of audio, that is 'len_samp' samples long
            start_samp = i*self.len_samp
            end_samp = (i+1)*self.len_samp
            x = self.input_audio[start_samp:end_samp, :]
            y = self.target_audio[start_samp:end_samp, :]
            return x, y

        def __len__(self):
            # The len method returns the total number of datapoints in the dataset
            return self.input_audio.shape[0]//self.len_samp

    return (AudioDataSet,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 5 - Training a Distortion Model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Lets train a distortion audio effect model using a Recurrent Neural Network.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Below is a RNN based distortion effects model, that can either use a basic RNN or a GRU:
    """)
    return


@app.cell
def _(torch):
    class RNNDist(torch.nn.Module):
        def __init__(self, input_size=1, hidden_size=8, output_size=1, unit_type='RNN'):
            super(RNNDist, self).__init__()
            self.input_size = input_size
            self.output_size = output_size
            if unit_type == 'RNN':
                self.rec = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
            elif unit_type == 'GRU':
                self.rec = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
            self.lin = torch.nn.Linear(hidden_size, output_size)
            self.hidden = None

        def forward(self, x):
            x, self.hidden = self.rec(x, self.hidden)
            return self.lin(x)

        # detach hidden state, this resets gradient tracking on the hidden state, for truncated backpropagation through time
        def detach_hidden(self):
            if self.hidden.__class__ == tuple:
                self.hidden = tuple([h.clone().detach() for h in self.hidden])
            else:
                self.hidden = self.hidden.clone().detach()

        # changes the hidden state to None, causing pytorch to create an all-zero hidden state when the rec unit is called
        def reset_hidden(self):
            self.hidden = None

    return (RNNDist,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here is the 'Error-To-Signal' Ratio loss function:
    """)
    return


@app.cell
def _(torch):
    # ESR loss calculates the Error-to-signal between the output/target
    class ESRLoss(torch.nn.Module):
        def __init__(self):
            super(ESRLoss, self).__init__()
            self.epsilon = 0.00001

        def forward(self, output, target):
            loss = torch.add(target, -output)
            loss = torch.pow(loss, 2)
            loss = torch.mean(loss)
            energy = torch.mean(torch.pow(target, 2)) + self.epsilon
            loss = torch.div(loss, energy)
            return loss

    return (ESRLoss,)


@app.cell
def _(ESRLoss):
    # Create loss_fcn object
    loss_fcn = ESRLoss()
    return


@app.cell
def _(RNNDist, torch):
    # Create RNN model and optimiser
    model_1 = RNNDist(input_size=1, output_size=1, hidden_size=8, unit_type='RNN')
    optimiser = torch.optim.Adam(model_1.parameters(), lr=0.0001)
    return


@app.cell
def _():
    loss_list = []
    return


@app.cell
def _(AudioDataSet, inp_train, tgt_train):
    tbptt_step = 1024
    epochs = 20

    # Create our training dataset.
    train_dataset = AudioDataSet(inp_train, tgt_train, 10000)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### ✏️✏️ Exercise: Implementing RNN Training ✏️✏️
    ---

    Your task is to implement the training loop:

    The training loop will carry out 'truncated backpropagation through time'.

    For each batch, the loop should do the following:
    1. Set the model hidden state to 'None'
    2. Initiate a for-loop that processes the batch sequentially, in segments of length 'tbptt_step', i.e each segment of the batch should be of shape [N, tbptt_step, 1]
    3. For each segment:
        1. Zero the gradients in the optimiser
        2. Get the model predictions for that segment
        3. Calculate the loss
        4. Do the backward pass
        5. call `optimiser.step()`
        6. Deatch the hidden state, using `model.detach_hidden()`
        7. append the loss to the loss_list
    """)
    return


app._unparsable_cell(
    r"""
    for n in range(epochs):
        print('start_epoch')
        for x,y in tqdm(DataLoader(train_dataset, 32, shuffle=True)): # Iterate over the Dataset
            ## TODO - Complete Training Loop!
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Further Work
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Validation Loss:
    - Create a validation dataset with segment length of 5 seconds
    - After each training epoch, use the model to process the validation dataset
    - Find the average loss on the validation dataset, and save it to a list called 'validation_losses'

    ### Validation/Test dataset:
    - After training has run for a while, process some of the validation/test set with the model, and listen to output
    - It should sound pretty similar to the target is the loss is in the region of 0.03 or so.
    """)
    return


if __name__ == "__main__":
    app.run()
