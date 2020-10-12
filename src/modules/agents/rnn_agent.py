import torch.nn as nn
import torch.nn.functional as F


# TODO: What kind of architecture is this RNN?
class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # f.e. inputs of shape [42, 3] arrive here -> 42-dim observation of 3 agents.
        # Linear transformation + pass result into rectified linear unit function element-wise
        x = F.relu(self.fc1(inputs))
        # Bring hidden state received from caller into correct shape for rnn(GRUCell)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        # Pass transformed input and hidden state into gated recurrent unit
        h = self.rnn(x, h_in)
        # Linear transformation from hidden state to q-value of each agent
        q = self.fc2(h)
        return q, h
