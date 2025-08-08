import torch
import torch.nn as nn


class RNNStep(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        alpha: float,
        activation: type[nn.Module] = nn.ReLU, #changed to ReLU
        place_cells=None,
    ) -> None:
        """
        A single time step of the RNN.
        """
        super(RNNStep, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.activation = activation()
        self.place_cells = place_cells

        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_rec = nn.Linear(hidden_size, hidden_size)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        h = (1 - self.alpha) * hidden + self.alpha * self.activation(
            self.W_in(input) + self.W_rec(hidden)
        )
        return h


class PathIntRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        alpha: float = 0.1,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        """
        Initialize the Path Integrating RNN.
        :param input_size: The size of the velocity input (= dimension of space).
        :param hidden_size: The size of the hidden state (number of neurons/"grid cells").
        :param output_size: The size of the output vector (dimension of space).
        :param alpha: RNN update rate.
        :param activation: The activation function.
        """
        super(PathIntRNN, self).__init__()
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.output_size: int = output_size

        self.rnn_step = RNNStep(input_size, hidden_size, alpha, activation)
        self.W_out = nn.Linear(hidden_size, output_size)

        # Layer to initialize hidden state
        self.W_h_init = nn.Linear(2, hidden_size)

        self.initialize_weights()

    def forward(
        self, inputs: torch.Tensor, pos_0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # inputs has shape (batch_size, time_steps, input_size)
        # pos_0 has shape (batch_size, 2)
        hidden_states = []
        outputs = []
        hidden = torch.tanh(self.W_h_init(pos_0))
        for t in range(inputs.shape[1]):
            input_t = inputs[:, t, :]
            hidden = self.rnn_step(input_t, hidden)
            hidden_states.append(hidden)
            outputs.append(self.W_out(hidden))
        return torch.stack(hidden_states, dim=1), torch.stack(outputs, dim=1)

    def initialize_weights(self) -> None:
        """Initialize weights for stable RNN training"""
        # 1. Input weights (W_in) - Xavier initialization
        nn.init.xavier_uniform_(self.rnn_step.W_in.weight)
        nn.init.zeros_(self.rnn_step.W_in.bias)

        # 2. Recurrent weights (W_rec) - Orthogonal initialization
        nn.init.orthogonal_(self.rnn_step.W_rec.weight)
        nn.init.zeros_(self.rnn_step.W_rec.bias)

        # 3. Output weights (W_out) - Xavier initialization
        nn.init.xavier_uniform_(self.W_out.weight)
        nn.init.zeros_(self.W_out.bias)

        # 4. Initial hidden state encoder (W_h_init) - Xavier initialization
        nn.init.xavier_uniform_(self.W_h_init.weight)
        nn.init.zeros_(self.W_h_init.bias)