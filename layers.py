import torch
import torch.nn as nn


class Conv1D(nn.Module):
    
    def __init__(self, input_dim, n_layers=1, window=5, hidden=64, dropout=0.0, use_batch_norm=False):
        super().__init__()
        self.input_dim = input_dim
        if isinstance(window, int):
            window = [window]
        if isinstance(hidden, int):
            hidden = [hidden] * len(window)
        self.n_layers = n_layers
        self.window = window
        self.hidden = hidden
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.convolutions = nn.ModuleList()
        for _ in range(self.n_layers):
            convolutions = nn.ModuleList()
            for i, (curr_window, curr_hidden) in enumerate(zip(self.window, self.hidden)):
                layer = nn.Conv1d(input_dim, curr_hidden, curr_window, padding=(curr_window // 2))
                convolutions.append(layer)
            curr_layer = {"convolutions": convolutions, "dropout": nn.Dropout(dropout)}
            if self.use_batch_norm:
                curr_layer["batch_norm"] = nn.BatchNorm1d(self.output_dim)
            self.convolutions.append(nn.ModuleDict(curr_layer))
            input_dim = self.output_dim
            
    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        for layer in self.convolutions:
            outputs = []
            for sublayer, curr_window in zip(layer["convolutions"], self.window):
                curr_output = sublayer(inputs)
                if curr_window % 2 == 0:
                    curr_output = curr_output[..., :-1]
                outputs.append(curr_output)
            outputs = torch.cat(outputs, dim=1)
            if self.use_batch_norm:
                outputs = layer["batch_norm"](outputs)
            outputs = torch.nn.ReLU()(outputs)
            outputs = layer["dropout"](outputs)
            inputs = outputs
        outputs = outputs.permute(0, 2, 1)
        return outputs

    @property
    def output_dim(self):
        return sum(self.hidden)
        
class OneSideConv1D(nn.Module):
    
    def __init__(self, input_dim, n_layers=1, window=5, hidden=64, dropout=0.0, use_batch_norm=False):
        assert n_layers > 0, "The number of layers should be positive."
        super().__init__()
        self.input_dim = input_dim
        if isinstance(window, int):
            window = [window]
        if isinstance(hidden, int):
            hidden = [hidden] * len(window)
        self.n_layers = n_layers
        self.window = window
        self.hidden = hidden
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.convolutions = nn.ModuleList()
        for layer_num in range(self.n_layers):
            left_convolutions = nn.ModuleList()
            right_convolutions = nn.ModuleList()
            for i, (curr_window, curr_hidden) in enumerate(zip(self.window, self.hidden)):
                if layer_num > 0:
                    input_dim = curr_hidden
                left_layer = nn.Conv1d(input_dim, curr_hidden, curr_window)
                left_convolutions.append(left_layer)
                right_layer = nn.Conv1d(input_dim, curr_hidden, curr_window)
                right_convolutions.append(right_layer)
            curr_layer = {"left": left_convolutions, "right": right_convolutions,
                          "left_dropout": nn.Dropout(dropout), "right_dropout": nn.Dropout(dropout)}
            if self.use_batch_norm:
                curr_layer["left_batch_norms"] = nn.ModuleList([
                    nn.BatchNorm1d(num_features=curr_hidden) for curr_hidden in self.hidden
                ])
                curr_layer["right_batch_norms"] = nn.ModuleList([
                    nn.BatchNorm1d(num_features=curr_hidden) for curr_hidden in self.hidden
                ])
            self.convolutions.append(nn.ModuleDict(curr_layer))

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        left_inputs = [inputs[:] for _ in range(self.n_windows)]
        right_inputs = [inputs[:] for _ in range(self.n_windows)]
        for layer in self.convolutions:
            left_outputs, right_outputs = [], []
            for sublayer, curr_window, curr_input in zip(layer["left"], self.window, left_inputs):
                if curr_window > 1:
                    curr_input = torch.cat([
                            torch.zeros_like(curr_input[...,:curr_window-1]), curr_input
                        ], dim=-1)
                curr_output = sublayer(curr_input)
                left_outputs.append(curr_output)
            if self.use_batch_norm:
                left_outputs = [sublayer(curr_output)
                                for sublayer, curr_output in zip(layer["left_batch_norms"], left_outputs)
                               ]
            left_outputs = torch.cat(left_outputs, dim=1)
            left_outputs = torch.nn.ReLU()(left_outputs)
            left_outputs = layer["left_dropout"](left_outputs)
            left_inputs = torch.split(left_outputs, self.hidden, dim=1)
            right_outputs = []
            for sublayer, curr_window, curr_input in zip(layer["right"], self.window, right_inputs):
                if curr_window > 1:
                    curr_input = torch.cat([
                            curr_input, torch.zeros_like(curr_input[...,:curr_window-1])
                        ], dim=-1)
                curr_output = sublayer(curr_input)
                right_outputs.append(curr_output)
            if self.use_batch_norm:
                right_outputs = [sublayer(curr_output)
                                 for sublayer, curr_output in zip(layer["right_batch_norms"], right_outputs)
                                ]
            right_outputs = torch.cat(right_outputs, dim=1)
            right_outputs = torch.nn.ReLU()(right_outputs)
            right_outputs = layer["right_dropout"](right_outputs)
            right_inputs = torch.split(right_outputs, self.hidden, dim=1)
        output = torch.cat([left_outputs, right_outputs], dim=1)
        output = output.permute(0, 2, 1)
        return output
    
    @property
    def oneside_dim(self):
        return sum(self.hidden)
    
    @property
    def output_dim(self):
        return 2 * sum(self.hidden)

    @property
    def n_windows(self):
        return len(self.window)