from itertools import zip_longest
import numpy as np
import pandas as pd
import torch
import torch.nn
from inr.activation_functions import Sine, WIRE, Relu, ComplexWIRE
from inr.pos_encoding import (
    PosEncodingNeRFOptimized,
    PosEncodingGaussian,
    PosEncodingNone,
    PosEncodingNeRF,
)


class Network(torch.nn.Module):
    def __init__(
        self,
        device,
        hidden_size,
        input_size,  # number of input dimensions (txyz)
        num_layers,
        num_time_layers,
        activation,
        time_activation,
        wire_omega_0,
        wire_sigma_0,
        mlp_type,
        pos_encoding,
        num_frequencies,
        freq_scale,
        output_size=1,
        use_sep_spacetime=True,
    ):
        super().__init__()
        self.mlp_type = mlp_type
        self.activation_type = activation
        self.use_separate_space_time = use_sep_spacetime
        self.num_layers = num_layers
        self.num_time_layers = num_time_layers
        self.hidden_size = hidden_size

        if self.use_separate_space_time:
            assert num_time_layers > 0 and (num_time_layers < num_layers), (
                f"cannot use {self.use_separate_space_time=} while {self.num_time_layers=} and {self.num_layers=}"
            )
            # decrement time from input size
            # only use pos encoding for xyz
            input_size -= 1

        activation_module = dict(
            sine=Sine, wire=WIRE, relu=Relu, complexwire=ComplexWIRE
        )[activation]
        time_activation_module = dict(
            sine=Sine, wire=WIRE, relu=Relu, complexwire=ComplexWIRE
        )[time_activation]

        # add positional encoding
        self.position_encoding = dict(
            none=PosEncodingNone,
            nerfoptimized=PosEncodingNeRFOptimized,
            gaussian=PosEncodingGaussian,
            nerf=PosEncodingNeRF,
        )[pos_encoding](
            input_size=input_size,
            num_frequencies=num_frequencies,
            freq_scale=freq_scale,
        )
        # if encoding is not enabled, encoding_size = input_size
        # if encoding is enabled and using separate spacetime, encoding_size = encoding(xyz)
        # if encoding is enabled and using combined spacetime, encoding_size = encoding(txyz)
        self.encoding_size = self.position_encoding.get_encoding_size()

        # Since complex numbers are two real numbers, reduce the number of hidden parameters by 2
        if self.activation_type == "complexwire":
            self.hidden_size = int(hidden_size / np.sqrt(2))

        # build global base network
        self.layers = torch.nn.ModuleList()
        self.time_layers = torch.nn.ModuleList()

        # add first layer
        self.layers.append(
            activation_module(
                self.encoding_size,
                hidden_size,
                is_first=True,
                wire_omega_0=wire_omega_0,
                wire_sigma_0=wire_sigma_0,
            )
        )
        if self.use_separate_space_time:
            self.time_layers.append(
                time_activation_module(
                    1,
                    hidden_size,
                    is_first=True,
                    wire_omega_0=wire_omega_0,
                    wire_sigma_0=wire_sigma_0,
                )
            )

        # add hidden layers
        for k in range(num_layers - 2):
            self.layers.extend(
                [
                    activation_module(
                        hidden_size,
                        hidden_size,
                        is_first=False,
                        wire_omega_0=wire_omega_0,
                        wire_sigma_0=wire_sigma_0,
                    )
                ]
            )
            if self.use_separate_space_time and (k < self.num_time_layers - 1):
                self.time_layers.extend(
                    [
                        time_activation_module(
                            hidden_size,
                            hidden_size,
                            is_first=False,
                            wire_omega_0=wire_omega_0,
                            wire_sigma_0=wire_sigma_0,
                        )
                    ]
                )

        # add last layer
        self.layers.append(
            activation_module(
                hidden_size,
                output_size,
                is_first=False,
                wire_omega_0=wire_omega_0,
                wire_sigma_0=wire_sigma_0,
            )
        )

    def forward(self, txyz):
        if self.use_separate_space_time:
            x = self.position_encoding(
                txyz[:, :, 1:]
            )  # shape: batch size, sampled voxels, 3(xyz)
        else:
            x = self.position_encoding(
                txyz
            )  # shape: batch size, sampled voxels, 4(txyz)

        last_k = 0
        xt = txyz[:, :, 0].unsqueeze(-1)
        for k, (layer, t_layer) in enumerate(
            zip_longest(self.layers, self.time_layers, fillvalue=None)
        ):
            x_prev = x
            xt_prev = xt
            x = layer(x)
            if self.use_separate_space_time and t_layer is not None:
                xt = t_layer(xt)

            # skip connections
            if self.mlp_type == "fullresidual":
                if layer.in_size == layer.out_size:
                    x = (x + x_prev) / 2
                if (
                    self.use_separate_space_time
                    and t_layer is not None
                    and t_layer.in_size == t_layer.out_size
                ):
                    xt = (xt + xt_prev) / 2
            elif self.mlp_type == "skip2residual":
                if layer.in_size == layer.out_size and k >= last_k + 2:
                    last_k = k
                    x = (x + x_prev) / 2
                if (
                    self.use_separate_space_time
                    and t_layer is not None
                    and t_layer.in_size == t_layer.out_size
                    and k >= last_k + 2
                ):
                    xt = (xt + xt_prev) / 2
            elif self.mlp_type == "none":
                pass
            else:
                raise NotImplementedError()

            # if using separate space time, add hidden outputs after passing through all time layers
            if self.use_separate_space_time and k == self.num_time_layers - 1:
                x += xt

        # NOTE: probably don't need this check, just return all real
        if self.activation_type == "complexwire":
            return x.real
        return x

    def get_global_parameters(self):
        params = list(self.layers.parameters())
        if self.use_separate_space_time:
            params += list(self.time_layers.parameters())

        return params
