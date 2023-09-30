import torch
import torch.nn as nn
from dwsnet.models import DWSModel
from dwsnet.layers import  InvariantLayer, ReLU

def google_nonlinear(x):
    return torch.sign(x) * (torch.sqrt(abs(x) + 1) - 1) + 0.001 * x


def init_params_gauss(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find("Conv") != -1:
        nn.init.uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

class ConcatWeights(nn.Module):
    def __init__(self,):
        super(ConcatWeights, self).__init__()

    def forward(self, x, start_dim=1):
        weights, biases = x
        W = torch.concat([w.flatten(start_dim=start_dim) for w in weights], dim=-1)
        B = torch.concat([b.flatten(start_dim=start_dim) for b in biases], dim=-1)
        return torch.concat([W, B], dim=-1)



class NeuralBanditModelDWS(nn.Module):
    def __init__(self, hparams ):
        super(NeuralBanditModelDWS, self).__init__()
        self.state_based = hparams.state_based_value
        self.no_embedding = hparams.no_embedding
        self.output_features = hparams.output_features
        self.non_linear_func = nn.ReLU
        self.use_invariant_layer = hparams.use_invariant_layer
        if self.no_embedding:
            self.latent_dim = 0
            for size in hparams.weight_shapes:
                self.latent_dim += size.numel()
            for size in hparams.bias_shapes:
                self.latent_dim += size.numel()
        else:
            if self.use_invariant_layer:
                self.latent_dim = hparams.latent_dim
            else:
                self.latent_dim = 0
                for size in hparams.weight_shapes:
                    self.latent_dim += size.numel()  * self.output_features
                for size in hparams.bias_shapes:
                    self.latent_dim += size.numel()  * self.output_features

        self.non_linear_func = nn.ReLU

        self.hparams = hparams
        policy_embedder_layers  = []
        policy_embedder_layers.append(DWSModel(
            weight_shapes=hparams.weight_shapes,
            bias_shapes=hparams.bias_shapes,
            input_features=1,
            hidden_dim=hparams.dim_hidden,
            output_features= self.output_features,
            n_hidden=hparams.n_hidden,
            reduction=hparams.reduction,
            n_fc_layers=hparams.n_fc_layers,
            set_layer=hparams.set_layer,
            dropout_rate=hparams.do_rate,
            bn=hparams.add_bn,
            add_skip=hparams.add_skip,
            add_layer_skip=hparams.add_layer_skip))
        # policy_embedder_layers.append(ReLU())
        if self.use_invariant_layer:
            policy_embedder_layers.append(
                InvariantLayer(
                weight_shapes=hparams.weight_shapes,
                bias_shapes=hparams.bias_shapes,
                in_features=self.output_features,
                out_features=self.latent_dim,
                reduction= hparams.reduction,
                n_fc_layers= hparams.n_out_fc,
                bias=True
            ))
        else:
            policy_embedder_layers.append(ConcatWeights())
        self.policy_embedder = nn.Sequential(*policy_embedder_layers)
        self.relu = ReLU()

        clf_layers = []
        for k in range(hparams.clf_layers - 1):
            clf_layers.append(nn.Linear(self.latent_dim, self.latent_dim, bias=True))
            if hparams.add_bn:
                clf_layers.append(nn.BatchNorm1d(self.latent_dim))
            clf_layers.append(self.non_linear_func())
        clf_layers.append(nn.Linear(self.latent_dim, self.latent_dim, bias=True))
        self.clf = nn.Sequential(*clf_layers)

        self.last_layer = (nn.Linear(self.latent_dim, 1, bias=False))

    def forward(self,  policy, encode_policy=True ):
        if self.no_embedding:
            z = self.concat_weights(policy)
            return torch.zeros_like(z[:, 0]), z
        else:
            if encode_policy:
                self.encoded_policy = self.policy_embedder(policy)
            z = self.encoded_policy
            z = self.clf(z)
            mean_value = self.last_layer(z)
            return mean_value, z

    def forward_sample(self, policy, encode_policy=True ):
        mean_value, z = self.forward(policy, encode_policy )
        return mean_value, z, None, None


    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def encode(self, state, policy):
        if self.no_embedding:
            z = self.concat_weights(policy)
            if self.state_based:
                z = torch.cat([z, state], dim=-1)
                z = self.mixed_latent(z)
            return z, None
        else:
            z = self.policy_embedder(policy)
            if self.state_based:
                state_latent = self.state_embedder(state)
                if z.shape[0] != state_latent.shape[0] and z.shape[0] == 1:
                    z = torch.cat([z for _ in range(state_latent.shape[0])],dim=0)
                z = torch.cat([z, state_latent], dim=-1)
                z = self.mixed_latent(z)
            z = self.clf(z)
            return z, None

    # def decode(self, z):
    #     return self.decoder(z)

    def concat_weights(self,x,start_dim=1):
        weights, biases = x
        W = torch.concat([w.flatten(start_dim=start_dim) for w in weights], dim=-1)
        B = torch.concat([b.flatten(start_dim=start_dim) for b in biases], dim=-1)
        return torch.concat([W,B],dim=-1)

    def sample(self, state, policy):
        x, _ = self.encode(state,policy)
        return x


    def get_last_layer_weights(self):
        return self.last_layer.weight.data[0]