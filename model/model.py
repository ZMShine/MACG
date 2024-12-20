import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from model.cell import DCGRUCell
import numpy as np
import pandas as pd
import torch.autograd.profiler as profiler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cosine_similarity_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).to(device)  # U是从区间[0,1]均匀分布中抽取的一组随机数，生成一个size为shape的tensor
    return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))  # 增加扰动，通过Gumbel分布求逆，从U生成


def gumbel_softmax_sample(logits, temperature, eps=1e-10):
    sample = sample_gumbel(logits.size(), eps=eps)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=True, eps=1e-10):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y_soft = gumbel_softmax_sample(logits, temperature=temperature, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)  # 用于找概率最大的下标
        y_hard = torch.zeros(*shape).to(device)  # 返回一个形状为为*shape,类型为torch.dtype，里面的每一个值都是0的tensor
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)  # 用来对标签进行 one-hot 编码
        y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


class Seq2SeqAttrs:
    def __init__(self, **model_kwargs):
        # self.adj_mx = adj_mx
        self.max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.filter_type = model_kwargs.get('filter_type', 'laplacian')
        self.num_nodes = int(model_kwargs.get('num_nodes', 1))
        self.num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        self.rnn_units = int(model_kwargs.get('rnn_units'))
        self.hidden_state_size = self.num_nodes * self.rnn_units


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.input_dim = int(model_kwargs.get('input_dim', 1))
        self.seq_len = int(model_kwargs.get('seq_len'))  # for the encoder
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, adj, hidden_state=None):
        """
        Encoder forward pass.
        :param inputs: shape (batch_size, self.num_nodes * self.input_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size),
                                       device=device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adj)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, **model_kwargs):
        # super().__init__(is_training, adj_mx, **model_kwargs)
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.output_dim = int(model_kwargs.get('output_dim', 1))
        self.horizon = int(model_kwargs.get('horizon', 1))  # for the decoder
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, adj, hidden_state=None):
        """
        :param inputs: shape (batch_size, self.num_nodes * self.output_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.num_nodes * self.output_dim)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adj)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)

        return output, torch.stack(hidden_states)


class MACGModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, temperature, logger, **model_kwargs):
        super().__init__()
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.encoder_model = EncoderModel(**model_kwargs)
        self.decoder_model = DecoderModel(**model_kwargs)
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        self._logger = logger
        self.temperature = temperature
        self.dim_fc = int(model_kwargs.get('dim_fc', False))
        self.dim_fc_ol = int(model_kwargs.get('dim_fc_ol', False))

        self.embedding_dim = 100
        
        self.conv1 = torch.nn.Conv1d(1, 8, 11, dilation= 1, stride=1)  # .to(device)
        self.conv2 = torch.nn.Conv1d(8, 16, 11, dilation= 1, stride=1)  # .to(device)

        self.conv3 = torch.nn.Conv1d(1, 8, 6, dilation= 2, stride=1)  # .to(device)
        self.conv4 = torch.nn.Conv1d(8, 16, 6, dilation= 2, stride=1)  # .to(device)

        self.hidden_drop = torch.nn.Dropout(0.2)
        self.fc = torch.nn.Linear(self.dim_fc, self.embedding_dim)
        self.bn1 = torch.nn.BatchNorm1d(8)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.bn3 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.fc_out = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.fc_cat = nn.Linear(self.embedding_dim, 2)

        self.nodevec1 = nn.Parameter(torch.randn(self.num_nodes, 10).to(device), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(10, self.num_nodes).to(device), requires_grad=True)

        self.alpha = nn.Parameter(torch.tensor(0.001))

        def encode_onehot(labels):
            classes = set(labels)
            classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                            enumerate(classes)}
            labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                     dtype=np.int32)
            return labels_onehot

        # Generate off-diagonal interaction graph生成非对角交互图
        off_diag = np.ones([self.num_nodes, self.num_nodes])  # 生成列数方阵
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec).to(device)
        self.rel_send = torch.FloatTensor(rel_send).to(device)

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def FeatureExtraction(self, node_feas, temp):
        x = node_feas.transpose(1, 0).view(self.num_nodes, 1, -1)  # transpose表示转置

        x_1 = self.conv1(x)
        x_1 = F.relu(x_1)
        x_1 = self.bn1(x_1)
        x_1 = self.conv2(x_1)
        x_1 = F.relu(x_1)
        x_1 = self.bn2(x_1)
        x_1 = x_1.view(self.num_nodes, -1)
        x_1 = self.fc(x_1)
        x_1 = F.relu(x_1)
        x_1 = self.bn3(x_1)

        receivers = torch.matmul(self.rel_rec, x_1)
        senders = torch.matmul(self.rel_send, x_1)
        x_1 = torch.cat([senders, receivers], dim=1)
        x_1 = torch.relu(self.fc_out(x_1))
        x_1 = self.fc_cat(x_1)

        adj_1 = gumbel_softmax(x_1, temperature=temp, hard=False)
        adj_1 = adj_1[:, 0].clone().reshape(self.num_nodes, -1)  # 指定行数，列未知
        mask = torch.eye(self.num_nodes, self.num_nodes).bool().to(device)  # 定义行列均为self.num_nodes的标准型矩阵并转换成tensor
        adj_1 = adj_1.masked_fill_(mask, 0)  # 将reshape后的adj的对角线值全部替换成0

        x_2 = self.conv3(x)
        x_2 = F.relu(x_2)
        x_2 = self.bn1(x_2)
        x_2 = self.conv4(x_2)
        x_2 = F.relu(x_2)
        x_2 = self.bn2(x_2)
        x_2 = x_2.view(self.num_nodes, -1)
        x_2 = self.fc(x_2)
        x_2 = F.relu(x_2)
        x_2 = self.bn3(x_2)

        receivers = torch.matmul(self.rel_rec, x_2)
        senders = torch.matmul(self.rel_send, x_2)
        x_2 = torch.cat([senders, receivers], dim=1)
        x_2 = torch.relu(self.fc_out(x_2))
        x_2 = self.fc_cat(x_2)

        adj_2 = gumbel_softmax(x_2, temperature=temp, hard=False)
        adj_2 = adj_2[:, 0].clone().reshape(self.num_nodes, -1)  # 指定行数，列未知
        mask = torch.eye(self.num_nodes, self.num_nodes).bool().to(device)  # 定义行列均为self.num_nodes的标准型矩阵并转换成tensor
        adj_2 = adj_2.masked_fill_(mask, 0)  # 将reshape后的adj的对角线值全部替换成0

        adj_f = (adj_1 + adj_2) / 2

        # add adaptive graph
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)

        adj_new = adj_f + adp

        return adj_new

    def encoder(self, inputs, adj):
        """
        Encoder forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(inputs[t], adj, encoder_hidden_state)

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, adj, labels=None, batches_seen=None):
        """
        Decoder forward pass
        :param encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.decoder_model.output_dim),
                                device=device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input, adj,
                                                                      decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, inputs, adj, labels=None, batches_seen=None):
        # inputs为当前batch训练集X值，node_feas为训练集的特征矩阵，labels为当前batch训练集Y值
        """
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :param labels: shape (horizon, batch_size, num_sensor * output)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """

        encoder_hidden_state = self.encoder(inputs, adj)
        self._logger.debug("Encoder complete, starting decoder")
        outputs = self.decoder(encoder_hidden_state, adj, labels, batches_seen=batches_seen)
        self._logger.debug("Decoder complete")

        if batches_seen == 0:
            self._logger.info(
                "Total trainable parameters {}".format(count_parameters(self))
            )

        return outputs