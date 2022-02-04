# library
# standard library
import os

# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import math

import torch.nn.functional as F

# ============ SCInet ===============

class SciNet(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim, layer_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.layer_dim = layer_dim
        """Initialize SciNet Model.
        Params
        ======
            input_dim (int): number of inputs
            output_dim (int): number of outputs
            latent_dim (int): number of latent neurons
            Layer_dim (int): number of neurons in hidden layers
        """
        super(SciNet, self).__init__()
        self.latent_dim = latent_dim
        self.enc1 = nn.Linear(input_dim, layer_dim)
        self.enc2 = nn.Linear(layer_dim, layer_dim)
        self.latent = nn.Linear(layer_dim, latent_dim * 2)
        self.dec1 = nn.Linear(latent_dim + 1, layer_dim)
        self.dec2 = nn.Linear(layer_dim, layer_dim)
        self.out = nn.Linear(layer_dim, output_dim)

    def feature(self):
        return {"self.input_dim": self.input_dim, "self.output_dim": self.output_dim,
                "self.latent_dim": self.latent_dim, "self.layer_dim": self.layer_dim}

    def encoder(self, x):
        z = F.elu(self.enc1(x))
        z = F.elu(self.enc2(z))
        z = self.latent(z)
        self.mu = z[:, 0:self.latent_dim]
        self.log_sigma = z[:, self.latent_dim:]
        self.sigma = torch.exp(self.log_sigma)

        # Use reparametrization trick to sample from gaussian
        eps = torch.randn(x.size(0), self.latent_dim)
        z_sample = self.mu + self.sigma * eps

        # Compute KL loss
        self.kl_loss = kl_divergence(self.mu, self.log_sigma, dim=self.latent_dim)

        return z_sample

    def decoder(self, z):
        x = F.elu(self.dec1(z))
        x = F.elu(self.dec2(x))
        return self.out(x)

    def forward(self, obs):
        q = obs[:, -1].reshape(obs.size(0), 1)
        obs = obs[:, 0:-1]
        self.latent_r = self.encoder(obs)
        dec_input = torch.cat((q, self.latent_r), 1)

        return self.decoder(dec_input)


def kl_divergence(means, log_sigma, dim, target_sigma=0.1):
    """
    Computes Kullback–Leibler divergence for arrays of mean and log(sigma)
    """
    target_sigma = torch.Tensor([target_sigma])
    return 1 / 2. * torch.mean(torch.mean(1 / target_sigma ** 2 * means ** 2 +
                                          torch.exp(2 * log_sigma) / target_sigma ** 2 - 2 * log_sigma + 2 * torch.log(
        target_sigma), dim=1) - dim)


# =========== cnn1d ============
class CNN1D(nn.Module):
    def feature(self):
        return{"channel_num_list":self.channel_num_list,"kernel_size":self.kernel_size,"stride":self.stride,
               "padding":self.padding,"Pool_kernel_size":self.Pool_kernel_size,"dropout":self.dropout,"activate_function":self.activate_function,"featureNum":self.featureNum }

    def setLayer(self, in_channel_num=1, out_channel_num=16, kernel_size=3, stride=1, padding=1, Pool_kernel_size=1
                 ,dropout=0,activate_function_layer = nn.ReLU(True)):
#         print("padding",type(padding),"stride",type(stride),"kernel_size",type(kernel_size))
        Layer = []
        Layer.append(nn.Conv1d(
            in_channels=in_channel_num,  # input height
            out_channels=out_channel_num,  # n_filters
            kernel_size=kernel_size,  # filter size
            stride=stride,  # filter movement/step
            padding=padding,  #
        ))

        Layer.append(nn.BatchNorm1d(out_channel_num))

        Layer.append(activate_function_layer)
        if (dropout > 0):
            Layer.append(nn.Dropout(dropout))
        if (Pool_kernel_size > 1):
            Layer.append(nn.MaxPool1d(kernel_size=Pool_kernel_size))
        return Layer

    def __init__(self,channel_num_list=[16,32,64], kernel_size=3, stride=1, padding=1, Pool_kernel_size=1,
                 dropout=0,featureNum = 13,activate_function = "relu"):

        self.channel_num_list,self.kernel_size,self.stride,self.padding,self.Pool_kernel_size,self.dropout,self.activate_function,self.featureNum = channel_num_list,kernel_size,stride,padding,Pool_kernel_size,dropout,activate_function,featureNum

        # 判断 激活函数
        if activate_function == 'relu':
            activate_function_layer = nn.ReLU(True)
        if activate_function == 'sigmoid':
            activate_function_layer = nn.Sigmoid()
        if activate_function == 'tanh':
            activate_function_layer = nn.Tanh()
        if activate_function == 'LeakyReLU':
            activate_function_layer = nn.LeakyReLU(True)




        super(CNN1D, self).__init__()
        ConvList= self.setLayer(in_channel_num=1, out_channel_num=channel_num_list[0], kernel_size=kernel_size,
                               stride=stride, padding=padding, Pool_kernel_size=Pool_kernel_size, dropout=dropout, activate_function_layer = activate_function_layer)

        outNum = ( ( self.featureNum - kernel_size + 2 * padding ) // stride + 1  )// Pool_kernel_size

        for num in range(1,len(channel_num_list)):
            ConvList.extend(self.setLayer(in_channel_num=channel_num_list[num - 1], out_channel_num=channel_num_list[num], kernel_size=kernel_size,
                                          stride=stride, padding=padding, Pool_kernel_size=Pool_kernel_size,dropout=dropout , activate_function_layer = activate_function_layer))

            outNum = ((outNum - kernel_size + 2 * padding) // stride + 1) // Pool_kernel_size

        self.Conv =  nn.Sequential(*ConvList)
        self.Linear = nn.Sequential(
            nn.Linear(int(outNum) * channel_num_list[-1], 1024),
            activate_function_layer,
            nn.Dropout(self.dropout),
            nn.Linear(1024, 1)
        )
        print("======== self.ConvList =================")
        print(self.Conv)
        # self.Conv1 = nn.Sequential(
        #     *self.setLayer(in_channel_num=1, out_channel_num=channel_num_list[0], kernel_size=kernel_size, stride=stride, padding=padding, Pool_kernel_size=Pool_kernel_size,dropout=dropout))
        # self.Conv2 = nn.Sequential(
        #     *self.setLayer(in_channel_num=channel_num_list[0], out_channel_num=channel_num_list[1], kernel_size=kernel_size, stride=stride, padding=padding,Pool_kernel_size=Pool_kernel_size, dropout=dropout))
        # self.Conv3 = nn.Sequential(
        #     *self.setLayer(in_channel_num=channel_num_list[1], out_channel_num=channel_num_list[2], kernel_size=kernel_size, stride=stride, padding=padding,Pool_kernel_size=Pool_kernel_size, dropout=dropout))

        self.residual_1 = nn.Sequential(
            *self.setLayer(in_channel_num=1, out_channel_num=16, kernel_size=1, stride=1, padding=0, Pool_kernel_size=1,dropout=0) )
        self.residual_2 = nn.Sequential(
            *self.setLayer(in_channel_num=16, out_channel_num=32, kernel_size=1, stride=1, padding=0,Pool_kernel_size=1,dropout=0))

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1])
        x = self.Conv(x)


        # residual = x
        # x = self.Conv1(x)
        # # residual= self.residual_1(residual)
        # # x =x + residual
        # x = self.Conv2(x)
        # # residual= self.residual_2(residual)
        # # x =x + residual
        # x = self.Conv3(x)
        x = x.view(x.shape[0], -        1)

        out = self.Linear(x)
        return out


# ===========   transformer ==========
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        #         print("PositionalEncoding",x.size())
        return x + self.pe[:x.size(0), :]
'''
d_model – 输入中预期特征的数量（必需）。

nhead – 多头注意模型中的头数（必需）。

dim_feedforward – 前馈网络模型的维度（默认值 = 2048）。

dropout – dropout 值（默认值 = 0.1）。

激活- 中间层、relu 或 gelu 的激活函数（默认值=relu）。

layer_norm_eps – 层归一化组件中的 eps 值（默认值 = 1e-5）。

batch_first – 如果True，则输入和输出张量提供为 (batch, seq, feature)。默认值：False。
'''


class TransAm(nn.Module):
    def __init__(self,  feature_size=250, num_layers=1, dropout=0.1, stackNum=2, dim_feedforward_num=2048,nhead=2,
                 activate_function='LeakyReLU', hidden_layers=[1024, 512]):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size,
                    nhead=nhead, dropout=dropout,dim_feedforward=dim_feedforward_num)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.feature_size = feature_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.activate_function = activate_function
        self.hidden_layers = hidden_layers
        self.dim_feedforward_num = dim_feedforward_num
        self.nhead = nhead

        # 判断 激活函数
        if self.activate_function == 'relu':
            activate_function = nn.ReLU(True)
        if self.activate_function == 'sigmoid':
            activate_function = nn.Sigmoid()
        if self.activate_function == 'tanh':
            activate_function = nn.Tanh()
        if self.activate_function == 'LeakyReLU':
            activate_function = nn.LeakyReLU(True)
        # 搭建 cnn1d 网络

        layers = []
        layers.append(nn.Linear(stackNum * feature_size, hidden_layers[0]))
        layers.append(activate_function)
        layers.append(nn.Dropout(self.dropout))

        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            layers.append(activate_function)
            layers.append(nn.Dropout(self.dropout))

        # output layer
        output_layer = nn.Linear(hidden_layers[-1], 1)
        layers.append(output_layer)

        self.decoder = nn.Sequential(  # self.Conv =  nn.Sequential(*ConvList)
            *layers
        )

        # self.init_weights()
        self.initialize_weights()

    def feature(self):
        return {"feature_size": self.feature_size,"nhead":self.nhead ,"num_layers": self.num_layers, "dropout": self.dropout,
        "dim_feedforward_num":self.dim_feedforward_num,"activate_function": self.activate_function, "hidden_layers":self.hidden_layers}

    # 定义权值初始化
    def initialize_weights(self):
        for m in self.decoder:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                # m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()

    # ————————————————
    # 版权声明：本文为CSDN博主「chris_1996」的原创文章，遵循CC
    # 4.0
    # BY - SA版权协议，转载请附上原文出处链接及本声明。
    # 原文链接：https: // blog.csdn.net / weixin_45833008 / article / details / 108643051

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        #         print("0",src.shape)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        #         print("1",src.shape)
        src = self.pos_encoder(src)
        #         print("2",src.shape)
        output = self.transformer_encoder(src, self.src_mask)  # , self.src_mask)
        output = output.view(output.shape[0], -1)
        # print("3",output.shape)
        output = self.decoder(output)
        # print("4",output.shape)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


# ===========   transformer + CNN_1D ==========
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        #         print("PositionalEncoding",x.size())

        return x + self.pe[:x.size(0), :]


class TransAm_CNN_1D(nn.Module):
    def __init__(self, feature_size=250, num_layers=1, dropout=0.1,stackNum = 2,activate_function = 'LeakyReLU',channel_layers = [32,18],
                kernel_size = 3,stride = 1,padding=1,Pool_kernel_size=1,dim_feedforward_num=2048,nhead=2):
        super(TransAm_CNN_1D, self).__init__()
        self.model_type = 'Transformer'
        
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead, dropout=dropout,dim_feedforward = dim_feedforward_num)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        self.feature_size = feature_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.activate_function = activate_function
        self.channel_layers = channel_layers
        self.dim_feedforward_num = dim_feedforward_num
        self.nhead = nhead

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.Pool_kernel_size = Pool_kernel_size
        
        # 判断 激活函数
        if self.activate_function == 'relu':
            activate_function = nn.ReLU(True)
        if self.activate_function == 'sigmoid':
            activate_function = nn.Sigmoid()
        if self.activate_function == 'tanh':
            activate_function = nn.Tanh()
        if self.activate_function == 'LeakyReLU':
            activate_function = nn.LeakyReLU(True)

        # 第一层 conv网络
        layers = []
        layers.extend( self.setLayer(in_channel_num=stackNum, out_channel_num=channel_layers[0], kernel_size=kernel_size, stride=stride, padding=padding, Pool_kernel_size=Pool_kernel_size,dropout=dropout,activate_function_layer = activate_function ) )
        
        outNum = ( ( self.feature_size - kernel_size + 2 * padding ) // stride + 1  )// Pool_kernel_size
        
        for i in range(1,len(channel_layers)):
            layers.extend( self.setLayer(in_channel_num=channel_layers[i-1], out_channel_num=channel_layers[i], kernel_size=kernel_size, stride=stride, padding=padding, Pool_kernel_size=Pool_kernel_size,dropout=dropout,activate_function_layer = activate_function ) )
            outNum = ((outNum - kernel_size + 2 * padding) // stride + 1) // Pool_kernel_size
        
        
        
        # output layer
#         output_layer = nn.Linear(hidden_layers[-1], 1)
#         layers.append(output_layer)
        
        
        
        self.decoder = nn.Sequential(
            *layers
        )
        
        self.Linear = nn.Sequential(
            nn.Linear(int(outNum) * channel_layers[-1], 1024),
            activate_function,
            nn.Dropout(self.dropout),
            nn.Linear(1024, 1)
        )


#        self.init_weights()
#        self.initialize_weights()

    # 设定需要的cnn1d 网络   Conv -> BatchNorm1d -> DropOut -> maxpool
    def setLayer(self, in_channel_num=1, out_channel_num=16, kernel_size=3, stride=1, padding=1, Pool_kernel_size=1
                 ,dropout=0,activate_function_layer = nn.ReLU(True)):
#         print("padding",type(padding),"stride",type(stride),"kernel_size",type(kernel_size))


        Layer = []
        Layer.append(nn.Conv1d(
            in_channels=in_channel_num,  # input height
            out_channels=out_channel_num,  # n_filters
            kernel_size=kernel_size,  # filter size
            stride=stride,  # filter movement/step
            padding=padding,  #
        ))
        
        Layer.append(nn.BatchNorm1d(out_channel_num))

        Layer.append(activate_function_layer)
        if (dropout > 0):
            Layer.append(nn.Dropout(dropout))
        if (Pool_kernel_size > 1):
            Layer.append(nn.MaxPool1d(kernel_size=Pool_kernel_size))
        return Layer   
        

    
    def feature(self):
        return {"feature_size": self.feature_size, "num_layers": self.num_layers, "dropout":self.dropout,"nhead" : self.nhead,
               "activate_function":self.activate_function,"channel_layers":self.channel_layers,"dim_feedforward_num":self.dim_feedforward_num,
               "kernel_size":self.kernel_size,"stride":self.stride,"padding":self.padding,"Pool_kernel_size":self.Pool_kernel_size}
    
    # 定义权值初始化
    def initialize_weights(self):
        for m in self.decoder:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                # m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()

    # ————————————————
    # 版权声明：本文为CSDN博主「chris_1996」的原创文章，遵循CC
    # 4.0
    # BY - SA版权协议，转载请附上原文出处链接及本声明。
    # 原文链接：https: // blog.csdn.net / weixin_45833008 / article / details / 108643051


    def forward(self, src):
        #         print("0",src.shape)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        #         print("1",src.shape)
        src = self.pos_encoder(src)
        #         print("2",src.shape)
        output = self.transformer_encoder(src, self.src_mask)  # , self.src_mask)
        
#         print("3",output.shape)
        output = self.decoder(output)
#         print("4",output.shape)
        output = output.view(output.shape[0], -1)
        output = self.Linear(output)
#         print("5",output.shape)

        
        
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask




# =========== transformaer + LSTM 



# ===========   transformer + CNN_1D + LSTM ==========

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        #         print("PositionalEncoding",x.size())
        return x + self.pe[:x.size(0), :]


class TransAm_CNN_LSTM(nn.Module):
    def __init__(self, feature_size=250, num_layers=1, dropout=0.1, stackNum=2, activate_function='LeakyReLU',nhead=2, dim_feedforward_num = 2048,
                 channel_layers=[32, 18],
                 kernel_size=3, stride=1, padding=1, Pool_kernel_size=1,
                 LSTM_hidden_size=32, LSTM_num_layers=2):
        super(TransAm_CNN_LSTM, self).__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size,nhead=nhead, dropout=dropout,dim_feedforward = dim_feedforward_num)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.nhead = nhead
        self.feature_size = feature_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.activate_function = activate_function
        self.channel_layers = channel_layers
        self.dim_feedforward_num = dim_feedforward_num

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.Pool_kernel_size = Pool_kernel_size

        self.LSTM_hidden_size = LSTM_hidden_size
        self.LSTM_num_layers = LSTM_num_layers

        # 判断 激活函数
        if self.activate_function == 'relu':
            activate_function = nn.ReLU(True)
        if self.activate_function == 'sigmoid':
            activate_function = nn.Sigmoid()
        if self.activate_function == 'tanh':
            activate_function = nn.Tanh()
        if self.activate_function == 'LeakyReLU':
            activate_function = nn.LeakyReLU(True)

        # 第一层 conv网络
        layers = []
        layers.extend(self.setLayer(in_channel_num=stackNum, out_channel_num=channel_layers[0], kernel_size=kernel_size,
                                    stride=stride, padding=padding, Pool_kernel_size=Pool_kernel_size, dropout=dropout,
                                    activate_function_layer=activate_function))
        outNum = ((self.feature_size - kernel_size + 2 * padding) // stride + 1) // Pool_kernel_size

        for i in range(1, len(channel_layers)):
            layers.extend(self.setLayer(in_channel_num=channel_layers[i - 1], out_channel_num=channel_layers[i],
                                        kernel_size=kernel_size, stride=stride, padding=padding,
                                        Pool_kernel_size=Pool_kernel_size, dropout=dropout,
                                        activate_function_layer=activate_function))
            outNum = ((outNum - kernel_size + 2 * padding) // stride + 1) // Pool_kernel_size

        self.decoder = nn.Sequential(
            *layers
        )

        # lstm 層
        self.lstm_network = nn.LSTM(feature_size, LSTM_hidden_size, LSTM_num_layers, batch_first=True)

        self.Linear = nn.Sequential(
            activate_function,
            nn.Dropout(dropout),
            nn.Linear(int(LSTM_hidden_size), 256),
            activate_function,
            nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )

        # self.init_weights()

    #         self.initialize_weights()

    # 设定需要的cnn1d 网络   Conv -> BatchNorm1d -> DropOut -> maxpool
    def setLayer(self, in_channel_num=1, out_channel_num=16, kernel_size=3, stride=1, padding=1, Pool_kernel_size=1
                 , dropout=0, activate_function_layer=nn.ReLU(True)):
        #         print("padding",type(padding),"stride",type(stride),"kernel_size",type(kernel_size))

        Layer = []
        Layer.append(nn.Conv1d(
            in_channels=in_channel_num,  # input height
            out_channels=out_channel_num,  # n_filters
            kernel_size=kernel_size,  # filter size
            stride=stride,  # filter movement/step
            padding=padding,  #
        ))

        Layer.append(nn.BatchNorm1d(out_channel_num))
        Layer.append(activate_function_layer)

        if (dropout > 0):
            Layer.append(nn.Dropout(dropout))
        if (Pool_kernel_size > 1):
            Layer.append(nn.MaxPool1d(kernel_size=Pool_kernel_size))
        return Layer

    def feature(self):
        return {"feature_size": self.feature_size, "num_layers": self.num_layers, "dropout": self.dropout,"nhead":self.nhead,
                "activate_function": self.activate_function, "channel_layers": self.channel_layers,
                "kernel_size": self.kernel_size, "stride": self.stride, "padding": self.padding,
                "Pool_kernel_size": self.Pool_kernel_size,"dim_feedforward_num":self.dim_feedforward_num,
                "LSTM_hidden_size": self.LSTM_hidden_size, "LSTM_hidden_size": self.LSTM_num_layers,
                }

    # 定义权值初始化
    def initialize_weights(self):
        for m in self.decoder:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                # m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()

    # ————————————————
    # 版权声明：本文为CSDN博主「chris_1996」的原创文章，遵循CC
    # 4.0
    # BY - SA版权协议，转载请附上原文出处链接及本声明。
    # 原文链接：https: // blog.csdn.net / weixin_45833008 / article / details / 108643051

    def forward(self, src):
        #         print("0",src.shape)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        #         print("1",src.shape)
        src = self.pos_encoder(src)
        #         print("2",src.shape)
        output = self.transformer_encoder(src, self.src_mask)  # , self.src_mask)
        output = self.decoder(output)  # CNN 1D

        self.lstm_network.flatten_parameters()
        output, (h, o) = self.lstm_network(output)
        # 将最后一个时间片扔进 Linear
        output = self.Linear(output[:, -1, :])

        #         print("5",output.shape)

        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


























'''





# ===========  CNN_1D + transformer  + LSTM ==========

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        #         print("PositionalEncoding",x.size())
        return x + self.pe[:x.size(0), :]


class CNN_TransAm_LSTM(nn.Module):
    def __init__(self, feature_size=250, num_layers=1, dropout=0.1, stackNum=2, activate_function='LeakyReLU',nhead=2, dim_feedforward_num = 2048,
                 channel_layers=[32, 18],
                 kernel_size=3, stride=1, padding=1, Pool_kernel_size=1,
                 LSTM_hidden_size=32, LSTM_num_layers=2):
        super(TransAm_CNN_LSTM, self).__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size,nhead=nhead, dropout=dropout,dim_feedforward = dim_feedforward_num)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.feature_size = feature_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.activate_function = activate_function
        self.channel_layers = channel_layers

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.Pool_kernel_size = Pool_kernel_size

        self.LSTM_hidden_size = LSTM_hidden_size
        self.LSTM_num_layers = LSTM_num_layers

        # 判断 激活函数
        if self.activate_function == 'relu':
            activate_function = nn.ReLU(True)
        if self.activate_function == 'sigmoid':
            activate_function = nn.Sigmoid()
        if self.activate_function == 'tanh':
            activate_function = nn.Tanh()
        if self.activate_function == 'LeakyReLU':
            activate_function = nn.LeakyReLU(True)

        # 第一层 conv网络
        layers = []
        layers.extend(self.setLayer(in_channel_num=stackNum, out_channel_num=channel_layers[0], kernel_size=kernel_size,
                                    stride=stride, padding=padding, Pool_kernel_size=Pool_kernel_size, dropout=dropout,
                                    activate_function_layer=activate_function))
        outNum = ((self.feature_size - kernel_size + 2 * padding) // stride + 1) // Pool_kernel_size

        for i in range(1, len(channel_layers)):
            layers.extend(self.setLayer(in_channel_num=channel_layers[i - 1], out_channel_num=channel_layers[i],
                                        kernel_size=kernel_size, stride=stride, padding=padding,
                                        Pool_kernel_size=Pool_kernel_size, dropout=dropout,
                                        activate_function_layer=activate_function))
            outNum = ((outNum - kernel_size + 2 * padding) // stride + 1) // Pool_kernel_size

        self.decoder = nn.Sequential(
            *layers
        )

        # lstm 層
        self.lstm_network = nn.LSTM(feature_size, LSTM_hidden_size, LSTM_num_layers, batch_first=True)

        self.Linear = nn.Sequential(
            activate_function,
            nn.Dropout(dropout),
            nn.Linear(int(LSTM_hidden_size), 256),
            activate_function,
            nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )

        # self.init_weights()

    #         self.initialize_weights()

    # 设定需要的cnn1d 网络   Conv -> BatchNorm1d -> DropOut -> maxpool
    def setLayer(self, in_channel_num=1, out_channel_num=16, kernel_size=3, stride=1, padding=1, Pool_kernel_size=1
                 , dropout=0, activate_function_layer=nn.ReLU(True)):
        #         print("padding",type(padding),"stride",type(stride),"kernel_size",type(kernel_size))

        Layer = []
        Layer.append(nn.Conv1d(
            in_channels=in_channel_num,  # input height
            out_channels=out_channel_num,  # n_filters
            kernel_size=kernel_size,  # filter size
            stride=stride,  # filter movement/step
            padding=padding,  #
        ))

        Layer.append(nn.BatchNorm1d(out_channel_num))

        Layer.append(activate_function_layer)

        if (dropout > 0):
            Layer.append(nn.Dropout(dropout))
        if (Pool_kernel_size > 1):
            Layer.append(nn.MaxPool1d(kernel_size=Pool_kernel_size))
        return Layer

    def feature(self):
        return {"feature_size": self.feature_size, "num_layers": self.num_layers, "dropout": self.dropout,
                "activate_function": self.activate_function, "channel_layers": self.channel_layers,
                "kernel_size": self.kernel_size, "stride": self.stride, "padding": self.padding,
                "Pool_kernel_size": self.Pool_kernel_size,
                "LSTM_hidden_size": self.LSTM_hidden_size, "LSTM_hidden_size": self.LSTM_num_layers,
                }

    # 定义权值初始化
    def initialize_weights(self):
        for m in self.decoder:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                # m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()

    # ————————————————
    # 版权声明：本文为CSDN博主「chris_1996」的原创文章，遵循CC
    # 4.0
    # BY - SA版权协议，转载请附上原文出处链接及本声明。
    # 原文链接：https: // blog.csdn.net / weixin_45833008 / article / details / 108643051

    def forward(self, src):
        #         print("0",src.shape)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        #         print("1",src.shape)
        src = self.pos_encoder(src)
        #         print("2",src.shape)
        output = self.transformer_encoder(src, self.src_mask)  # , self.src_mask)
        output = self.decoder(output)  # CNN 1D

        self.lstm_network.flatten_parameters()
        output, (h, o) = self.lstm_network(output)
        # 将最后一个时间片扔进 Linear
        output = self.Linear(output[:, -1, :])

        #         print("5",output.shape)

        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask



'''