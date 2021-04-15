import torch
import torch.nn as nn
import math


class LSTM_MultiModal(nn.Module):
    def __init__(self, input_size, hidden_size, second_input_size):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size

        W_f = torch.Tensor(input_size + hidden_size, hidden_size)
        b_f = torch.zeros(hidden_size)
        self.W_f = nn.Parameter(W_f) 
        self.b_f = nn.Parameter(b_f) 

        W_i = torch.Tensor(input_size + hidden_size, hidden_size)
        b_i = torch.zeros(hidden_size)
        self.W_i = nn.Parameter(W_i) 
        self.b_i = nn.Parameter(b_i) 

        W_c = torch.Tensor(input_size + hidden_size, hidden_size)
        b_c = torch.zeros(hidden_size)
        self.W_c = nn.Parameter(W_c) 
        self.b_c = nn.Parameter(b_c) 

        W_o = torch.Tensor(input_size + hidden_size, hidden_size)
        b_o = torch.zeros(hidden_size)
        self.W_o = nn.Parameter(W_o) 
        self.b_o = nn.Parameter(b_o)

        W_m = torch.Tensor(second_input_size, hidden_size)
        b_m = torch.zeros(hidden_size)
        self.W_m = nn.Parameter(W_m)
        self.b_m = nn.Parameter(b_m)

        W_cm = torch.Tensor(second_input_size + input_size, hidden_size)
        b_cm = torch.zeros(hidden_size)
        self.W_cm = nn.Parameter(W_cm)
        self.b_cm = nn.Parameter(b_cm)

        self.sigmoid = nn.Sigmoid()
        self.tanh    = nn.Tanh()

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.W_f, a=math.sqrt(5)) # weight init
        nn.init.kaiming_uniform_(self.W_c, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_o, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_i, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_m, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_cm, a=math.sqrt(5))

    def forward(self, x, x_fix):
        #w_times_x= torch.mm(x, self.weights.t())
        #return torch.add(w_times_x, self.bias)  # w times x + b
        
        batch_size = x.shape[0]

        x_fix = x_fix.sum(dim=1)

        h_tm1 = torch.zeros(batch_size, self.hidden_size).to(device='cuda')
        c_tm1 = torch.zeros(batch_size, self.hidden_size).to(device='cuda')

        hidden_states = []

        for i in range(x.shape[1]):
            x_t     = x[:,i,:]
            x_fix_t = torch.cat((x_t, x_fix), dim=1)
            x_t     = torch.cat((x_t, h_tm1), dim=1)

            # forget gate
            f_t = torch.mm(x_t, self.W_f)
            f_t = torch.add(f_t, self.b_f)
            f_t = self.sigmoid(f_t)

            i_t = torch.mm(x_t, self.W_i)
            i_t = torch.add(i_t, self.b_i)
            i_t = self.sigmoid(i_t)

            c_t = torch.mm(x_t, self.W_c)
            c_t = torch.add(c_t, self.b_c)
            c_t = self.tanh(c_t)

            o_t = torch.mm(x_t, self.W_o)
            o_t = torch.add(o_t, self.b_o)
            o_t = self.sigmoid(o_t)

            m_t = torch.mm(x_fix, self.W_m)
            m_t = torch.add(m_t, self.b_m)
            m_t = self.sigmoid(m_t)

            c_t = torch.mul(f_t, c_tm1) + torch.mul(i_t, c_t)
            c_t = self.tanh(c_t)
            c_tm1 = c_t

            x_fix_t = torch.mm(x_fix_t, self.W_cm)
            x_fix_t = torch.add(x_fix_t, self.b_cm)
            x_fix_t = self.sigmoid(x_fix_t)

            o_t = torch.mul(o_t, x_fix_t)

            h_t = torch.mul(o_t, c_t)
            h_tm1 = h_t
            hidden_states.append(h_t.unsqueeze(1))
        
        return torch.cat(hidden_states, dim=1), h_tm1

