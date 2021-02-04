# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:14:01 2019

@author: Kecheng Chen
This code, i.e.,3D GCN module, is for the paper "3D Graph Convolutional Networks for Slice-based
Medical Image denoising."
Main idea:
You can impose this module to extract the non-local information of  medical image intra-slice
with a play-and-plug fashion. Also, this can leverage the 3D spatial information between slices.
Finally, those information are aggregated as a hybrid one through a trainable weight.
If you have any questions, please feel free to conduct me (cs.ckc96@gmail.com).
"""
import tensorflow as tf


class GCN_3D:
    def __init__(self, config):
        self.config = config
        self.N = config.N
        self.min_nn = config.min_nn
        self.min_depth_nn = config.min_depth_nn
        self.depth = config.depth
        self.input_channel = config.input_channel
        self.output_channel = config.output_channel
        self.dn_vars = []

        name_block = "intra_slice_gcn"
        self.create_gconv_variables(name_block, 1, self.input_channel, self.input_channel, self.output_channel,
                                    config.rank_theta, config.stride_pregconv, config.stride_pregconv)

        name_block = "inter_slice_gcn"
        self.create_gconv_variables(name_block, 1, self.input_channel, self.input_channel, self.output_channel,
                                    config.rank_theta, config.stride_pregconv, config.stride_pregconv)
        name = 'scaling_beta'
        self.W[name] = tf.Variable(0, trainable=True, dtype=tf.float32, name=name)
        self.dn_vars = self.dn_vars + [self.W[name]]

        name = 'scaling_alpha'
        self.W[name] = tf.Variable(1, trainable=True, dtype=tf.float32, name=name)
        self.dn_vars = self.dn_vars + [self.W[name]]

    def create_gconv_variables(self, name_block, i, in_feat, fnet_feat, out_feat, rank_theta, stride_th1, stride_th2):

        name = name_block + "_nl_" + str(i) + "_flayer0"
        self.W[name] = tf.get_variable(name, [in_feat, fnet_feat], dtype=tf.float32,
                                       initializer=tf.glorot_normal_initializer())
        self.b[name] = tf.get_variable("b_" + name, [1, fnet_feat], dtype=tf.float32,
                                       initializer=tf.zeros_initializer())
        self.dn_vars = self.dn_vars + [self.W[name], self.b[name]]
        name = name_block + "_nl_" + str(i) + "_flayer1"
        self.W[name + "_th1"] = tf.get_variable(name + "_th1", [fnet_feat, stride_th1 * rank_theta], dtype=tf.float32,
                                                initializer=tf.random_normal_initializer(0, 1.0 / (
                                                        np.sqrt(fnet_feat + 0.0) * np.sqrt(in_feat + 0.0))))
        self.b[name + "_th1"] = tf.get_variable(name + "_b_th1", [1, rank_theta, in_feat], dtype=tf.float32,
                                                initializer=tf.zeros_initializer())
        self.W[name + "_th2"] = tf.get_variable(name + "_th2", [fnet_feat, stride_th2 * rank_theta], dtype=tf.float32,
                                                initializer=tf.random_normal_initializer(0, 1.0 / (
                                                        np.sqrt(fnet_feat + 0.0) * np.sqrt(in_feat + 0.0))))
        self.b[name + "_th2"] = tf.get_variable(name + "_b_th2", [1, rank_theta, out_feat], dtype=tf.float32,
                                                initializer=tf.zeros_initializer())
        self.W[name + "_thl"] = tf.get_variable(name + "_thl", [fnet_feat, rank_theta], dtype=tf.float32,
                                                initializer=tf.random_normal_initializer(0, 1.0 / np.sqrt(
                                                    rank_theta + 0.0)))
        self.b[name + "_thl"] = tf.get_variable(name + "_b_thl", [1, rank_theta], dtype=tf.float32,
                                                initializer=tf.zeros_initializer())
        self.dn_vars = self.dn_vars + [self.W[name + "_th1"], self.b[name + "_th1"], self.W[name + "_th2"],
                                       self.b[name + "_th2"], self.W[name + "_thl"], self.b[name + "_thl"]]
        name = name_block + "_l_" + str(i)
        self.W[name] = tf.get_variable(name, [3, 3, in_feat, out_feat], dtype=tf.float32,
                                       initializer=tf.glorot_normal_initializer())
        self.dn_vars = self.dn_vars + [self.W[name]]
        name = name_block + "_" + str(i)
        self.b[name] = tf.get_variable(name, [1, out_feat], dtype=tf.float32, initializer=tf.zeros_initializer())
        self.dn_vars = self.dn_vars + [self.b[name]]

    # obtain intra-slice graph
    def compute_graph_intra(self, h):
        '''
                  Input : [batch, n_height * in_width, in_channels]
        '''
        id_mat = 2 * tf.eye(self.N)  # The diagonal values is 2

        h = tf.cast(h, tf.float64)  #

        sq_norms = tf.reduce_sum(h * h, 2)  # (B,N)
        D = tf.abs(
            tf.expand_dims(sq_norms, 2) + tf.expand_dims(sq_norms, 1) - 2 * tf.matmul(h, h,
                                                                                      transpose_b=True))  # (B, N, N)
        D = tf.cast(D, tf.float32)
        D = tf.multiply(D, self.local_mask)
        D = D - id_mat

        h = tf.cast(h, tf.float32)

        return D

    # obtain inter-slice graph
    def compute_graph_inter(self, h):
        '''
          Input : [batch, in_depth, in_height, in_width, in_channels]
        '''

        # sq_norms [B,D,N]
        sq_norms = tf.reduce_sum(h * h, 3)
        D = []
        # for i in Depth:
        for i in range(self.depth):
            if i == int(self.depth / 2):
                D1 = tf.reduce_sum(
                    2 * tf.multiply(h[:, int(self.depth / 2), :, :], 0), 2)
                D1 = D1 - 2  # set a signature for central nodes
                D.append(D1)
            else:
                D1 = tf.add(sq_norms[:, int(self.depth / 2), :], sq_norms[:, i, :]) - tf.reduce_sum(
                    2 * tf.multiply(h[:, int(self.depth / 2), :, :], h[:, i, :, :]), 2)  # [B,N]
                D.append(D1)
        D = tf.stack(D, axis=2)
        return D

    # perform intra-slices GCN
    def intra_gconv(self, h, name, in_feat, out_feat, stride_th1, stride_th2, compute_graph=True, return_graph=False,
                    D=[]):
        if compute_graph:
            D = self.compute_graph_intra(h)

        _, top_idx = tf.nn.top_k(-D, self.config.min_nn + 1)  # (B, N, d+1)
        # self index number is equal to the non-local
        top_idx2 = tf.reshape(tf.tile(tf.expand_dims(top_idx[:, :, 0], 2), [1, 1, self.config.min_nn - 8]),
                              [-1, self.N * (self.config.min_nn - 8)])  # (B, N*d)
        # the non-local index
        top_idx = tf.reshape(top_idx[:, :, 9:], [-1, self.N * (self.config.min_nn - 8)])  # (B, N*d)
        # the index of non-local neighbors for every pixel
        x_tilde1 = tf.batch_gather(h, top_idx)  # (B, K, dlm1)	# h=[B,N,C] top_idx = (B, N*d)
        x_tilde2 = tf.batch_gather(h, top_idx2)  # (B, K, dlm1)
        # The vector of every non-local vertices
        labels = x_tilde1 - x_tilde2  # (B, K, dlm1) # (B,N*d=K,dlm1)

        # to be consistent the shape of  vertices and labels
        x_tilde1 = tf.reshape(x_tilde1, [-1, in_feat])  # (B*K, dlm1)
        labels = tf.reshape(labels, [-1, in_feat])  # (B*K, dlm1)

        # sum operation along with channel direction
        e_ij = tf.reshape(tf.reduce_sum(labels * labels, 1), [-1, self.config.min_nn - 8]) / h.shape[2]  # (B*N, d)

        name_flayer = name + "_flayer0"
        labels = tf.nn.leaky_relu(tf.matmul(labels, self.W[name_flayer]) + self.b[name_flayer])  # (B*K, F)
        name_flayer = name + "_flayer1"
        labels_exp = tf.expand_dims(labels, 1)  # (B*K, 1, F)
        labels1 = labels_exp + 0.0
        for ss in range(1, in_feat / stride_th1):
            labels1 = tf.concat([labels1, self.myroll(labels_exp, shift=(ss + 1) * stride_th1, axis=2)],
                                axis=1)  # (B*K, dlm1/stride, dlm1)
        labels2 = labels_exp + 0.0
        for ss in range(1, out_feat / stride_th2):
            labels2 = tf.concat([labels2, self.myroll(labels_exp, shift=(ss + 1) * stride_th2, axis=2)],
                                axis=1)  # (B*K, dl/stride, dlm1)
        theta1 = tf.matmul(tf.reshape(labels1, [-1, in_feat]),
                           self.W[name_flayer + "_th1"])  # (B*K*dlm1/stride, R*stride)
        theta1 = tf.reshape(theta1, [-1, self.config.rank_theta, in_feat]) + self.b[name_flayer + "_th1"]
        theta2 = tf.matmul(tf.reshape(labels2, [-1, in_feat]),
                           self.W[name_flayer + "_th2"])  # (B*K*dl/stride, R*stride)
        theta2 = tf.reshape(theta2, [-1, self.config.rank_theta, out_feat]) + self.b[name_flayer + "_th2"]
        thetal = tf.expand_dims(tf.matmul(labels, self.W[name_flayer + "_thl"]) + self.b[name_flayer + "_thl"],
                                2)  # (B*K, R, 1)
        # theta1=>    tf.expand_dims(x_tilde1, 2)=> (B*K(N*d), dlm1,1)
        x = tf.matmul(theta1, tf.expand_dims(x_tilde1, 2))  # (B*K, R, 1)
        x = tf.multiply(x, thetal)  # (B*K, R, 1)
        x = tf.matmul(theta2, x, transpose_a=True)[:, :, 0]  # (B*K, dl)

        # obtain multiple non-local information +
        x = tf.reshape(x, [-1, self.config.min_nn - 8, out_feat])  # (N, d, dl)
        # here is the change point, use a softmax probability to weight every term.
        # x = tf.multiply(x, tf.expand_dims(tf.exp(-tf.div(d_labels, 10)), 2))  # (N, d, dl)
        # use a softmax probability to weight every term.

        # here need to verify by the numpy
        e_tilde = tf.reshape(e_ij, [-1, self.N, (self.config.min_nn - 8)])  # [B,N,d]
        exp_e_tilde = tf.exp(e_tilde)  # [B,N,d]
        sum_e_tilde = tf.reduce_sum(exp_e_tilde, 2)  # [B,N]
        sum_e_tilde = tf.tile(tf.expand_dims(sum_e_tilde, 2), [1, 1, self.config.min_nn - 8])  # [B,N,d]
        # softmax probability
        a = tf.expand_dims(tf.reshape(tf.div(-exp_e_tilde, -sum_e_tilde), [-1, self.config.min_nn - 8]), 2)
        x = tf.multiply(x, a)

        x = tf.reduce_mean(x, 1)  # (N, dl)
        x = tf.reshape(x, [-1, self.N, out_feat])  # (B, N, dl)

        if return_graph:
            return x, D
        else:
            return x

    # perform inter-slices gconv
    def inter_gconv(self, h, name, in_feat, out_feat, stride_th1, stride_th2, compute_graph=True, return_graph=False,
                    D=[]):
        if compute_graph:
            D = self.compute_graph(h)

        _, top_idx = tf.nn.top_k(-D, self.min_depth_nn + 1)  # (B, N, d+1)
        # self index number is equal to the non-local
        top_idx2 = tf.reshape(tf.tile(tf.expand_dims(top_idx[:, :, 0], 2), [1, 1, self.min_depth_nn]),
                              [-1, self.N * (self.min_depth_nn)])  # (B, N*d)
        # the non-local index
        top_idx = tf.reshape(top_idx[:, :, 1:], [-1, self.N * (self.min_depth_nn)])  # (B, N*d)

        # perform the tansormation of dimension
        h = tf.transpose(h, [0, 2, 1, 3])
        top_idx = tf.reshape(top_idx, [-1, N, self.min_depth_nn])
        top_idx2 = tf.reshape(top_idx2, [-1, N, self.min_depth_nn])

        # the index of non-local neighbors for every pixel
        x_tilde1 = tf.reshape(tf.batch_gather(h, top_idx),
                              [-1, N * self.min_depth_nn, self.input_channel])  # (B, K, dlm1)
        x_tilde2 = tf.reshape(tf.batch_gather(h, top_idx2), [-1, N * self.min_depth_nn, self.input_channel])
        # The vector of every non-local vertices
        labels = x_tilde1 - x_tilde2  # (B, K, dlm1) # (B,N*d=K,dlm1)

        # to be consistent the shape of  vertices and labels
        x_tilde1 = tf.reshape(x_tilde1, [-1, self.input_channel])  # (B*K, dlm1)
        labels = tf.reshape(labels, [-1, self.input_channel])  # (B*K, dlm1)

        # sum operation along with channel direction
        e_ij = tf.reshape(tf.reduce_sum(labels * labels, 1), [-1, self.min_depth_nn]) / self.input_channel  # (B*N, d)

        name_flayer = name + "_flayer0"
        labels = tf.nn.leaky_relu(tf.matmul(labels, self.W[name_flayer]) + self.b[name_flayer])  # (B*K, F)
        name_flayer = name + "_flayer1"
        labels_exp = tf.expand_dims(labels, 1)  # (B*K, 1, F)
        labels1 = labels_exp + 0.0
        for ss in range(1, in_feat / stride_th1):
            labels1 = tf.concat([labels1, self.myroll(labels_exp, shift=(ss + 1) * stride_th1, axis=2)],
                                axis=1)  # (B*K, dlm1/stride, dlm1)
        labels2 = labels_exp + 0.0
        for ss in range(1, out_feat / stride_th2):
            labels2 = tf.concat([labels2, self.myroll(labels_exp, shift=(ss + 1) * stride_th2, axis=2)],
                                axis=1)  # (B*K, dl/stride, dlm1)
        theta1 = tf.matmul(tf.reshape(labels1, [-1, in_feat]),
                           self.W[name_flayer + "_th1"])  # (B*K*dlm1/stride, R*stride)
        theta1 = tf.reshape(theta1, [-1, self.config.rank_theta, in_feat]) + self.b[name_flayer + "_th1"]
        theta2 = tf.matmul(tf.reshape(labels2, [-1, in_feat]),
                           self.W[name_flayer + "_th2"])  # (B*K*dl/stride, R*stride)
        theta2 = tf.reshape(theta2, [-1, self.config.rank_theta, out_feat]) + self.b[name_flayer + "_th2"]
        thetal = tf.expand_dims(tf.matmul(labels, self.W[name_flayer + "_thl"]) + self.b[name_flayer + "_thl"],
                                2)  # (B*K, R, 1)
        # theta1=>    tf.expand_dims(x_tilde1, 2)=> (B*K(N*d), dlm1,1)
        x = tf.matmul(theta1, tf.expand_dims(x_tilde1, 2))  # (B*K, R, 1)
        x = tf.multiply(x, thetal)  # (B*K, R, 1)
        x = tf.matmul(theta2, x, transpose_a=True)[:, :, 0]  # (B*K, dl)

        # obtain multiple non-local information +
        x = tf.reshape(x, [-1, self.config.min_nn - 8, out_feat])  # (N, d, dl)
        # here is the change point, use a softmax probability to weight every term.
        # x = tf.multiply(x, tf.expand_dims(tf.exp(-tf.div(d_labels, 10)), 2))  # (N, d, dl)
        # use a softmax probability to weight every term.

        # here need to verify by the numpy
        e_tilde = tf.reshape(e_ij, [-1, self.N, self.min_depth_nn])  # [B,N,d]
        exp_e_tilde = tf.exp(e_tilde)  # [B,N,d]
        sum_e_tilde = tf.reduce_sum(exp_e_tilde, 2)  # [B,N]
        sum_e_tilde = tf.tile(tf.expand_dims(sum_e_tilde, 2), [1, 1, self.config.min_nn - 8])  # [B,N,d]
        a = tf.expand_dims(tf.reshape(tf.div(-exp_e_tilde, -sum_e_tilde), [-1, self.config.min_nn - 8]), 2)
        x = tf.multiply(x, a)

        x = tf.reduce_mean(x, 1)  # (N, dl)
        x = tf.reshape(x, [-1, self.N, out_feat])  # (B, N, dl)

        if return_graph:
            return x, D
        else:
            return x

    # same as new tf.roll but only for 3D input and axis=2
    def myroll(self, h, shift=0, axis=2):

        h_len = h.get_shape()[2]
        return tf.concat([h[:, :, h_len - shift:], h[:, :, :h_len - shift]], axis=2)

    # use the general 3D conv to obtain the local information
    def conv_3d(self, h, padding='SAME', out_channel=0, filter_size=3):
        local = tf.layers.conv3d(h, out_channel, filter_size, padding=padding,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(), name='GCN_conv1',
                                 use_bias=False)
        # obtain the central slice, default the stride of 3D conv: [1,1,1]
        x = local[:, int(((self.depth - filter_size) / 1 + 1) / 2)]
        x = tf.reshape(x, [-1, self.N, out_feat])  # (B, N, dl)
        return x

    def general_lnl_aggregation(self, h, intra, inter, local):
        name = 'scaling_beta'
        output = tf.multiple((1 - self.W[name]), local) + \
                 tf.multiply(self.W[name], tf.add(intra, inter))
        local[:, int(self.depth/2)] = output
        output = h + local
        return output

    def general_lnl_aggregation2(self, h, intra, inter, local):
        name = 'scaling_beta'
        name_2 = 'scaling_alpha'
        output = tf.multiple(self.W[name_2], local) + tf.multiply(self.W[name], inter) + tf.multiply(1 - self.W[name],
                                                                                                     intra)
        local[:, int(self.depth / 2)] = output
        output = h + local
        return output

    def get_PAP_map(self, input, local_mask):

        intra = self.intra_gconv(input[:,int(self.depth/2)], 'intra_slice_gcn', self.input_channel, self.output_channel,
                            self.config.stride_th1, self.config.stride_th2, compute_graph=True, return_graph=False,D=[])
        inter = self.intra_gconv(input, 'inter_slice_gcn', self.input_channel,
                            self.output_channel,
                            self.config.stride_th1, self.config.stride_th2, compute_graph=True, return_graph=False,
                            D=[])
        local = self.conv_3d(input,padding='SAME',out_channel=self.output_channel,filter_size=3)

        aggregate = self.PAP_lnl_aggregation(input,intra,inter,local)
        return aggregate

