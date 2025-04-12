#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
file  : LFISTA_cpss_s.py
author: Yaoteng Tan
email : ytan082@ucr.edu
date  : 2024-12-06

Implementation of Learned ISTA with support selection and coupled weights.
"""

import numpy as np
import tensorflow as tf
import utils.train

from utils.tf import shrink_ss
from models.LISTA_base import LISTA_base


class lrn_U_LFISTA_cpss_m (LISTA_base):

    """
    Implementation of deep neural network model.
    """

    def __init__(self, A, T, lam, percent, max_percent,
                 untied, coord, scope):
        """
        :prob:     : Instance of Problem class, describing problem settings.
        :T         : Number of layers (depth) of this LISTA model.
        :lam  : Initial value of thresholds of shrinkage functions.
        :untied    : Whether weights are shared within layers.
        """
        self._A    = A.astype (np.float32)
        # orthogonalized A
        A =  A.astype (np.float32)
        Q,R = np.linalg.qr(A.T)
        # self._A_orth   = (np.linalg.pinv(R.T)@A).astype (np.float32)
        self._A_mod = np.linalg.pinv(R)@np.linalg.pinv(R.T).astype (np.float32)
        
        self._T    = T
        self._p    = percent
        self._maxp = max_percent
        self._lam  = lam
        self._M    = self._A.shape [0]
        self._N    = self._A.shape [1]

        self._scale = 1.001 * np.linalg.norm (A, ord=2)**2
        self._theta = (self._lam / self._scale).astype(np.float32)
        if coord:
            self._theta = np.ones ((self._N, 1), dtype=np.float32) * self._theta

        self._ps = [(t+1) * self._p for t in range (self._T)]
        self._ps = np.clip (self._ps, 0.0, self._maxp)

        self._mu = (0.3 * np.ones_like(self._theta)).astype(np.float32)

        self._untied = untied
        self._coord  = coord
        self._scope  = scope

        """ Set up layers."""
        self.setup_layers()


    def setup_layers(self):
        """
        Implementation of LISTA model proposed by LeCun in 2010.

        :prob: Problem setting.
        :T: Number of layers in LISTA.
        :returns:
            :layers: List of tuples ( name, xh_, var_list )
                :name: description of layers.
                :xh: estimation of sparse code at current layer.
                :var_list: list of variables to be trained seperately.

        """
        # Ws_ = []
        # W1s_     = []
        Us_ = []
        alphas_ = []
        thetas_ = []
        mus_list_ = []

        # W = (np.transpose (self._A) / self._scale).astype (np.float32)
        # B = (np.transpose (self._A) / self._scale).astype (np.float32)
        # W1 = np.eye (self._N, dtype=np.float32) - np.matmul (B, self._A)

        with tf.variable_scope (self._scope, reuse=False) as vs:
            # constant
            self._kA_ = tf.constant (value=self._A, dtype=tf.float32)
            self._U_ = tf.constant (value=self._A_mod, dtype=tf.float32) # system mod matrix
            # use orthogonalized A
            # self._kA_ = tf.constant (value=self._A_orth, dtype=tf.float32) 

            if not self._untied: # tied model
                Us_.append (tf.get_variable (name='U',
                            dtype=tf.float32,
                            initializer=self._A_mod))
                Us_ = Us_ * self._T
            #     Ws_.append (tf.get_variable (name='W', dtype=tf.float32,
            #                                  initializer=W))
            #     Ws_ = Ws_ * self._T

                # W1s_.append (tf.get_variable (name='W1', dtype=tf.float32,
                #                              initializer=W1))
                # W1s_ = W1s_ * self._T

            for t in range (self._T):
                alphas_.append(tf.get_variable(name="alpha_%d"%(t+1),
                                               dtype=tf.float32,
                                               initializer=1.0))
                thetas_.append (tf.get_variable (name="theta_%d"%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=self._theta))
                mus_ = np.zeros (self._T, dtype=np.float32)
                mus_[:t+1] = 0.3
                mus_list_.append(tf.get_variable(name="mu_%d"%(t+1),
                                               dtype=tf.float32,
                                               initializer=mus_))               
                if self._untied: # untied model
                    Us_.append (tf.get_variable (name="U_%d"%(t+1),
                                dtype=tf.float32,
                                initializer=self._A_mod))                
                # if self._untied: # untied model
                #     Ws_.append (tf.get_variable (name="W_%d"%(t+1),
                #                                  dtype=tf.float32,
                #                                  initializer=W))
                    # W1s_.append (tf.get_variable (name="W1_%d"%(t+1),
                    #                              dtype=tf.float32,
                    #                              initializer=W1))
        # Collection of all trainable variables in the model layer by layer.
        # We name it as `vars_in_layer` because we will use it in the manner:
        # vars_in_layer [t]
        # self.vars_in_layer = list (zip (Ws_, W1s_, thetas_, mus_))
        self.vars_in_layer = list (zip (Us_, alphas_, thetas_, mus_list_))


    def inference (self, y_, x0_=None):
        xhs_  = [] # collection of the regressed sparse codes

        if x0_ is None:
            batch_size = tf.shape (y_) [-1]
            xh_ = tf.zeros (shape=(self._N, batch_size), dtype=tf.float32)
        else:
            xh_ = x0_
        xhs_.append (xh_)

        with tf.variable_scope (self._scope, reuse=True) as vs:
            # AT = tf.transpose(self._kA_)
            
            # use modified space
            AT_ori = tf.transpose(self._kA_)
            AT = tf.matmul (AT_ori, self._U_) # A.T becomes A.T@U
                        
            for t in range (self._T):
                U_, alpha_, theta_, mus_ = self.vars_in_layer [t]
                percent = self._ps [t]

                AT = tf.matmul (AT_ori, U_) # A.T becomes A.T@U
                W_ = alpha_* AT
                # res_ = y_ - tf.matmul (self._kA_, xh_)
                
                if t>1:
                    xh1_ = xhs_[-2] # x_k-1
                    # res_ = y_ - tf.matmul (self._kA_, xh_) - mu_*tf.matmul (self._kA_, xh1_)
                    # xh_ = shrink_ss (xh_ + mu_*xh1_ + tf.matmul (W_, res_), theta_, percent)
                    res_momentum = 0.0
                    momentum = 0.0
                    for i,xhk_ in enumerate(xhs_[:-1]):
                        momentum = momentum + mus_[i]*xhk_
                        res_momentum = res_momentum + mus_[i]*tf.matmul (self._kA_, xhk_)
                    res_ = y_ - tf.matmul (self._kA_, xh_) - res_momentum
                    xh_ = shrink_ss (xh_ + momentum + tf.matmul (W_, res_), theta_, percent)
                else:
                    # res_ = y_ - tf.matmul (self._kA_, xh_) - mu_*tf.matmul (self._kA_, xhs_[0])
                    # xh_ = shrink_ss (xh_ + mu_*xhs_[0] + tf.matmul (W_, res_), theta_, percent)
                    res_momentum = mus_[0]*tf.matmul (self._kA_, xhs_[0])
                    momentum = mus_[0]*xhs_[0]
                    res_ = y_ - tf.matmul (self._kA_, xh_) - res_momentum
                    xh_ = shrink_ss (xh_ + momentum + tf.matmul (W_, res_), theta_, percent)
                
                # xh_ = shrink_ss (xh_ + tf.matmul (W_, res_), theta_, percent)
                xhs_.append (xh_)

        return xhs_

