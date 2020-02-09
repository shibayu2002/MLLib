#!/usr/bin/env python
# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from .BaseML import BaseML as BaseML
import numpy as np
import matplotlib.pyplot as plt
import logging

class LstmML(BaseML):
  predict_term = 100 # �\������

  # �R���X�g���N�^
  def __init__(self, in_size, out_size, n_hidden):
    super().__init__()

    # LSTM�w�K���f���̐���
    self.model = Sequential()
    self.model.add(LSTM(n_hidden, batch_input_shape=(None, in_size, out_size), return_sequences=False))
    self.model.add(Dense(out_size))
    self.model.add(Activation("linear"))
    optimizer = Adam(lr=0.001)
    self.model.compile(loss="mean_squared_error", optimizer=optimizer)

  # ��������(�w�K�f�[�^�̓ǂݍ��݂Ɖ��H)
  def init(self):
    super().init()
    
    # �e�N���X�œǂݍ��񂾊w�K�f�[�^��LSTM�ň�����l�ɉ��H
    logging.info('-- reshape data tran_In --------------------------')
    x = self._tran_In
    self._tran_In = np.array(x).reshape(len(x), x.shape[1] , 1)
    logging.info(len(self._tran_In))
    
    logging.info('-- reshape data tran_Out --------------------------')
    y = self._tran_Out
    self._tran_Out = np.array(y).reshape(len(y), 1)
    logging.info(len(self._tran_Out))

  # �w�K���ʂ̕`��
  def draw_tran_result(self, expected, actual):
    plt.figure()
    plt.plot(range(0, len(actual)), actual, color="b", label="actual")
    plt.plot(range(0, len(expected)), expected, color="r", label="expected")
    plt.legend()
    plt.savefig(self.tran_result)

  # �\�������̎���(LSTM�����\��)
  def predict_impl(self):
    # �w�K�f�[�^�̍Ō�̃f�[�^���擾
    curIn = self._tran_In[len(self._tran_In) - 1]
    logging.info('-- data start curIn(x) --------------------------')
    logging.info(curIn)
    
    # �w�肳�ꂽ���Ԃ܂ŏ��Ԃɗ\������
    # 0 = �w�K�f�[�^�̍Ō�
    # 1�` = �����̗\��
    outSum = np.empty((0))
    term = 0
    while term <= self.predict_term:
      # 1��̖�����\��
      curOut = self.model.predict(np.reshape(curIn, (1, len(curIn), 1)))
      # �\�����ʂ�curIn�̍Ō�ɒǉ��B��ԌÂ�curIn�̒l�͍폜
      curIn = np.delete(curIn, 0)
      curIn = np.append(curIn, curOut)
      # �\�����ʂ�\�����ʏW�v�ɒǉ�
      outSum = np.append(outSum, curOut)
      if (term != 0) and (term % 10 == 0):
        logging.info('term ' + str(term) + ' finished.')
      term += 1
    return outSum
    
  # �\�����ʂ̕`��
  def draw_predict_result(self, actual):
    plt.figure()
    plt.plot(range(0, len(actual)), actual, color="b", label="actual")
    plt.savefig(self.predict_result)
