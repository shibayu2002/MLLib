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
import csv

class LstmML(BaseML):
  predict_term = 100 # �\������
  multi_result = 1   # �\�z���ʂ̑����{��

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
  def init(self, t):
    super().init(t)
    
    # �e�N���X�œǂݍ��񂾊w�K�f�[�^��LSTM�ň�����l�ɉ��H
    logging.info('-- reshape data tran_In --------------------------')
    x = self._tran_In
    self._tran_In = np.array(x).reshape(len(x), x.shape[1] , 1)
    logging.info(len(self._tran_In))
    
    logging.info('-- reshape data tran_Out --------------------------')
    y = self._tran_Out
    if t == 'tran':
      self._tran_Out = np.array(y).reshape(len(y), 1)
    else:
      self._tran_Out = np.array(y).reshape(len(y), y.shape[1])
    logging.info(len(self._tran_Out))
      
  # �w�K���ʂ̕`��
  def draw_tran_result(self, expected, actual):
    actual = actual * self.multi_result
    plt.figure()
    plt.plot(range(0, len(expected)), expected, color="r", label="real")
    plt.plot(range(0, len(actual)), actual, color="b", label="ai")
    plt.legend()
    plt.savefig(self.tran_result)
    plt.close()

  # �\�������̎���(LSTM�����\��)
  def predict_impl(self):
    expected_diff = []
    actual_diff = []
    expected = []
    actual = []
    
    i = 0
    for curIn in self._tran_In:
      logging.info('-- data start curIn(x) --------------------------')
      logging.info(len(curIn))
      
      # �w�肳�ꂽ���Ԃ܂ŏ��Ԃɗ\������
      # 0 = �w�K�f�[�^�̍Ō�
      # 1�` = �����̗\��
      outSumDiff = np.empty((0))
      outSum = np.empty((0))
      tmpSum = 0
      term = 0
      while term < self.predict_term:
        # 1��̖�����\��
        curOut = self.model.predict(np.reshape(curIn, (1, len(curIn), 1)))
        # �\�����ʂ�curIn�̍Ō�ɒǉ��B��ԌÂ�curIn�̒l�͍폜
        curIn = np.delete(curIn, 0)
        curIn = np.append(curIn, curOut)
        # �\�z���ʂ��������Ă�����ۂ��̂ő���
        curOut = curOut * self.multi_result
        # �\�����ʂ�\�����ʏW�v�ɒǉ�
        outSumDiff = np.append(outSumDiff, curOut)
        tmpSum = tmpSum + curOut
        outSum = np.append(outSum, tmpSum)
        if (term != 0) and (term % 10 == 0):
          logging.info('term ' + str(term) + ' finished.')
        term += 1
      actual_diff.append(outSumDiff)
      actual.append(outSum)
      
      ySumDiff = np.empty((0))
      ySum = np.empty((0))
      tmpY = 0
      for y in self._tran_Out[i]:
        ySumDiff = np.append(ySumDiff, y)
        tmpY = tmpY + y
        ySum = np.append(ySum, tmpY)
      expected_diff.append(ySumDiff)
      expected.append(ySum)
      i = i + 1
    return expected_diff, actual_diff, expected, actual
    
  # �\�����ʂ̕`��
  def draw_predict_result(self, label, expected_diff, actual_diff, expected, actual):
    with open(self.predict_result_csv, 'w') as f:
      writer = csv.writer(f)
      header = []
      header.append('label')
      for i in range(0, len(expected[0])):
        header.append('y' + str(i + 1))
      for i in range(0, len(actual[0])):
        header.append("y'" + str(i + 1))
      writer.writerow(header)
      
      for i in range(0, len(actual)):
        plt.figure()
        plt.plot(range(0, len(expected_diff[i])), expected_diff[i], color="g", linestyle="dotted", label="real(diff)")
        plt.plot(range(0, len(actual_diff[i])), actual_diff[i], color="y", linestyle="dotted", label="ai(diff)")
        plt.plot(range(0, len(expected[i])), expected[i], color="r", label="real")
        plt.plot(range(0, len(actual[i])), actual[i], color="b", label="ai")
        plt.legend()
        plt.savefig(self.predict_result.format(label.iloc[i,0]))
        plt.close()
        
        row = []
        row.append(label.iloc[i,0])
        row.extend(expected[i])
        row.extend(actual[i])
        writer.writerow(row)
