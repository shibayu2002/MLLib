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
  predict_term = 100 # 予測期間
  multi_result = 1   # 予想結果の増幅倍率

  # コンストラクタ
  def __init__(self, in_size, out_size, n_hidden):
    super().__init__()

    # LSTM学習モデルの生成
    self.model = Sequential()
    self.model.add(LSTM(n_hidden, batch_input_shape=(None, in_size, out_size), return_sequences=False))
    self.model.add(Dense(out_size))
    self.model.add(Activation("linear"))
    optimizer = Adam(lr=0.001)
    self.model.compile(loss="mean_squared_error", optimizer=optimizer)

  # 準備処理(学習データの読み込みと加工)
  def init(self, t):
    super().init(t)
    
    # 親クラスで読み込んだ学習データをLSTMで扱える様に加工
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
      
  # 学習結果の描画
  def draw_tran_result(self, expected, actual):
    actual = actual * self.multi_result
    plt.figure()
    plt.plot(range(0, len(expected)), expected, color="r", label="real")
    plt.plot(range(0, len(actual)), actual, color="b", label="ai")
    plt.legend()
    plt.savefig(self.tran_result)
    plt.close()

  # 予測処理の実態(LSTM未来予測)
  def predict_impl(self):
    expected_diff = []
    actual_diff = []
    expected = []
    actual = []
    
    i = 0
    for curIn in self._tran_In:
      logging.info('-- data start curIn(x) --------------------------')
      logging.info(len(curIn))
      
      # 指定された期間まで順番に予測する
      # 0 = 学習データの最後
      # 1〜 = 未来の予測
      outSumDiff = np.empty((0))
      outSum = np.empty((0))
      tmpSum = 0
      term = 0
      while term < self.predict_term:
        # 1つ先の未来を予測
        curOut = self.model.predict(np.reshape(curIn, (1, len(curIn), 1)))
        # 予測結果をcurInの最後に追加。一番古いcurInの値は削除
        curIn = np.delete(curIn, 0)
        curIn = np.append(curIn, curOut)
        # 予想結果が減衰しているっぽいので増幅
        curOut = curOut * self.multi_result
        # 予測結果を予測結果集計に追加
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
    
  # 予測結果の描画
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
