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
  predict_term = 100 # 予測期間

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
  def init(self):
    super().init()
    
    # 親クラスで読み込んだ学習データをLSTMで扱える様に加工
    logging.info('-- reshape data tran_In --------------------------')
    x = self._tran_In
    self._tran_In = np.array(x).reshape(len(x), x.shape[1] , 1)
    logging.info(len(self._tran_In))
    
    logging.info('-- reshape data tran_Out --------------------------')
    y = self._tran_Out
    self._tran_Out = np.array(y).reshape(len(y), 1)
    logging.info(len(self._tran_Out))

  # 学習結果の描画
  def draw_tran_result(self, expected, actual):
    plt.figure()
    plt.plot(range(0, len(actual)), actual, color="b", label="actual")
    plt.plot(range(0, len(expected)), expected, color="r", label="expected")
    plt.legend()
    plt.savefig(self.tran_result)

  # 予測処理の実態(LSTM未来予測)
  def predict_impl(self):
    # 学習データの最後のデータを取得
    curIn = self._tran_In[len(self._tran_In) - 1]
    logging.info('-- data start curIn(x) --------------------------')
    logging.info(curIn)
    
    # 指定された期間まで順番に予測する
    # 0 = 学習データの最後
    # 1～ = 未来の予測
    outSum = np.empty((0))
    term = 0
    while term <= self.predict_term:
      # 1つ先の未来を予測
      curOut = self.model.predict(np.reshape(curIn, (1, len(curIn), 1)))
      # 予測結果をcurInの最後に追加。一番古いcurInの値は削除
      curIn = np.delete(curIn, 0)
      curIn = np.append(curIn, curOut)
      # 予測結果を予測結果集計に追加
      outSum = np.append(outSum, curOut)
      if (term != 0) and (term % 10 == 0):
        logging.info('term ' + str(term) + ' finished.')
      term += 1
    return outSum
    
  # 予測結果の描画
  def draw_predict_result(self, actual):
    plt.figure()
    plt.plot(range(0, len(actual)), actual, color="b", label="actual")
    plt.savefig(self.predict_result)
