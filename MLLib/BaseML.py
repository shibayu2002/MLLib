#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from pandas import Series,DataFrame
from keras.callbacks import EarlyStopping
from keras import models
import pandas as pd
import numpy as np
import logging

# ML基底クラス
class BaseML(metaclass=ABCMeta):
  tran_ds_file = './tran.csv'  # 学習用データファイル
  test_ds_file = './test.csv'  # テスト用データファイル
  tran_result = './result/tran_result.png' # 学習結果ファイル
  predict_result = './result/predict_result{0}.png' # 予想結果ファイル
  predict_result_csv = './result/predict_result.csv' # 予想結果ファイル(csv)
  model_file = './model/model.json' # モデルファイル
  hdf5_file = './model/param.hdf5' # 学習済パラメタファイル
  batch_size = 300 # バッチサイズ
  epochs = 100 # Epochs
  validation_split = 0.1  # 学習データの中の訓練データとテストデータの割合
  
  # コンストラクタ
  def __init__(self):
    self._label = None      # ラベルデータ
    self._tran_In = None    # 入力データ
    self._tran_Out = None   # 出力データ(教師データ)
    self.model = None       # 学習モデル

  # メイン処理
  def run(self, t='all'):
    logging.info('this target is [' + t + ']')
    
    # 学習と予測を実施
    if t == 'all':
      self.init('tran')
      self.tran()
      self.init('test')
      self.predict()
    # 学習のみ実施
    elif t == 'tran':
      self.init('tran')
      self.tran()
    # 予測のみ実施
    elif t == 'test':
      self.init('test')
      self.predict()
    # パラメタエラー
    else:
      print("run('all' | 'tran' | 'test')")
  
  # 準備処理(学習データの読み込み)
  def init(self, t):
    path = self.tran_ds_file
    if t == 'test':
      path = self.test_ds_file

    logging.info('dataset is [' + path + ']')
    csv = pd.read_csv(path, sep=',', header=0)

    logging.info('-- data label---- --------------------------')
    self._label = csv.loc[:, csv.columns.str.contains('label.*')]
    logging.info(self._label)
    logging.info('-- data tran_In(x) --------------------------')
    self._tran_In = csv.loc[:, csv.columns.str.contains('x.*')]
    logging.info(self._tran_In)
    
    logging.info('-- data tran_Out(y) -------------------------')
    self._tran_Out = csv.loc[:, csv.columns.str.contains('y.*')]
    logging.info(self._tran_Out)

  # 学習
  def tran(self):
    # 学習
    early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=20)
    self.model.fit(self._tran_In, self._tran_Out,
      batch_size = self.batch_size,
      epochs = self.epochs,
      validation_split = self.validation_split,
      shuffle = False,
      callbacks=[early_stopping]
      )
    # 学習済モデル保存
    json = self.model.to_json()
    with open(self.model_file, mode='w') as f:
      f.write(json)
    logging.info('model_file saved to ' + self.model_file)
    # 学習済パラメタ保存
    self.model.save_weights(self.hdf5_file)
    logging.info('hdf5_file saved to ' + self.hdf5_file)
    # 学習結果を描画
    out = self.model.predict(self._tran_In)
    self.draw_tran_result(self._tran_Out, out)
    logging.info('tran_result saved to ' + self.tran_result)

  # 学習結果の描画(派生クラスで実装する)
  @abstractmethod
  def draw_tran_result(self, expected, actual):
    pass

  # 予測
  def predict(self):
    # 学習済モデル読込
    with open(self.model_file) as f:
      json = f.read()
    self.model = models.model_from_json(json)
    logging.info('model_file load from ' + self.model_file)
    # 学習済パラメタの読み込み
    self.model.load_weights(self.hdf5_file)
    # 予測処理
    expected_diff, actual_diff, expected, actual= self.predict_impl()
    # 予測結果を描画
    self.draw_predict_result(self._label, expected_diff, actual_diff, expected, actual)
    logging.info('predict_result saved to ' + self.predict_result)

  # 予測処理の実態(派生クラスで実装する)
  @abstractmethod
  def predict_impl(self):
    pass
    
  # 予測結果の描画(派生クラスで実装する)
  @abstractmethod
  def draw_predict_result(self, label, expected_diff, actual_diff, expected, actual):
    pass
