#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from pandas import Series,DataFrame
from keras.callbacks import EarlyStopping
from keras import models
import pandas as pd
import numpy as np
import logging

# ML���N���X
class BaseML(metaclass=ABCMeta):
  tran_ds_file = './tran.csv'  # �w�K�p�f�[�^�t�@�C��
  test_ds_file = './test.csv'  # �e�X�g�p�f�[�^�t�@�C��
  tran_result = './result/tran_result.png' # �w�K���ʃt�@�C��
  predict_result = './result/predict_result{0}.png' # �\�z���ʃt�@�C��
  predict_result_csv = './result/predict_result.csv' # �\�z���ʃt�@�C��(csv)
  model_file = './model/model.json' # ���f���t�@�C��
  hdf5_file = './model/param.hdf5' # �w�K�σp�����^�t�@�C��
  batch_size = 300 # �o�b�`�T�C�Y
  epochs = 100 # Epochs
  validation_split = 0.1  # �w�K�f�[�^�̒��̌P���f�[�^�ƃe�X�g�f�[�^�̊���
  
  # �R���X�g���N�^
  def __init__(self):
    self._label = None      # ���x���f�[�^
    self._tran_In = None    # ���̓f�[�^
    self._tran_Out = None   # �o�̓f�[�^(���t�f�[�^)
    self.model = None       # �w�K���f��

  # ���C������
  def run(self, t='all'):
    logging.info('this target is [' + t + ']')
    
    # �w�K�Ɨ\�������{
    if t == 'all':
      self.init('tran')
      self.tran()
      self.init('test')
      self.predict()
    # �w�K�̂ݎ��{
    elif t == 'tran':
      self.init('tran')
      self.tran()
    # �\���̂ݎ��{
    elif t == 'test':
      self.init('test')
      self.predict()
    # �p�����^�G���[
    else:
      print("run('all' | 'tran' | 'test')")
  
  # ��������(�w�K�f�[�^�̓ǂݍ���)
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

  # �w�K
  def tran(self):
    # �w�K
    early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=20)
    self.model.fit(self._tran_In, self._tran_Out,
      batch_size = self.batch_size,
      epochs = self.epochs,
      validation_split = self.validation_split,
      shuffle = False,
      callbacks=[early_stopping]
      )
    # �w�K�σ��f���ۑ�
    json = self.model.to_json()
    with open(self.model_file, mode='w') as f:
      f.write(json)
    logging.info('model_file saved to ' + self.model_file)
    # �w�K�σp�����^�ۑ�
    self.model.save_weights(self.hdf5_file)
    logging.info('hdf5_file saved to ' + self.hdf5_file)
    # �w�K���ʂ�`��
    out = self.model.predict(self._tran_In)
    self.draw_tran_result(self._tran_Out, out)
    logging.info('tran_result saved to ' + self.tran_result)

  # �w�K���ʂ̕`��(�h���N���X�Ŏ�������)
  @abstractmethod
  def draw_tran_result(self, expected, actual):
    pass

  # �\��
  def predict(self):
    # �w�K�σ��f���Ǎ�
    with open(self.model_file) as f:
      json = f.read()
    self.model = models.model_from_json(json)
    logging.info('model_file load from ' + self.model_file)
    # �w�K�σp�����^�̓ǂݍ���
    self.model.load_weights(self.hdf5_file)
    # �\������
    expected_diff, actual_diff, expected, actual= self.predict_impl()
    # �\�����ʂ�`��
    self.draw_predict_result(self._label, expected_diff, actual_diff, expected, actual)
    logging.info('predict_result saved to ' + self.predict_result)

  # �\�������̎���(�h���N���X�Ŏ�������)
  @abstractmethod
  def predict_impl(self):
    pass
    
  # �\�����ʂ̕`��(�h���N���X�Ŏ�������)
  @abstractmethod
  def draw_predict_result(self, label, expected_diff, actual_diff, expected, actual):
    pass
