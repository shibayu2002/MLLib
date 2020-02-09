#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from pandas import Series,DataFrame
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import logging

# ML���N���X
class BaseML(metaclass=ABCMeta):
  tran_ds_file = './tran.csv'  # �w�K�f�[�^�t�@�C��
  tran_result = './tran_result.png' # �w�K���ʃt�@�C��
  predict_result = './predict_result.png' # �\�z���ʃt�@�C��
  hdf5_file = './param.hdf5' # �w�K�σp�����^�t�@�C��
  batch_size = 300 # �o�b�`�T�C�Y
  epochs = 100 # Epochs
  validation_split = 0.1  # �w�K�f�[�^�̒��̌P���f�[�^�ƃe�X�g�f�[�^�̊���
  
  # �R���X�g���N�^
  def __init__(self):
    self._tran_In = None    # ���̓f�[�^
    self._tran_Out = None   # �o�̓f�[�^(���t�f�[�^)
    self.model = None       # �w�K���f��

  # ���C������
  def run(self, t='all'):
    logging.info('this target is [' + t + ']')
    
    # �w�K�Ɨ\�������{
    if t == 'all':
      self.init()
      self.tran()
      self.predict()
    # �w�K�̂ݎ��{
    elif t == 'tran':
      self.init()
      self.tran()
    # �\���̂ݎ��{
    elif t == 'pre':
      self.init()
      self.predict()
    # �p�����^�G���[
    else:
      print("run('all' | 'tran' | 'pre')")
  
  # ��������(�w�K�f�[�^�̓ǂݍ���)
  def init(self):
    logging.info('dataset is [' + self.tran_ds_file + ']')
    csv = pd.read_csv(self.tran_ds_file, sep=',', header=0)
    
    logging.info('-- data tran_In(x) --------------------------')
    self._tran_In = DataFrame(csv.drop('y',axis=1))
    logging.info(self._tran_In)
    
    logging.info('-- data tran_Out(y) -------------------------')
    self._tran_Out = DataFrame(csv['y'])
    logging.info(self._tran_Out)

  # �w�K
  def tran(self):
    # �w�K
    early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=20)
    self.model.fit(self._tran_In, self._tran_Out,
      batch_size = self.batch_size,
      epochs = self.epochs,
      validation_split = self.validation_split,
      callbacks=[early_stopping]
      )
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
    # �w�K�σp�����^�̓ǂݍ���
    self.model.load_weights(self.hdf5_file)
    # �\������
    actual = self.predict_impl()
    # �\�����ʂ�`��
    self.draw_predict_result(actual)
    logging.info('predict_result saved to ' + self.predict_result)

  # �\�������̎���(�h���N���X�Ŏ�������)
  @abstractmethod
  def predict_impl(self):
    pass
    
  # �\�����ʂ̕`��(�h���N���X�Ŏ�������)
  @abstractmethod
  def draw_predict_result(self, actual):
    pass
