#!/usr/bin/env python
# -*- coding: utf-8 -*-
from MLLib import LstmML as LstmML
import logging
import sys
import MySQLdb
import csv
import numpy as np

def connectDB():
  con = MySQLdb.connect(
      host='localhost',
      user='apl',
      passwd='common1512',
      db='fin_db',
      use_unicode=True,
      charset="utf8")
  return con
  
def makeCsvDataset(_from, _to, _path, x_size = 1, y_size = 1):
  con = connectDB()
  try:
    cur = con.cursor()
    cur.execute("select A.date, A.close "\
               "from cur_price A inner join ("\
               "select date, max(time) as time "\
               "from cur_price "\
               "where ticker = 'USDJPY' and date >= '" + _from + "' and date <= '" + _to +"'"\
               "group by date"\
               ") B on A.date = B.date and A.time = B.time "\
               "where A.ticker = 'USDJPY' and A.date >= '" + _from +"' and A.date <= '" + _to + "'"\
               "order by A.date")
    res = cur.fetchall()
    
    head = ['label']
    for i in range(1, 1 + x_size):
      head.append('x' + str(i))
    for i in range(1, 1 + y_size):
      head.append('y' + str(i))
    
    pre_price = -1
    with open(_path, 'w') as f:
      writer = csv.writer(f)
      writer.writerow(head)
    
      cols = []
      for r in res:
        date = r[0]
        price = r[1]
        if pre_price == -1:
          pre_price = price
        else:
          diff = price - pre_price
#          diff = np.tanh(price - pre_price)
          if len(cols) == x_size + (y_size - 1):
            col = cols.pop(0)
            col.insert(0, date)
            col.append(diff)
            writer.writerow(col)
          for col in cols:
            col.append(diff)
          cols.append([diff])
          pre_price = price
  finally:
    con.close()
  
if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  mode = sys.argv[1]
  if mode == 'make':
    makeCsvDataset('20180101', '20181231', 'tran.csv', 25, 1)
    makeCsvDataset('20190101', '20191231', 'test.csv', 25, 20)
    exit()
  ml = LstmML(25, 1, 300)
  ml.epochs = 100
  ml.predict_term = 20
  ml.multi_result = 3
  ml.run(sys.argv[1])
