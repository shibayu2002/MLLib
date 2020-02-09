#!/usr/bin/env python
# -*- coding: utf-8 -*-
from MLLib import LstmML as LstmML
import logging
import sys

if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  ml = LstmML(25, 1, 300)
  ml.epochs = 30
  
  if len(sys.argv) >1 :
    ml.run(sys.argv[1])
  else:
    ml.run()
