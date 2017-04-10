from __future__ import print_function
import numpy as np
import time

class Logger():
    def __init__(self,file_name=None,show=True):
        self.show=show
        self.file_name=file_name

    def __call__(self,str):
        str='%s  '%(time.strftime('%H:%M:%S'),)+str
        if self.file_name:
            with open(self.file_name,'a+') as f:
                f.write(str+'\n')
        if self.show:
            print(str)