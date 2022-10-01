import os
from data_nn import common
from data_nn import multiscalesrdata_test as srdata


class Benchmark(srdata.SRData):
    def __init__(self, args, name='', train=True):
        super(Benchmark, self).__init__(
            args, name=name, train=train, benchmark=True
        )

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name+'_mjw', '4.0_3.0_165_10')
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.ext = ('.png','.png')
        print(self.dir_hr)
        print(self.dir_lr)
