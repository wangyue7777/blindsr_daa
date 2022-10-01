import os
from data_nn import common
from data_nn import multiscalesrdata_realsr as srdata


class RealSR_TEST(srdata.SRData):
    def __init__(self, args, name='', train=True):
        super(RealSR_TEST, self).__init__(
            args, name=name, train=train, benchmark=True
        )

    def _set_filesystem(self, dir_data):
        self.apath = dir_data
        self.dir_hr = os.path.join(self.apath, 'RealSR(V3)/test_x4','HR')
        self.dir_lr = os.path.join(self.apath, 'RealSR(V3)/test_x4', 'LR')
        self.ext = ('.png','.png')
        print(self.dir_hr)
        print(self.dir_lr)

    def _set_filesystem_train(self, dir_data):
        self.apath = dir_data
        self.dir_hr = os.path.join(self.apath, 'RealSR(V3)/train_x4','HR_sub')
        self.dir_lr = os.path.join(self.apath, 'RealSR(V3)/train_x4', 'LR_sub')
        self.ext = ('.png','.png')
        print(self.dir_hr)
        print(self.dir_lr)
