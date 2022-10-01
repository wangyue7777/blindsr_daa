import os
from data_nn import multiscalesrdata


class DF2K(multiscalesrdata.SRData):
    def __init__(self, args, name='DF2K', train=True, benchmark=False):
        super(DF2K, self).__init__(args, name=name, train=train, benchmark=benchmark)

    def _scan(self):
        names_hr = super(DF2K, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]

        return names_hr

    def _set_filesystem_train(self, dir_data):
        # super(DF2K, self)._set_filesystem_train(dir_data)
        # self.dir_hr = os.path.join(self.apath, 'HR')
        # self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.apath = dir_data
        self.dir_hr = os.path.join(self.apath, 'DIV_FLI/','HR')
        self.dir_lr = os.path.join(self.apath, 'DIV_FLI/', 'LR_x4')
        self.ext = ('.png','.png')
        print(self.dir_hr)
        print(self.dir_lr)

