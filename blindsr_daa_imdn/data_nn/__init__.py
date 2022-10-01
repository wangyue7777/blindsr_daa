from importlib import import_module
from dataloader import MSDataLoader

class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            module_train = import_module('data_nn.' + args.data_train.lower())     ## load the right dataset loader module
            trainset = getattr(module_train, args.data_train)(args)             ## load the dataset, args.data_train is the  dataset name
            self.loader_train = MSDataLoader(
                args,
                trainset,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu
            )

        if args.data_test in ['Set5', 'Set14', 'B100', 'Manga109', 'Urban100']:
            module_test = import_module('data_nn.benchmark_test')
            testset = getattr(module_test, 'Benchmark')(args, name=args.data_test,train=False)
            # module_test = import_module('data_nn.benchmark')
            # testset = getattr(module_test, 'Benchmark')(args, name=args.data_test,train=False)
        else:
            module_test = import_module('data_nn.' + args.data_test.lower())
            testset = getattr(module_test, args.data_test)(args, train=False)

        if args.data_test_tt in ['Set5_test', 'Set14_test', 'B100_test', 'Manga109_test', 'Urban100_test']:
            module_test_tt = import_module('data_nn.benchmark_test1')
            testset_tt = getattr(module_test_tt, 'Benchmark')(args, name=args.data_test_tt,train=False)
            # module_test = import_module('data_nn.benchmark')
            # testset = getattr(module_test, 'Benchmark')(args, name=args.data_test,train=False)
        else:
            module_test_tt = import_module('data_nn.' + args.data_test_tt.lower())
            testset_tt = getattr(module_test_tt, args.data_test_tt)(args, train=False)

        if args.data_test1 in ['Set5', 'Set14', 'B100', 'Manga109', 'Urban100']:
            module_test1 = import_module('data_nn.benchmark')
            testset1 = getattr(module_test1, 'Benchmark')(args, name=args.data_test1,train=False)
        else:
            module_test1 = import_module('data_nn.' + args.data_test1.lower())
            testset1 = getattr(module_test1, args.data_test1)(args, train=False)

        self.loader_test = MSDataLoader(
            args,
            testset,
            batch_size=1,
            shuffle=False,
            pin_memory=not args.cpu
        )

        self.loader_test_tt = MSDataLoader(
            args,
            testset_tt,
            batch_size=1,
            shuffle=False,
            pin_memory=not args.cpu
        )

        self.loader_test1 = MSDataLoader(
            args,
            testset1,
            batch_size=1,
            shuffle=False,
            pin_memory=not args.cpu
        )

