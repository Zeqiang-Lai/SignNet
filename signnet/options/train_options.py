from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--train-set', type=str, default='hkx_split/train')
        parser.add_argument('--val-set', type=str, default='hkx_split/validation')

        parser.add_argument('--epochs', type=int, default=5, metavar='N',
                            help='number of epochs to train (default: 14)')
        parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                            help='learning rate (default: 1e-4)')
        parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                            help='Learning rate step gamma (default: 0.7)')

        parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
        parser.add_argument('--dry-run', action='store_true', default=False,
                            help='quickly check a single pass')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')

        parser.add_argument('--val-interval', type=int, default=1)
        parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                            help='how many batches to wait before logging training status')
        parser.add_argument('--save-interval', type=int, default=1,
                            help='how many epochs to wait before saving a model')

        parser.add_argument('--save-model', action='store_true', default=True,
                            help='For Saving the current Model')

        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        self.isTrain = True
        return parser
