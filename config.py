class Config(object):

    def __init__(self):
        # directories
        self.save_dir = ''
        self.log_dir = ''
        self.train_data_file = ''
        self.val_data_file = ''

        # parameters
        self.patch_size = [4, 4, 1]
        self.N = self.patch_size[0] * self.patch_size[1]

        # 3D gconv params
        self.rank_theta = 11
        self.stride = self.Nfeat / 3
        self.stride_pregconv = self.Nfeat / 3
        self.min_nn = 16 + 8
        self.min_depth_nn = 2
        self.depth = 3
        self.input_channel = 1
        self.output_channel = 1

        # learning
        self.batch_size = 12
        self.grad_accum = 1
        self.N_iter = 400000
        self.starter_learning_rate = 1e-4
        self.end_learning_rate = 1e-5
        self.decay_step = 1000
        self.decay_rate = (self.end_learning_rate / self.starter_learning_rate) ** (
                    float(self.decay_step) / self.N_iter)
        self.Ngpus = 2

        # debugging
        self.save_every_iter = 250
        self.summaries_every_iter = 5
        self.validate_every_iter = 100
        self.test_every_iter = 250

        # testing
        self.minisize = 49 * 3  # must be integer multiple of search window
        self.search_window = [49, 49]
        self.searchN = self.search_window[0] * self.search_window[1]
