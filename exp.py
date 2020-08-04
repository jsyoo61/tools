class Experiment(object):
    '''
    ------------------
    Data descriptors:
    dirs
        Hold all directory information to be used in the experiment
        in dict() form.
        To see all directories that are related, refer to:
        list(self.dirs.items())
    model_p
        Hold all parameters related to NN model learning in dict() form
    train_p
        Hold all parameters related to training in dict() form
    speaker_list
        Hold all name of speakers
    '''
    def __init__(self, num_speakers = 4, exp_name = None, exp_dir='exp', new = True, model_p = None, train_p = None, lambd = None, debug = False):
        # 0] Random seed
        np.random.seed(0)
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1] Hyperparameters setting - Default
        self.model_p = dict(
        vae_lr = 1e-2,
        vae_betas = (0.9,0.999),
        sc_lr = 0.0002,
        sc_betas = (0.5,0.999),
        asr_lr = 0.00001,
        asr_betas = (0.5,0.999),
        ac_lr = 0.00005,
        ac_betas = (0.5,0.999),
        )
        if model_p is not None:
            self.model_p.update(model_p)

        self.train_p = dict(
        n_train_frames = 128,
        batch_size = 32,
        mini_batch_size = 8,
        start_epoch = 1,
        n_epoch = 300,
        model_save_epoch = 3,
        validation_epoch = 3,
        sample_per_path = 10,
        )
        self.train_p['iter_per_ep'] = self.train_p['batch_size'] // self.train_p['mini_batch_size']
        if train_p is not None:
            self.train_p.update(train_p)
        try:
            assert self.train_p['iter_per_ep'] * self.train_p['mini_batch_size'] == self.train_p['batch_size'], 'Specified batch_size "%s" cannot be divided by mini_batch_size "%s"'%(self.train_p['batch_size'], self.train_p['mini_batch_size'])
        except:
            print("Invalid train_p['iter_per_ep'] setting!")
            print('iter_per_ep: %s'%self.train_p['iter_per_ep'])
            print('batch_size: %s'%self.train_p['batch_size'])
            print('mini_batch_size: %s'%self.train_p['mini_batch_size'])
            print('Setting (iter_per_ep) = (batch_size) // (mini_batch_size)')
            self.train_p['iter_per_ep'] = self.train_p['batch_size'] // self.train_p['mini_batch_size']
        self.train_p['epoch'] = self.train_p['start_epoch'] - 1

        self.lambd = dict(
        KLD = 1,
        rec = 20,
        SI = 0,
        LI = 0,
        AC = 0,
        SC = 0,
        C = 0,
        CC = 0,
        )
        if lambd is not None:
            self.lambd.update(lambd)
        self.lambd_total = sum(self.lambd.values())
        if self.lambd['C'] is not 0:
            self.lambd_total -= self.lambd['C']
            self.lambd_total += self.lambd['C'] * (self.lambd['KLD'] + self.lambd['rec'])
        self.lambda_norm = True

        self.preprocess_p = dict(
        sr = 16000,
        frame_period = 5.0,
        num_mcep = 36,
        )
        self.loss_index = ['loss_VAE','loss_KLD','loss_rec','loss_SI','loss_LI','loss_AC','loss_SC','loss_C_KLD', 'loss_C_rec']
        self.performance_measure_index = ['mcd', 'msd_vector', 'gv']
        self.lr_index = ['VAE_lr']
        self.loss_summary = pd.DataFrame(columns = self.loss_index)
        self.validation_summary = pd.DataFrame(columns = self.performance_measure_index)
        self.lr_summary = pd.DataFrame(columns = self.lr_index)

        self.model_kept = []
        self.max_keep=100

        # 2] Initialize environment and variables
        self.create_env(exp_dir = exp_dir, exp_name = exp_name, new = new)
        # self.speaker_list = sorted(os.listdir(self.dirs['train_data']))
        self.speaker_list = ['p225','p226','p227','p228']
        self.num_speakers = len(self.speaker_list)
        assert self.num_speakers == num_speakers, 'Specified "num_speakers" and "num_speakers in train data" does not match'
        self.build_model(params = self.model_p)
        self.p = Printer(filewrite_dir = self.dirs['log'])
        if debug == False:
            sys.stdout = open(self.dirs['log_all'], 'a')
        append(self.dirs['loss_log'], 'epoch '+' '.join(self.loss_index)+'\n')
        append(self.dirs['validation_log'], 'epoch '+' '.join(self.performance_measure_index)+'\n')

        # 3] Hyperparameters for saving model


        # 4] If the experiment is not new, Load most recent model
        if new == False:
            self.model_kept= sorted(os.listdir(self.dirs['model']), key = lambda x: int(x.split('.')[0].split('_')[-1]))
            most_trained_model = self.model_kept[-1]
            epoch_trained = int(most_trained_model.split('_')[-1].split('.')[0])
            self.train_p['start_epoch'] += epoch_trained
            # Update lr_scheduler
            print('Loading model from %s'%most_trained_model)
            self.load_model_all(self.dirs['model'], epoch_trained)

    def create_env(self, exp_dir = 'exp', exp_name = None, new = True):
        '''Create experiment environment
        Store all "static directories" required for experiment in "self.dirs"(dict)

        Store every experiment result in: exp/exp_name/ == exp_dir
        including log, model, test(validation) etc
        '''
        # 0] exp_dir == master directory
        self.dirs = dict()
        # exp_dir = 'exp/'
        model_dir = 'model/'

        # 1] Set up Experiment directory
        if exp_name == None:
            exp_name = time.strftime('%m%d_%H%M%S')
        self.dirs['exp'] = os.path.join(exp_dir, exp_name)
        if new == True:
            assert not os.path.isdir(self.dirs['exp']), 'New experiment, but exp_dir with same name exists'
            os.makedirs(self.dirs['exp'])
        else:
            assert os.path.isdir(self.dirs['exp']), 'Existing experiment, but exp_dir doesn\'t exist'

        # 2] Model parameter directory

    def save_log(self, result, log_dir):
        # 1. Write to log
        log_content = str(self.train_p['epoch'])
        for value in result.mean():
            log_content += ' '+str(value)
        append(log_dir, log_content+'\n')
        # 2. Print result statistics
        self.p.print('Mean\n' + str(result.mean().to_frame().T))
        self.p.print('Std\n' + str(result.std().to_frame().T))

    def save_plot(self, summary, plot_dir):
        for measure in summary.columns:
            fig_save_dir = os.path.join(plot_dir, measure+'.png')
            axes = summary.plot(y = measure, style='o-')
            fig = axes.get_figure()
            fig.savefig(fig_save_dir)
        plt.close('all')

    def performance_measure(self):
        pass
