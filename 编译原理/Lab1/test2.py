class DCGAN:
    def __init__(self, d_model, g_model,
                 input_dim=784, g_dim=100,
                 max_step=100, sample_size=256, d_iter=3, kind='normal'):
        self.input_dim = input_dim #12 
        self.g_dim = g_dim  # 13.
        self.max_step =max_step  # 14. 
        self.sample_size = sample_size
        self.d_iter = d_iter
        self.kind = kind
        self.d_model = d_model  # 
        self.g_model = g_model  # 
        self.m_model = self.merge_model()  # 
        self.optimizer = adam_v2.Adam(lr=0.002, beta_1=0.5)
        self.d_model.compile(optimizer=self.optimizer, loss='binary_crossentropy')

    def merge_model(self):
        noise = Input(shape=(self.g_dim,))
        gen_sample = self.g_model(noise)
        self.d_model.trainable =False
        d_output = self.d_model(gen_sample)
        m_model = Model(noise, d_output)
        m_model.compile(optimizer='adam', loss='binary_crossentropy')
        return m_model

    def gen_noise(self, num_sample):
         if self.kind == 'normal':
            f = normal_sampling
        elif self.kind == 'uniform':
            f = uniform_sampling
        else:
            raise ValueError(' {}'.format(self.kind))
        return f(num_sample, self.g_dim)

    def gen_real_data(self, train_data):
        n_samples = train_data.shape[0]
        inds = np.random.randint(0, n_samples, size=self.sample_size)
        real_data = train_data[inds]
        real_label = np.random.uniform(0,0.3,size=(self.sample_size,))
