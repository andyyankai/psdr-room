class FineStageConfig:
    def __init__(self):

        self.toolkit_path = 'C:/Program Files/Adobe/Adobe Substance 3D Designer/'
        self.forward_npass = 16
        self.backward_npass = 4

        self.spp = 8
        self.bounce = 2

        self.num_iter = 201
        self.mgt_res = 8

        self.vggresizeres = 512



        self.lr_color = 0.01
        self.lr_emi = 0.05
        self.lr_rot = 0.005
        self.lr_scale = 0.01
        self.lr_mat_norm_opt = 0.1
