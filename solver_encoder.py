from model_vc import Generator
import torch
import torch.nn.functional as F
import time
import datetime


class Solver(object):

    def __init__(self, vcc_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.vcc_loader = vcc_loader

        # Model configurations.
        self.lambda_cd = config.lambda_cd
        self.dim_neck = config.dim_neck
        self.dim_emb = config.dim_emb
        self.dim_pre = config.dim_pre
        self.freq = config.freq

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        
        # Miscellaneous.
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        self.log_step = config.log_step

        # Build the model and tensorboard.
        self.build_model()

            
    def build_model(self):
        
        self.G = Generator(self.dim_neck, self.dim_emb, self.dim_pre, self.freq)        
        
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), 0.0001)
        
        self.G.to(self.device)
        

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
      
    
    #=====================================================================================================================================#
    
    
                
    def train(self):
        # Set data loader.
        data_loader = self.vcc_loader
        
        # Print logs in specified order
        keys = ['G/loss_id','G/loss_id_psnt','G/loss_cd']
            
        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch data.
            try:
                x_real, emb_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, emb_org = next(data_iter)
            
            
            x_real = x_real.to(self.device) 
            emb_org = emb_org.to(self.device) 
                        
       
            # =================================================================================== #
            #                               2. Train the generator                                #
            # =================================================================================== #
            
            self.G = self.G.train()
                        
            # Identity mapping loss
            x_identic, x_identic_psnt, code_real = self.G(x_real, emb_org, emb_org)
            # g_loss_id = F.mse_loss(x_real, x_identic)   
            # g_loss_id_psnt = F.mse_loss(x_real, x_identic_psnt) 
            # 原本的收斂太快  
            g_loss_id = F.mse_loss(x_real, x_identic.squeeze())   
            g_loss_id_psnt = F.mse_loss(x_real, x_identic_psnt.squeeze())   
            
            # Code semantic loss.
            code_reconst = self.G(x_identic_psnt, emb_org, None)
            g_loss_cd = F.l1_loss(code_real, code_reconst)


            # Backward and optimize.
            g_loss = g_loss_id + g_loss_id_psnt + self.lambda_cd * g_loss_cd
            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()

            # Logging.
            loss = {}
            loss['G/loss_id'] = g_loss_id.item()
            loss['G/loss_id_psnt'] = g_loss_id_psnt.item()
            loss['G/loss_cd'] = g_loss_cd.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag in keys:
                    log += ", {}: {:.4f}".format(tag, loss[tag])
                print(log)
                

    
# 這是一個聲音轉換模型的訓練程式，使用GAN架構訓練生成器模型（Generator），使其能夠將一個語音信號轉換成另一個語音信號，實現語音風格轉換等功能。在訓練過程中，主要包含以下步驟：

# 預處理輸入數據
# 訓練生成器模型
# 計算損失函數
# 訓練過程中打印日誌
# 其中，訓練生成器模型的過程包括：

# 計算身份映射損失（Identity Mapping Loss）：使用相同語音語者的語音片段作為輸入和輸出，並最小化生成的語音信號和原始語音信號之間的均方誤差。
# 計算身份映射與語者相關的損失（Identity Mapping Loss with Speaker Embedding）：使用相同語音語者的語音片段作為輸入和輸出，同時加入語音語者的語音特徵向量，並最小化生成的語音信號和原始語音信號之間的均方誤差。
# 計算語義代碼損失（Code Semantic Loss）：使用語音特徵向量來表示語音的語義特徵，最小化生成的語音信號和原始語音信號的語義特徵之間的L1誤差。
# 訓練過程中，打印日誌，包括已經過的時間、當前迭代次數以及各個損失函數的值。
    

    