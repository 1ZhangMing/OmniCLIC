""" Example for OmniCLIC classification
"""
#


import time

from train_test import train_test
from train_test import prepare_trte_data


if __name__ == "__main__":
    # 记录程序开始时间
    start_time = time.time()
    
    data_folder = 'KIRP'
    #data_folder = 'BRCA'
    #data_folder ='OV'
    #data_folder = 'GBM'

    tau=0.35
    alpha=0.2
    
    view_list = [1,2,3]#三个组学
    num_epoch_pre = 0
    num_epoch = 4000#训练模型的轮数
    lr_e_pretrain = 1e-3
    lr_e = 1e-3
    lr_c = 1e-3
    dr=0.15
    if data_folder in[ 'KIRP' ]:
        num_class = 2
        no=0.01
    if data_folder in['BRCA'  , 'GBM']:
        num_class = 5
        no=0.25
    if data_folder== 'OV':
        num_class = 4
        no=0.25
    
    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list)
    print(data_tr_list[0].shape)
    print(data_tr_list[1].shape)
    print(data_tr_list[2].shape)
    start_experiment_time = time.time()  # 记录当前实验的开始时间
    train_test(data_folder, view_list, num_class,
                               lr_e_pretrain, lr_e, lr_c,
                               num_epoch_pre, num_epoch,0.5,data_tr_list, data_trte_list, trte_idx,labels_trte,tau,alpha,dr=dr,noise=no)
    
    # 计算耗时
    end_time = time.time()        
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    time_str = f"Elapsed Time: {int(minutes)} minutes {int(seconds)} seconds"
    print(time_str)
