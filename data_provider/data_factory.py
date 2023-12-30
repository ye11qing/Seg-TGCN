from data_provider.data_loader import Dataset_sz_min, Dataset_pred
from torch.utils.data import DataLoader

data_dict = {
    'sz': Dataset_sz_min,
}

def data_provider(args, flag):
    Data = data_dict[args.data]
    
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        data = Dataset_pred
    elif flag == 'train':
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.pred_len],
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader   
    