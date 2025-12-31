import argparse
import os
import time
from scipy.io import loadmat, savemat
import numpy as np
import logging
import datetime

import torch
from torch import optim
from torch.utils.data import DataLoader

import network
import loaders
from utils import create_synthetic_forward_matrix


def custom_collate_fn(batch):
    """Custom collate function to handle variable-length valid_labels"""
    # Separate the valid_labels which have variable lengths
    valid_labels = [item['valid_labels'] for item in batch]
    
    # Remove valid_labels from items for default collation
    batch_no_valid_labels = []
    for item in batch:
        new_item = {k: v for k, v in item.items() if k != 'valid_labels'}
        batch_no_valid_labels.append(new_item)
    
    # Use default collation for the rest
    collated = torch.utils.data.default_collate(batch_no_valid_labels)
    
    # Add back valid_labels as a list
    collated['valid_labels'] = valid_labels
    
    return collated


def main():
    start_time = time.time()
    # parse the input
    parser = argparse.ArgumentParser(description='DeepSIF Transformer Model')
    parser.add_argument('--save', type=int, default=1, help='save each epoch or not')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--device', default='cuda:0', type=str, help='device running the code')
    parser.add_argument('--arch', default='TransformerTemporalInverseNet', type=str, help='network architecture class')
    parser.add_argument('--dat', default='LabeledDatasetLoader', type=str, help='data loader')
    parser.add_argument('--train', default='labeled_dataset', type=str, help='train dataset directory')
    parser.add_argument('--test', default='labeled_dataset', type=str, help='test dataset directory')
    parser.add_argument('--model_id', default='1', type=str, help='model id')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--resume', default='', type=str, help='epoch id to resume')
    parser.add_argument('--epoch', default=50, type=int, help='total number of epoch')
    parser.add_argument('--fwd', default='', type=str, help='forward matrix to use (optional)')
    parser.add_argument('--transformer_layers', default=3, type=int, help='number of transformer layers')
    parser.add_argument('--d_model', default=512, type=int, help='transformer model dimension')
    parser.add_argument('--nhead', default=8, type=int, help='number of attention heads')
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate')
    parser.add_argument('--info', default='', type=str, help='other information regarding this model')
    parser.add_argument('--train_split', default=0.8, type=float, help='training split ratio')
    args = parser.parse_args()

    # ======================= PREPARE PARAMETERS =====================================================================================================
    use_cuda = torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else "cpu")
    print(f"Using device: {device}")

    result_root = f'model_result/{args.model_id}_transformer_model'
    if not os.path.exists(result_root):
        os.makedirs(result_root)
    
    # Forward matrix (optional - data is pre-processed)
    fwd = None
    if args.fwd and os.path.exists(f'anatomy/{args.fwd}'):
        fwd = loadmat(f'anatomy/{args.fwd}')['fwd']
        print(f"Loaded forward matrix from anatomy/{args.fwd} with shape {fwd.shape}")
    else:
        print("No forward matrix specified - using pre-processed labeled dataset")

    # Define logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(result_root + f'/outputs_{args.arch}.log')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    logger.info("============================= {} ====================================".format(datetime.datetime.now()))
    logger.info("Training data is {}, and testing data is {}".format(args.train, args.test))
    # Save every parameters in args
    for v in args.__dict__:
        if v not in ['workers', 'train', 'test']:
            logger.info('{} is {}'.format(v, args.__dict__[v]))

    # ================================== LOAD DATA ===================================================================================================
    # Get all data files and split into train/test
    import glob
    all_files = glob.glob(os.path.join(args.train, "*.mat"))
    all_files.sort()
    
    n_train = int(len(all_files) * args.train_split)
    train_files = all_files[:n_train]
    test_files = all_files[n_train:]
    
    print(f"Total files: {len(all_files)}, Train: {len(train_files)}, Test: {len(test_files)}")
    
    # Create data loaders with train/test split
    train_data = loaders.__dict__[args.dat](
        args.train, 
        fwd=fwd,
        args_params={'dataset_len': len(train_files)}
    )
    train_data.file_list = train_files
    train_loader = DataLoader(
        train_data, 
        batch_size=args.batch_size, 
        num_workers=args.workers, 
        pin_memory=True, 
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    
    test_data = loaders.__dict__[args.dat](
        args.test, 
        fwd=fwd, 
        args_params={'dataset_len': len(test_files)}
    )
    test_data.file_list = test_files
    test_loader = DataLoader(
        test_data, 
        batch_size=args.batch_size, 
        num_workers=args.workers, 
        pin_memory=True, 
        shuffle=False,
        collate_fn=custom_collate_fn
    )

    # ================================== CREATE MODEL ================================================================================================
    
    if args.arch == 'TransformerTemporalInverseNet':
        net = network.__dict__[args.arch](
            num_sensor=75, 
            num_source=994, 
            transformer_layers=args.transformer_layers,
            spatial_model=network.MLPSpatialFilter,
            temporal_model=network.TransformerTemporalFilter,
            spatial_output='value_activation', 
            temporal_output='transformer', 
            spatial_activation='ELU', 
            temporal_activation='ELU',
            temporal_input_size=500,
            d_model=args.d_model,
            nhead=args.nhead,
            dropout=args.dropout
        ).to(device)
    else:
        # Fallback to original LSTM model
        net = network.__dict__[args.arch](
            num_sensor=75, 
            num_source=994, 
            rnn_layer=args.transformer_layers,
            spatial_model=network.MLPSpatialFilter,
            temporal_model=network.TemporalFilter,
            spatial_output='value_activation', 
            temporal_output='rnn', 
            spatial_activation='ELU', 
            temporal_activation='ELU',
            temporal_input_size=500
        ).to(device)
    
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-6)
    criterion = torch.nn.MSELoss(reduction='sum')

    args.start_epoch = 0
    best_result = np.Inf
    train_loss = []
    test_loss = []

    # =============================== RESUME =========================================================================================================
    if args.resume and os.path.exists(args.resume):
        print(f"=> Load checkpoint {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        args.start_epoch = checkpoint['epoch']
        best_result = checkpoint['best_result']
        net.load_state_dict(checkpoint['state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"=> Loaded checkpoint epoch {args.start_epoch}, current best result: {best_result}")
        
        # Load training history if available
        history_file = os.path.join(result_root, 'train_test_error.mat')
        if os.path.exists(history_file):
            tte = loadmat(history_file)
            train_loss.extend(tte['train_loss'][0][:args.start_epoch + 1].tolist())
            test_loss.extend(tte['test_loss'][0][:args.start_epoch + 1].tolist())

    print('Number of parameters:', net.count_parameters())
    print('Prepare time:', time.time() - start_time)
    logger.info(f'Number of parameters: {net.count_parameters()}')

    # =============================== TRAINING =======================================================================================================
    for epoch in range(args.start_epoch + 1, args.epoch + 1):

        # train for one epoch
        train_lss_all = train(train_loader, net, criterion, optimizer, {'device': device, 'logger': logger})
        # evaluate on validation set
        test_lss_all = validate(test_loader, net, criterion, {'device': device})
        
        train_loss.append(np.mean(train_lss_all))
        test_loss.append(np.mean(test_lss_all))

        print_s = 'Epoch {}: Time:{:6.2f}, '.format(epoch, time.time() - start_time) + \
                  'Train Loss:{:06.5f}'.format(train_loss[-1]) + ', Test Loss:{:06.5f}'.format(test_loss[-1])
        logger.info(print_s)
        print(print_s)
        
        is_best = test_loss[-1] < best_result
        best_result = min(test_loss[-1], best_result)
        
        if is_best:
            torch.save({
                'epoch': epoch, 
                'arch': args.arch, 
                'state_dict': net.state_dict(), 
                'best_result': best_result, 
                'lr': args.lr, 
                'info': args.info,
                'train': args.train, 
                'test': args.test, 
                'attribute_list': net.attribute_list, 
                'optimizer': optimizer.state_dict()
            }, result_root + '/model_best.pth')
            
        if args.save:
            # save checkpoint
            torch.save({
                'epoch': epoch, 
                'arch': args.arch, 
                'state_dict': net.state_dict(), 
                'best_result': best_result, 
                'lr': args.lr, 
                'info': args.info,
                'train': args.train, 
                'test': args.test, 
                'attribute_list': net.attribute_list, 
                'optimizer': optimizer.state_dict()
            }, result_root + f'/epoch_{epoch}')
            
            savemat(result_root + '/train_test_error.mat', {
                'train_loss': train_loss, 
                'test_loss': test_loss
            })
            
    logger.info("Training completed!")
    print("Training completed!")


def train(train_loader, model, criterion, optimizer, args_params):
    device = args_params['device']
    logger = args_params['logger']
    
    model.train()
    train_loss = []
    start_time = time.time()
    
    for batch_idx, sample_batch in enumerate(train_loader):
        # load data
        data = sample_batch['data'].to(device)
        nmm = sample_batch['nmm'].to(device)

        # training process
        optimizer.zero_grad()
        model_output = model(data)
        out = model_output['last']
        loss = criterion(out, nmm)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        
        if (batch_idx + 1) % 100 == 0:
            print_s = f"batch_idx_{batch_idx}_time_{time.time() - start_time:.2f}_train_loss_{train_loss[-1]:.6f}"
            logger.info(print_s)
            print(print_s)
            
    return train_loss


def validate(val_loader, model, criterion, args_params):
    device = args_params['device']
    model.eval()
    val_loss = []
    
    with torch.no_grad():
        for batch_idx, sample_batch in enumerate(val_loader):
            data = sample_batch['data'].to(device)
            nmm = sample_batch['nmm'].to(device)
            model_output = model(data)
            out = model_output['last']
            loss = criterion(out, nmm)
            val_loss.append(loss.item())
            
    return val_loss


if __name__ == '__main__':
    main()
