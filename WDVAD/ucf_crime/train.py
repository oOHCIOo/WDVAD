import torch
import numpy as np
import os
from losses import KMXMILL_individual, normal_smooth
from generate_new_labels import calculate_MMD_and_update_labels
import numpy as np
import options
from model import Model_single
from video_dataset_anomaly_balance_uni_sample import dataset
from torch.utils.data import DataLoader
import torch.nn.init as torch_init
import torch.nn as nn
import torch.nn.functional as F

def load_model_from_ckpt(model, pretrained_ckpt):
    """
    Load model weights from a given checkpoint file.

    Args:
        model: Model to load weights into.
        pretrained_ckpt: Path to pretrained checkpoint file.

    Returns:
        Model with loaded weights.
    """
    # Load checkpoint
    checkpoint = torch.load(pretrained_ckpt)

    # Load model weights
    model.load_state_dict(checkpoint)
    print(f"Model weights loaded from {pretrained_ckpt}")

    return model

def train(round_num, model, save_path, dataset_name, cid, args, epochs, device, pretrained_ckpt, video_labels_path, frame_labels_path):
    """
    Train model and update weights using federated learning.

    Args:
        round_num: Current federated learning round.
        pretrained_ckpt: Path to pretrained model checkpoint for this client.
    """

    # Create result directory for first round
    if round_num == 1:
        if not os.path.exists(os.path.join(args.cur_path, f'result_{dataset_name}')):
            os.makedirs(os.path.join(args.cur_path, f'result_{dataset_name}'))
        
        # Save training arguments
        with open(os.path.join(args.cur_path, f'result_{dataset_name}', f'result_cid_{cid}.txt'), 'w') as f:
            for key, value in vars(args).items():
                f.write(f'{key}:{value}\n')
        
        # Load pretrained model without training
        traindataset = dataset(args=args, dataset_name=dataset_name, train=True)
        print(f"Skipping training for round {round_num}")
        model = load_model_from_ckpt(model, pretrained_ckpt)
        return len(traindataset)
    
    itr = 0

    # Handle teacher clients (no training)
    if cid in args.t_id:
        print(f"Client {cid} will not update model parameters, only aggregation.")
        optimizer = None 
        new_video_filename, new_frame_filename = None, None
    else:
        print(f"Training on client {cid}")

        # Prepare training dataset
        testlist = f'{dataset_name}_train.txt'
        traindataset = dataset(args=args, new_video_filename=None, dataset_name=dataset_name, train=False, testlist=testlist)
        train_loader = DataLoader(
            dataset=traindataset, batch_size=args.batch_size, pin_memory=True,
            num_workers=0, shuffle=False
        )
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Generate new labels starting from round 2
        if round_num >= 2:
            # Create model instance for original weights
            original_model = Model_single(n_feature=2048).to(device)
            original_model = load_model_from_ckpt(original_model, pretrained_ckpt)

            # Calculate MMD and update labels
            new_video_filename, new_frame_filename = calculate_MMD_and_update_labels(
                args=args,
                train_loader=train_loader,
                model=model,
                original_model=original_model,
                device=device,
                video_labels_path=video_labels_path,
                frame_labels_path=frame_labels_path,
                current_round=round_num,
                dataset_name=dataset_name
            )
        else:
            new_video_filename, new_frame_filename = None, None

    # Handle teacher clients (return dataset size only)
    if cid in args.t_id:
        traindataset_new = dataset(args=args, new_video_filename=new_video_filename, dataset_name=dataset_name, train=True)
        train_loader_new = DataLoader(
            dataset=traindataset_new, batch_size=args.batch_size, pin_memory=True,
            num_workers=0, shuffle=True  # Use 0 on Windows to avoid multiprocessing issues
        )
        return len(train_loader_new.dataset)
    
    # Prepare training dataset with new labels
    traindataset_new = dataset(args=args, new_video_filename=new_video_filename, dataset_name=dataset_name, train=True)
    print(f'Current training dataset size: {len(traindataset_new)}')
    train_loader_new = DataLoader(
        dataset=traindataset_new, batch_size=args.batch_size, pin_memory=True,
        num_workers=0, shuffle=True  # Use 0 on Windows to avoid multiprocessing issues
    )
    
    # Model initialization strategy
    if round_num == 2:
        # Use original pretrained model for round 2
        original_model = Model_single(n_feature=2048).to(device)
        original_model = load_model_from_ckpt(original_model, pretrained_ckpt)
        model = original_model
        print("Round 2: Training with original model")
    else:
        print(f"Round {round_num}: Training with aggregated model")
    
    # Training loop
    for epoch in range(epochs):
        for i, data in enumerate(train_loader_new):
            itr += 1
            [anomaly_features, normaly_features], [anomaly_label, normaly_label], stastics_data = data
            
            # Prepare features and labels
            features = torch.cat((anomaly_features.squeeze(0), normaly_features.squeeze(0)), dim=0)
            videolabels = torch.cat((anomaly_label.squeeze(0), normaly_label.squeeze(0)), dim=0)
            seq_len = torch.sum(torch.max(features.abs(), dim=2)[0] > 0, dim=1).numpy()
            features = features[:, :np.max(seq_len), :]
            features = features.float().to(device)
            videolabels = videolabels.float().to(device)

            # Forward pass
            element_logits = model(features)
            
            # Calculate losses
            weights = args.Lambda.split('_')
            m_loss = KMXMILL_individual(
                element_logits=element_logits,
                seq_len=seq_len,
                labels=videolabels,
                device=device,
                loss_type='CE',
                args=args
            )
            n_loss = normal_smooth(
                element_logits=element_logits,
                labels=videolabels,
                device=device
            )

            # Combine losses
            total_loss = float(weights[0]) * m_loss + float(weights[1]) * n_loss
            
            # Log training progress
            if itr % 20 == 0 and itr != 0:
                print(f'Iteration: {itr}, Loss: {total_loss.data.cpu().detach().numpy()}')

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    # Return training dataset size
    return len(train_loader_new.dataset)