from .uniform import Uniform
from .herding import Herding
from .kcentergreedy import kCenterGreedy
from .forgetting import Forgetting
from common import define_model 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

def start_coreset(args, trainset, testset): 
    print(f"\n{'='*60}")
    print(f"Coreset Selection: {args.selection}")
    print(f"Fraction: {args.fraction} ({int(args.fraction * len(trainset))} samples)")
    print(f"Balance: {args.balance}")
    print(f"{'='*60}\n")
    
    method_dict = {
        'Uniform': Uniform,
        'Herding': Herding,
        'KCenterGreedy': kCenterGreedy,
        'Forgetting': Forgetting
    }
    
    selection_args = {
        'random_seed': args.seed, 
        'balance': args.balance,
    }
    
    if args.selection in ['Herding', 'KCenterGreedy', 'Forgetting']:
        selection_args['epochs'] = args.selection_epochs
        selection_args['specific_model'] = args.eval_model[0]
    
    MethodClass = method_dict[args.selection]
    method = MethodClass(
        dst_train=trainset,
        args=args,
        fraction=args.fraction,
        random_seed=args.seed,
        **selection_args
    )
    
    print("Selecting subset...")
    subset = method.select()
    indices = subset['indices']
    
    print(f"Selected {len(indices)} samples")
    
    if args.balance:
        targets = trainset.targets[indices]
        for c in range(args.num_classes):
            count = (targets == c).sum()
            print(f"  Class {c}: {count} samples")
    
    subset_dataset = Subset(trainset, indices)
    
    subset_loader = DataLoader(
        subset_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
    )
    
    print("\nTraining model on selected coreset...")
    model = define_model(args, args.num_classes, e_model=args.eval_model[0])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=args.eval_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    criterion = nn.CrossEntropyLoss()
    
    num_epochs = args.epochs_eval
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(subset_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(subset_loader)}] '
                      f'Loss: {loss.item():.4f} Acc: {100.*correct/total:.2f}%')
        
        epoch_loss = train_loss / len(subset_loader)
        epoch_acc = 100. * correct / total
        print(f'\n>>> Epoch [{epoch+1}/{num_epochs}] '
              f'Train Loss: {epoch_loss:.4f} Train Acc: {epoch_acc:.2f}%')
        
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            test_acc, test_loss = eval_coreset(args, model, testset)
            print(f'>>> Test Loss: {test_loss:.4f} Test Acc: {test_acc:.2f}%\n')
    
    print(f"\nTraining completed!")
    
    return model, indices, subset_dataset


def eval_coreset(args, model, testset): 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Create test loader
    test_loader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    
    criterion = nn.CrossEntropyLoss()
    
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    return accuracy, avg_loss