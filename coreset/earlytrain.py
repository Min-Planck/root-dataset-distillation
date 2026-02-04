from .coresetmethod import CoresetMethod
import torch
from torch import nn
import numpy as np
from common import define_model

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class EarlyTrain(CoresetMethod):


    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, 
                 specific_model=None, fraction_pretrain=1.0, dst_test=None, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed)
        self.epochs = epochs
        self.n_train = len(dst_train)
        self.coreset_size = round(self.n_train * fraction)
        self.specific_model = specific_model
        self.dst_test = dst_test
        
        if fraction_pretrain <= 0. or fraction_pretrain > 1.:
            raise ValueError("Illegal pretrain fraction value. Must be in (0, 1]")
        self.fraction_pretrain = fraction_pretrain
        
        self.n_pretrain_size = round(self.n_train * self.fraction_pretrain)

    def train(self, epoch, list_of_train_idx, **kwargs):
        """Train model for one epoch"""
        self.before_train()
        self.model.train()

        print(f'\n=> Training Epoch #{epoch}')
        
        trainset_permutation_inds = np.random.permutation(list_of_train_idx)
        batch_sampler = torch.utils.data.BatchSampler(
            trainset_permutation_inds, 
            batch_size=self.args.selection_batch,
            drop_last=False
        )
        trainset_permutation_inds = list(batch_sampler)

        train_loader = torch.utils.data.DataLoader(
            self.dst_train,  
            shuffle=False, 
            batch_sampler=batch_sampler,
            pin_memory=True
        )

        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            self.model_optimizer.zero_grad()
            

            outputs = self.model.forward_deepcore(inputs)
      
            loss = self.criterion(outputs, targets)

            self.after_loss(outputs, loss, targets, trainset_permutation_inds[i], epoch)

            loss = loss.mean()
            self.while_update(outputs, loss, targets, epoch, i, self.args.selection_batch)
            
            loss.backward()
            self.model_optimizer.step()
            
        return self.finish_train()

    def run(self):
    
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.train_indx = np.arange(self.n_train)

        self.model = define_model(self.args, self.args.num_classes, e_model=self.specific_model)
        self.model = self.model.to(DEVICE)

   
        self.criterion = nn.CrossEntropyLoss().to(DEVICE)

        self.model_optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=self.args.selection_lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay
        )

        self.before_run()

        # Training loop
        for epoch in range(self.epochs):
     
            if self.fraction_pretrain < 1.0:
                list_of_train_idx = np.random.choice(
                    np.arange(self.n_train),  
                    self.n_pretrain_size, 
                    replace=False
                )
            else:

                list_of_train_idx = np.arange(self.n_train)
            
            self.before_epoch()
            self.train(epoch, list_of_train_idx)
            
            if (self.dst_test is not None 
                and hasattr(self.args, 'selection_test_interval')
                and self.args.selection_test_interval > 0 
                and (epoch + 1) % self.args.selection_test_interval == 0):
                self.test(epoch)
            
            self.after_epoch()

        return self.finish_run()

    def test(self, epoch):

        if hasattr(self.model, 'no_grad'):
            self.model.no_grad = True
        self.model.eval()

        if hasattr(self.args, 'selection_test_fraction') and self.args.selection_test_fraction < 1.0:
            test_indices = np.random.choice(
                np.arange(len(self.dst_test)),
                round(len(self.dst_test) * self.args.selection_test_fraction),
                replace=False
            )
            test_dataset = torch.utils.data.Subset(self.dst_test, test_indices)
        else:
            test_dataset = self.dst_test
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.args.selection_batch, 
            shuffle=False,
            num_workers=self.args.workers if hasattr(self.args, 'workers') else 0,
            pin_memory=True
        )
        
        correct = 0
        total = 0
        test_loss = 0.0

        print(f'\n=> Testing Epoch #{epoch}')

        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(test_loader):
                output = self.model.forward_deepcore(input.to(DEVICE))
                loss = self.criterion(output, target.to(DEVICE))
                
                test_loss += loss.item()
                predicted = torch.max(output.data, 1).indices.cpu()
                correct += predicted.eq(target).sum().item()
                total += target.size(0)

                if hasattr(self.args, 'print_freq') and batch_idx % self.args.print_freq == 0:
                    print(f'| Test Epoch [{epoch}/{self.epochs}] '
                          f'Iter[{batch_idx + 1}/{len(test_loader)}] '
                          f'Loss: {loss.item():.4f} Acc: {100. * correct / total:.3f}%')

        avg_loss = test_loss / len(test_loader)
        accuracy = 100. * correct / total
        print(f'>>> Test Result: Loss: {avg_loss:.4f} Acc: {accuracy:.2f}%')

        if hasattr(self.model, 'no_grad'):
            self.model.no_grad = False

    def num_classes_mismatch(self):
        pass

    def before_train(self):
        pass

    def after_loss(self, outputs, loss, targets, batch_inds, epoch):
        pass

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        pass

    def finish_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def before_run(self):
        pass

    def finish_run(self):
        pass

    def select(self, **kwargs):
        selection_result = self.run()
        return selection_result
