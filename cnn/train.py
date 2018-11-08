import torch
import torchvision
import time
import numpy as np
import argschema
import os

#from networks import model
#from networks import Dataset

class TrainNetwork:
    
    def __init__(self):
        self.args = {
                'image_dir': 'images/',
                'info_file': 'info.txt',
                'model_prefix': 'model',
                'model_save_dir': './',
                'num_iter': 100,
                'gpu': 'True',
                'learning_rate': 0.1,
                'epochs': 1
                }
    
    def train(self, model, dataset, optimizer, loss_fn, epoch):
        for it in range(self.args['num_iter']):
            print('Iteration {} of {}'.format(it+1, self.args['num_iter']))

            model.train()

            startt = time.time()
            for batch_idx, (image_batch, label_batch) in enumerate(dataset):
                image_batch.unsqueeze_(0)
                if self.args['gpu']: 
                    image_batch = image_batch.cuda()
                    label_batch = label_batch.cuda()

                optimizer.zero_grad()

                output_batch = model(image_batch)
                loss = loss_fn(output_batch, torch.max(label_batch,1)[1])
                loss.backward()
                optimizer.step()

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage Loss: {:.6f}'.format(epoch+1,(batch_idx+1) * len(image_batch), len(dataset), 100.*len(image_batch) * (batch_idx+1) / len(dataset), loss.data[0]))
                
            endt = time.time()
            print('Iteration {0:.0f} finished in {1:.4f} seconds; {2:.4f} seconds per batch'.format((it+1), (endt-startt), ((endt-startt)/(len(dataset)))))

    def run(self):
        model = #TODO
        
        if self.args['gpu']:
            gpu_count = torch.cuda.device_count()
            assert(gpu_count > 0)
            if gpu_count > 1: model = torch.nn.DataParallel(model)
            model.cuda()
         
        #data normalization pre processing if necessary
        #grayscale = torchvision.transforms.Grayscale()
        #resize = torchvision.transforms.Resize(512)
        #horizontal = torchvision.transforms.RandomHorizontalFlip(p=0.5)
        #vertical = torchvision.transforms.RandomVerticalFlip(p=0.5)
        #rotate = torchvision.transforms.RandomRotation(180)
        #color_jitter = torchvision.transforms.ColorJitter(brightness=0.3,contrast=0.3,saturation=0,hue=0)
        image_transforms = []

        #TODO
        #loader = Dataset(
        #    self.args['image_dir'],
        #    self.args['info_file'],
        #    image_transforms=image_transforms
        #)

        dataset = torch.utils.data.DataLoader(
            loader,
            batch_size=self.args['batch_size'],
            shuffle=False
        )

        optimizer = torch.optim.SGD(model.parameters(), lr=self.args['learning_rate'])
        loss_fn = torch.nn.CrossEntropyLoss()

        for ep in range(self.args['epochs']):
            self.train(model, dataset, optimizer, loss_fn, ep)
            
            torch.save(
                model.state_dict(),
                os.path.join(self.args['model_save_dir'], '{}_ep_{}.pt'.format(self.args['model_prefix'], ep+1))
            )

if __name__=='__main__':
    mod = TrainNetwork(input_data=example)
    mod.run()