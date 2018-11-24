import torch
import torchvision
import time
import numpy as np
import argschema
import os
import argparse

from sl_loader import SL_Dataset_Train
#from networks import model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='')
    parser.add_argument('--label_file', type=str, default='', help='file with image names and label probabilities')
    parser.add_argument('--model_prefix', type=str, default='untitled', help='what to call the model')
    parser.add_argument('--model_save_dir', type=str, default='./', help='where to save the model')
    parser.add_argument('--num_iter', type=int, default=100)
    parser.add_argument('--gpu', type=bool, default=True, help='use gpu?')
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=1)
    args = parser.parse_args()
    return args

class TrainNetwork:
    
    def __init__(self, args):
        self.args = args
    
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
                loss = loss_fn(output, label_batch) #todo manage loss
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


        loader = SL_Dataset_Train(
            self.args['image_dir'],
            self.args['label_file'],
            image_transforms=image_transforms
        )

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
    args = get_args()
    model = TrainNetwork(args)
    model.run()