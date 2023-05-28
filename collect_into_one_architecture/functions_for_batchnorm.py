import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from matplotlib.pyplot import imshow, imsave
import matplotlib.pyplot as plt

import numpy as np
import datetime
import os, sys
import time
import pandas as pd
from sklearn.metrics import classification_report

from torch.nn.functional import interpolate
from torchvision.models import inception_v3
from scipy.linalg import sqrtm

from matplotlib.pyplot import imshow, imsave
from plotly.subplots import make_subplots
import plotly.express as px

import torch
import torchvision.utils as vutils
from torch.autograd import Variable
import shutil
import os
from PIL import Image

import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import inception_v3
from scipy.linalg import sqrtm

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## architecture
class Discriminator(nn.Module):
    """
        Convolutional Discriminator for MNIST
    """
    def __init__(self, in_channel=1, input_size=784, condition_size=10, num_classes=1):
        super(Discriminator, self).__init__()
        self.transform = nn.Sequential(
            nn.Linear(input_size+condition_size, 784),
            nn.LeakyReLU(0.2),
        )
        self.conv = nn.Sequential(
            # 28 -> 14
            nn.Conv2d(in_channel, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            # 14 -> 7
            nn.Conv2d(512, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # 7 -> 4
            nn.Conv2d(256, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(4),
        )
        self.fc = nn.Sequential(
            # reshape input, 128 -> 1
            nn.Linear(128, 1)#,
            #nn.Sigmoid()
        )
    
    def forward(self, x, c=None):
        # x: (N, 1, 28, 28), c: (N, 10)
        x, c = x.view(x.size(0), -1), c.float() # may not need
        v = torch.cat((x, c), 1) # v: (N, 794)
        y_ = self.transform(v) # (N, 784)
        y_ = y_.view(y_.shape[0], 1, 28, 28) # (N, 1, 28, 28)
        y_ = self.conv(y_)
        y_ = y_.view(y_.size(0), -1)
        y_ = self.fc(y_)
        return y_


class Generator(nn.Module):
    """
        Convolutional Generator for MNIST
    """
    def __init__(self, input_size=100, condition_size=10):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size+condition_size, 4*4*512),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            # input: 4 by 4, output: 7 by 7
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # input: 7 by 7, output: 14 by 14
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # input: 14 by 14, output: 28 by 28
            nn.ConvTranspose2d(128, 1, 4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )
        
    def forward(self, x, c):
        # x: (N, 100), c: (N, 10)
        x, c = x.view(x.size(0), -1), c.float() # may not need
        v = torch.cat((x, c), 1) # v: (N, 110)
        y_ = self.fc(v)
        y_ = y_.view(y_.size(0), 512, 4, 4)
        y_ = self.conv(y_) # (N, 28, 28)
        return y_
    


def to_onehot(x, num_classes=10):
    assert isinstance(x, int) or isinstance(x, (torch.LongTensor, torch.cuda.LongTensor))
    if isinstance(x, int):
        c = torch.zeros(1, num_classes).long()
        c[0][x] = 1
    else:
        x = x.cpu()
        c = torch.LongTensor(x.size(0), num_classes)
        c.zero_()
        c.scatter_(1, x, 1) # dim, index, src value
    return c


def get_sample_image(G, n_noise=100):
    """
        save sample 100 images
    """
    img = np.zeros([280, 280])
    for j in range(10):
        c = torch.zeros([10, 10]).to(DEVICE)
        c[:, j] = 1
        z = torch.randn(10, n_noise).to(DEVICE)
        y_hat = G(z,c).view(10, 28, 28)
        result = y_hat.cpu().data.numpy()
        img[j*28:(j+1)*28] = np.concatenate([x for x in result], axis=-1)
    return img


def download_data(batch_size):
  transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5],
                                  std=[0.5])]
  )

  mnist = datasets.MNIST(root='../data/', train=True, transform=transform, download=True)
  data_loader = DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True, drop_last=True)
  return data_loader



# # load model
def load_model(path, n_noise):

  D = Discriminator().to(DEVICE)
  G = Generator(n_noise).to(DEVICE)
#   D_opt = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=(0., 0.9)) ## may change
#   G_opt = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(0., 0.9)) ## may change


  checkpoint = torch.load(f"{path}/model.tar")

  G.load_state_dict(checkpoint['G_state_dict'])
  D.load_state_dict(checkpoint['D_state_dict'])

  # D_opt.load_state_dict(checkpoint['g_optimizer_state_dict'])
  # D_opt.load_state_dict(checkpoint['d_optimizer_state_dict'])
  # eval(G)

  

  D_loss = checkpoint['D_loss']
  G_loss = checkpoint['G_loss']
  return G_loss, D_loss

def calculate_f1_score(labels, predictions, threshold):
    tp = ((labels == 1) & (predictions >= threshold)).sum().item()
    fp = ((labels == 0) & (predictions >= threshold)).sum().item()
    fn = ((labels == 1) & (predictions < threshold)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)

    return precision, recall, f1_score
    
    

def train_WGAN(model_name, path_model, path_img, learning_rate, batch_size, n_critic, max_epoch, clip_value, n_noise, random_seed, threshold, beta):
    step = 0
    g_step = 0
    if random_seed:
      torch.manual_seed(42)

    D = Discriminator().to(DEVICE)
    G = Generator(n_noise).to(DEVICE)

    D_opt = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=(0., beta))
    G_opt = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(0., beta))
    data_loader = download_data(batch_size)
    print(f"Starting the training {model_name}")
    print(f"Batch size = {batch_size}, learning rate = {learning_rate}")
    print(f"Beta = {beta}")
    print(f"N critic = {n_critic}, N epochs = {max_epoch}, Random Seed = {random_seed}, Threshold = {threshold}")

    g_loss = []
    d_loss = []
    g_loss_per_step = []
    d_loss_per_step = []
    steps = []
    epochs = []
    avg_f1 = []
    avg_recall = []
    avg_precision = []
    for epoch in range(max_epoch):
      precision_scores = []
      recall_scores = []
      f1_scores = []
      for idx, (images, labels) in enumerate(data_loader):
                    
          # Training Discriminator
          x = images.to(DEVICE)
          y = labels.view(batch_size, 1)
          y = to_onehot(y).to(DEVICE)
          x_outputs = D(x, y)

          z = torch.randn(batch_size, n_noise).to(DEVICE)
          z_outputs = D(G(z, y), y)
          D_x_loss = torch.mean(x_outputs)
          D_z_loss = torch.mean(z_outputs)
          D_loss = D_z_loss - D_x_loss
          
          D.zero_grad()
          D_loss.backward()
          D_opt.step()
                      
          if step % n_critic == 0:
              g_step += 1
              # Training Generator
              z = torch.randn(batch_size, n_noise).to(DEVICE)
              z_outputs = D(G(z, y), y)
              G_loss = -torch.mean(z_outputs)

              D.zero_grad()
              G.zero_grad()
              G_loss.backward()
              G_opt.step()

          with torch.no_grad():
            fake_images = G(torch.randn(batch_size, n_noise).to(DEVICE), y).detach()
            D.eval()
            predictions_real = D(images.to(DEVICE), y).cpu().numpy()
            predictions_fake = D(fake_images, y).cpu().numpy()
            D.train()
            
            labels_real = np.ones((batch_size, 1))
            labels_fake = np.zeros((batch_size, 1))
            labels = np.concatenate([labels_real, labels_fake], axis=0)
            predictions = np.concatenate([predictions_real, predictions_fake], axis=0)
            precision, recall, f1_score = calculate_f1_score(labels, predictions, threshold)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1_score)
 
          step += 1
          if step % 50 == 0:
              g_loss_per_step.append(G_loss.item())
              d_loss_per_step.append(D_loss.item())
              steps.append(step)
              print('Step: {}, D Loss: {}, G Loss: {}'.format(step, D_loss.item(), G_loss.item()))

            
      print('Epoch: {}/{}, D Loss: {}, G Loss: {}'.format(epoch, max_epoch, D_loss.item(), G_loss.item()))
      print(f"Epoch {epoch}: Precision = {sum(precision_scores) / len(precision_scores)}, Recall = {sum(recall_scores) / len(recall_scores)}, F1 Score = {sum(f1_scores) / len(f1_scores)}")
      avg_f1.append(sum(f1_scores) / len(f1_scores))
      avg_recall.append(sum(recall_scores) / len(recall_scores))
      avg_precision.append(sum(precision_scores) / len(precision_scores))
      epochs.append(epoch)
      g_loss.append(G_loss.item())
      d_loss.append(D_loss.item())

      G.eval()
      img = get_sample_image(G, n_noise)
      imsave(f'{path_img}/{model_name}_epoch{str(epoch).zfill(3)}.jpg', img, cmap='gray')
      G.train()

    G.eval()
    img = get_sample_image(G, n_noise)
    plt.figure(figsize=(12, 300/400*12))
    imshow(get_sample_image(G, n_noise), cmap='gray')
    imsave(f'{path_img}/{model_name}_last.jpg', img, cmap='gray')
    state_dict = {
                'epoch': epoch,
                'D_loss': D_loss,
                'G_loss': G_loss,
                'D_state_dict': D.state_dict(),
                'G_state_dict': G.state_dict()}
    print("Finishing the training with losses: D loss = ", D_loss.item(), " G loss = ", G_loss.item())
    # save the state dictionary to a file
    torch.save(state_dict, f'{path_model}/model.tar')
    results_loss = pd.DataFrame()
    results_loss["epoch"] = epochs
    results_loss["G_loss"] = g_loss
    results_loss["D_loss"] = d_loss
    results_loss["F1_score"] = avg_f1
    results_loss["Precision"] = avg_precision
    results_loss["Recall"] = avg_recall
    results_loss.to_csv(f'{path_model}/{model_name}_G_D_loss_scores_per_epoch.csv')
    print(f"loss values are saved in {path_model}")
    
    results_per_step = pd.DataFrame()
    results_per_step["step"] = steps
    results_per_step["G_loss_step"] = g_loss_per_step
    results_per_step["D_loss_step"] = d_loss_per_step
    results_per_step.to_csv(f'{path_model}/{model_name}_G_D_loss_scores_per_step.csv')
    print(f"loss values PER STEP are saved in {path_model}_G_D_loss_scores_per_step.csv")



def train_WGAN_GP(model_name, path_model, path_img, learning_rate, batch_size, n_critic, max_epoch, n_noise, random_seed, threshold, beta, penalty=10):
  step=0
  if random_seed:
    torch.manual_seed(42)
#   p_coeff = 10 # lambda 
  p_coeff = penalty
  D = Discriminator().to(DEVICE)
  G = Generator(n_noise).to(DEVICE)

  D_labels = torch.ones([batch_size, 1]).to(DEVICE) # Discriminator Label to real
  D_fakes = torch.zeros([batch_size, 1]).to(DEVICE) # Discriminator Label to fake

  D_opt = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=(0., beta))
  G_opt = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(0., beta))

  data_loader = download_data(batch_size)
  print(f"Starting the training {model_name}")
  print(f"Batch size = {batch_size}, learning rate = {learning_rate}")
  print(f"Beta = {beta}")
  print(f"Gradient Penalty = {p_coeff}")
  print(f"N critic = {n_critic}, N epochs = {max_epoch}, Random Seed = {random_seed}, Threshold = {threshold}")

  g_loss = []
  d_loss = []
  g_loss_per_step = []
  d_loss_per_step = []
  steps = []
  epochs = []
  avg_f1 = []
  avg_recall = []
  avg_precision = []
  for epoch in range(max_epoch):
    precision_scores = []
    recall_scores = []
    f1_scores = []
    for idx, (images, labels) in enumerate(data_loader):
        D.zero_grad()
        ## Training Discriminator
        # Real data
        x = images.to(DEVICE)
        y = labels.view(batch_size, 1)
        y = to_onehot(y).to(DEVICE)
        
        # Sampling
        z = torch.randn(batch_size, n_noise).to(DEVICE)
        x_fake = G(z, y)
                
        # Gradient Penalty (e.g. gradients w.r.t x_penalty)
        eps = torch.rand(batch_size, 1, 1, 1).to(DEVICE) # x shape: (64, 1, 28, 28)
        x_penalty = eps*x + (1-eps)*x_fake
        x_penalty = x_penalty.view(x_penalty.size(0), -1)
        p_outputs = D(x_penalty, y)
        xp_grad = autograd.grad(outputs=p_outputs, inputs=x_penalty, grad_outputs=D_labels,
                                create_graph=True, retain_graph=True, only_inputs=True)
        grad_penalty = p_coeff * torch.mean(torch.pow(torch.norm(xp_grad[0], 2, 1) - 1, 2))
        
        # Wasserstein loss
        x_outputs = D(x, y)
        z_outputs = D(x_fake, y)
        D_x_loss = torch.mean(x_outputs)
        D_z_loss = torch.mean(z_outputs)
        D_loss = D_z_loss - D_x_loss + grad_penalty
        
        D_loss.backward()
        D_opt.step()        
        if step % n_critic == 0:
            D.zero_grad()
            G.zero_grad()
            # Training Generator
            z = torch.randn(batch_size, n_noise).to(DEVICE)
            z_outputs = D(G(z, y), y)
            G_loss = -torch.mean(z_outputs)

            G_loss.backward()
            G_opt.step()


        with torch.no_grad():
          fake_images = G(torch.randn(batch_size, n_noise).to(DEVICE), y).detach()
          D.eval()
          predictions_real = D(images.to(DEVICE), y).cpu().numpy()
          predictions_fake = D(fake_images, y).cpu().numpy()
          D.train()
          
          labels_real = np.ones((batch_size, 1))
          labels_fake = np.zeros((batch_size, 1))
          labels = np.concatenate([labels_real, labels_fake], axis=0)
          predictions = np.concatenate([predictions_real, predictions_fake], axis=0)
          precision, recall, f1_score = calculate_f1_score(labels, predictions, threshold)
          precision_scores.append(precision)
          recall_scores.append(recall)
          f1_scores.append(f1_score)

        step += 1
        if step % 50 == 0:
            g_loss_per_step.append(G_loss.item())
            d_loss_per_step.append(D_loss.item())
            steps.append(step)
            print('Step: {}, D Loss: {}, G Loss: {}'.format(step, D_loss.item(), G_loss.item()))

    print('Epoch: {}/{}, D Loss: {}, G Loss: {}'.format(epoch, max_epoch, D_loss.item(), G_loss.item()))
    print(f"Epoch {epoch}: Precision = {sum(precision_scores) / len(precision_scores)}, Recall = {sum(recall_scores) / len(recall_scores)}, F1 Score = {sum(f1_scores) / len(f1_scores)}")
    avg_f1.append(sum(f1_scores) / len(f1_scores))
    avg_recall.append(sum(recall_scores) / len(recall_scores))
    avg_precision.append(sum(precision_scores) / len(precision_scores))


    epochs.append(epoch)
    g_loss.append(G_loss.item())
    d_loss.append(D_loss.item())
    G.eval()
    img = get_sample_image(G, n_noise)
    imsave(f'{path_img}/{model_name}_epoch{str(epoch).zfill(3)}.jpg', img, cmap='gray')
    G.train()
      
  G.eval()
  img = get_sample_image(G, n_noise)
  plt.figure(figsize=(12, 300/400*12))
  imshow(get_sample_image(G, n_noise), cmap='gray')
  imsave(f'{path_img}/{model_name}_last.jpg', img, cmap='gray')
  state_dict = {
              'epoch': epoch,
              'D_loss': D_loss,
              'G_loss': G_loss,
              'D_state_dict': D.state_dict(),
              'G_state_dict': G.state_dict()}
  print("Finishing the training with losses: D loss = ", D_loss.item(), " G loss = ", G_loss.item())
  # save the state dictionary to a file
  torch.save(state_dict, f'{path_model}/model.tar') 
  results_loss = pd.DataFrame()
  results_loss["epoch"] = epochs
  results_loss["G_loss"] = g_loss
  results_loss["D_loss"] = d_loss

  results_loss["F1_score"] = avg_f1
  results_loss["Precision"] = avg_precision
  results_loss["Recall"] = avg_recall
  results_loss.to_csv(f'{path_model}/{model_name}_G_D_loss_scores_per_epoch.csv')
  print(f"loss values are saved in {path_model}")
  
  results_per_step = pd.DataFrame()
  results_per_step["step"] = steps
  results_per_step["G_loss_step"] = g_loss_per_step
  results_per_step["D_loss_step"] = d_loss_per_step
  results_per_step.to_csv(f'{path_model}/{model_name}_G_D_loss_scores_per_step.csv')
  print(f"loss values PER STEP are saved in {path_model}_G_D_loss_scores_per_step.csv")


def train_Vanilla_GAN(model_name, path_model, path_img, learning_rate, batch_size, n_critic, max_epoch, n_noise, random_seed, threshold, beta):
  step = 0 ## delete this
  if random_seed:
    torch.manual_seed(42)

  criterion = nn.BCELoss()

  D = Discriminator().to(DEVICE)
  G = Generator(n_noise).to(DEVICE)

  D_opt = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=(0., beta)) ## may change
  G_opt = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(0., beta)) ## may change

  D_labels = torch.ones([batch_size, 1]).to(DEVICE) # Discriminator Label to real
  D_fakes = torch.zeros([batch_size, 1]).to(DEVICE) # Discriminator Label to fake


  data_loader = download_data(batch_size)
  print(f"Starting the training {model_name}")
  print(f"Batch size = {batch_size}, learning rate = {learning_rate}")
  print(f"Beta = {beta}")
  print(f"N critic = {n_critic}, N epochs = {max_epoch}, Random Seed = {random_seed}, Threshold = {threshold}")
  epochs = []
  g_loss = []
  d_loss = []
  g_loss_per_step = []
  d_loss_per_step = []
  steps = []
  avg_f1 = []
  avg_recall = []
  avg_precision = []
  for epoch in range(max_epoch):

    precision_scores = []
    recall_scores = []
    f1_scores = []
    for idx, (images, labels) in enumerate(data_loader):
        D.zero_grad()
        ## Training Discriminator
        # Real data
        x = images.to(DEVICE)
        y = labels.view(batch_size, 1)
        y = to_onehot(y).to(DEVICE)
        
        # Sampling
        z = torch.randn(batch_size, n_noise).to(DEVICE)
        x_fake = G(z, y)
        
        # Discriminator Loss
        x_outputs = D(x, y)
        z_outputs = D(x_fake, y)
        D_x_loss = torch.mean(torch.nn.functional.binary_cross_entropy_with_logits(x_outputs, D_labels))
        D_z_loss = torch.mean(torch.nn.functional.binary_cross_entropy_with_logits(z_outputs, D_fakes))
        D_loss = D_x_loss + D_z_loss
        
        D_loss.backward()
        D_opt.step()        
        if step % n_critic == 0:
            D.zero_grad()
            G.zero_grad()
            # Training Generator
            z = torch.randn(batch_size, n_noise).to(DEVICE)
            z_outputs = D(G(z, y), y)
            G_loss = torch.mean(torch.nn.functional.binary_cross_entropy_with_logits(z_outputs, D_labels))

            G_loss.backward()
            G_opt.step()
        

        with torch.no_grad():
          fake_images = G(torch.randn(batch_size, n_noise).to(DEVICE), y).detach()
          D.eval()
          predictions_real = D(images.to(DEVICE), y).cpu().numpy()
          predictions_fake = D(fake_images, y).cpu().numpy()
          D.train()
          
          labels_real = np.ones((batch_size, 1))
          labels_fake = np.zeros((batch_size, 1))
          labels = np.concatenate([labels_real, labels_fake], axis=0)
          predictions = np.concatenate([predictions_real, predictions_fake], axis=0)
          precision, recall, f1_score = calculate_f1_score(labels, predictions, threshold)
          precision_scores.append(precision)
          recall_scores.append(recall)
          f1_scores.append(f1_score)

        step += 1
        if step % 50 == 0:
            g_loss_per_step.append(G_loss.item())
            d_loss_per_step.append(D_loss.item())
            steps.append(step)
            print('Step: {}, D Loss: {}, G Loss: {}'.format(step, D_loss.item(), G_loss.item()))
    print('Epoch: {}/{}, D Loss: {}, G Loss: {}'.format(epoch, max_epoch, D_loss.item(), G_loss.item()))
    
    epochs.append(epoch)
    g_loss.append(G_loss.item())
    d_loss.append(D_loss.item())
    G.eval()
    img = get_sample_image(G, n_noise)
    imsave(f'{path_img}/{model_name}_epoch{str(epoch).zfill(3)}.jpg', img, cmap='gray')
    G.train()
    ### ADDED

    
    print(f"Epoch {epoch}: Precision = {sum(precision_scores) / len(precision_scores)}, Recall = {sum(recall_scores) / len(recall_scores)}, F1 Score = {sum(f1_scores) / len(f1_scores)}")
    avg_f1.append(sum(f1_scores) / len(f1_scores))
    avg_recall.append(sum(recall_scores) / len(recall_scores))
    avg_precision.append(sum(precision_scores) / len(precision_scores))


## TO HERE


  G.eval()
  img = get_sample_image(G, n_noise)
  plt.figure(figsize=(12, 300/400*12))
  imshow(get_sample_image(G, n_noise), cmap='gray')
  imsave(f'{path_img}/{model_name}_last.jpg', img, cmap='gray')
  state_dict = {
              'epoch': epoch,
              'D_loss': D_loss,
              'G_loss': G_loss,
              'D_state_dict': D.state_dict(),
              'G_state_dict': G.state_dict()}
  print("Finishing the training with losses: D loss = ", D_loss.item(), " G loss = ", G_loss.item())
  
  # save the state dictionary to a file
  torch.save(state_dict, f'{path_model}/model.tar') 
  results_loss = pd.DataFrame()
  results_loss["epoch"] = epochs
  results_loss["G_loss"] = g_loss
  results_loss["D_loss"] = d_loss
  results_loss["F1_score"] = avg_f1
  results_loss["Precision"] = avg_precision
  results_loss["Recall"] = avg_recall
  results_loss.to_csv(f'{path_model}/{model_name}_G_D_loss_scores_per_epoch.csv')
  print(f"loss values are saved in {path_model}")
  
  results_per_step = pd.DataFrame()
  results_per_step["step"] = steps
  results_per_step["G_loss_step"] = g_loss_per_step
  results_per_step["D_loss_step"] = d_loss_per_step
  results_per_step.to_csv(f'{path_model}/{model_name}_G_D_loss_scores_per_step.csv')
  print(f"loss values PER STEP are saved in {path_model}_G_D_loss_scores_per_step.csv")






def train_LSGAN(model_name, path_model, path_img, learning_rate, batch_size, n_critic, max_epoch, n_noise, random_seed, threshold, beta):
  step = 0 
  if random_seed:
    torch.manual_seed(42)
  criterion = nn.MSELoss()

  D = Discriminator().to(DEVICE)
  G = Generator(n_noise).to(DEVICE)

  D_opt = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=(0., beta)) ## may change
  G_opt = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(0., beta)) ## may change

  D_labels = torch.ones([batch_size, 1]).to(DEVICE) # Discriminator Label to real
  D_fakes = torch.zeros([batch_size, 1]).to(DEVICE) # Discriminator Label to fake


  data_loader = download_data(batch_size)
  print(f"Starting the training {model_name}")
  print(f"Batch size = {batch_size}, learning rate = {learning_rate}")
  print(f"Beta = {beta}")
  print(f"N critic = {n_critic}, N epochs = {max_epoch}, Random Seed = {random_seed}, Threshold = {threshold}")
  epochs = []
  g_loss = []
  d_loss = []
  g_loss_per_step = []
  d_loss_per_step = []
  steps = []
  avg_f1 = []
  avg_recall = []
  avg_precision = []
  for epoch in range(max_epoch):
      precision_scores = []
      recall_scores = []
      f1_scores = []
      for idx, (images, labels) in enumerate(data_loader):
          D.zero_grad()
          ## Training Discriminator
          # Real data
          x = images.to(DEVICE)
          y = labels.view(batch_size, 1)
          y = to_onehot(y).to(DEVICE)

          # Sampling
          z = torch.randn(batch_size, n_noise).to(DEVICE)
          x_fake = G(z, y)

          # Discriminator Loss
          x_outputs = D(x, y)
          z_outputs = D(x_fake, y)
          D_x_loss = criterion(x_outputs, D_labels)
          D_z_loss = criterion(z_outputs, D_fakes)
          D_loss = 0.5 * (D_x_loss + D_z_loss)

          D_loss.backward()
          D_opt.step()

          if step % n_critic == 0:
              D.zero_grad()
              G.zero_grad()
              # Training Generator
              z = torch.randn(batch_size, n_noise).to(DEVICE)
              z_outputs = D(G(z, y), y)
              G_loss = criterion(z_outputs, D_labels)

              G_loss.backward()
              G_opt.step()

          with torch.no_grad():
            fake_images = G(torch.randn(batch_size, n_noise).to(DEVICE), y).detach()
            D.eval()
            predictions_real = D(images.to(DEVICE), y).cpu().numpy()
            predictions_fake = D(fake_images, y).cpu().numpy()
            D.train()
            
            labels_real = np.ones((batch_size, 1))
            labels_fake = np.zeros((batch_size, 1))
            labels = np.concatenate([labels_real, labels_fake], axis=0)
            predictions = np.concatenate([predictions_real, predictions_fake], axis=0)
            precision, recall, f1_score = calculate_f1_score(labels, predictions, threshold)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1_score)

          step += 1
          if step %50==0:
              g_loss_per_step.append(G_loss.item())
              d_loss_per_step.append(D_loss.item())
              steps.append(step)
              print('Step: {}, D Loss: {}, G Loss: {}'.format(step, D_loss.item(), G_loss.item()))

      print('Epoch: {}/{}, D Loss: {}, G Loss: {}'.format(epoch, max_epoch, D_loss.item(), G_loss.item()))
      print(f"Epoch {epoch}: Precision = {sum(precision_scores) / len(precision_scores)}, Recall = {sum(recall_scores) / len(recall_scores)}, F1 Score = {sum(f1_scores) / len(f1_scores)}")
      avg_f1.append(sum(f1_scores) / len(f1_scores))
      avg_recall.append(sum(recall_scores) / len(recall_scores))
      avg_precision.append(sum(precision_scores) / len(precision_scores))


      epochs.append(epoch)
      g_loss.append(G_loss.item())
      d_loss.append(D_loss.item())
      G.eval()
      img = get_sample_image(G, n_noise)
      imsave(f'{path_img}/{model_name}_epoch{str(epoch).zfill(3)}.jpg', img, cmap='gray')
      G.train()

  
  G.eval()
  img = get_sample_image(G, n_noise)
  plt.figure(figsize=(12, 300/400*12))
  imshow(get_sample_image(G, n_noise), cmap='gray')
  imsave(f'{path_img}/{model_name}_last.jpg', img, cmap='gray')
  state_dict = {
              'epoch': epoch,
              'D_loss': D_loss,
              'G_loss': G_loss,
              'D_state_dict': D.state_dict(),
              'G_state_dict': G.state_dict()}
  print("Finishing the training with losses: D loss = ", D_loss.item(), " G loss = ", G_loss.item())
  # save the state dictionary to a file
  torch.save(state_dict, f'{path_model}/model.tar') 
  results_loss = pd.DataFrame()
  results_loss["epoch"] = epochs
  results_loss["G_loss"] = g_loss
  results_loss["D_loss"] = d_loss

  results_loss["F1_score"] = avg_f1
  results_loss["Precision"] = avg_precision
  results_loss["Recall"] = avg_recall
  results_loss.to_csv(f'{path_model}/{model_name}_G_D_loss_scores_per_epoch.csv')
  print(f"loss values are saved in {path_model}")
 
  results_per_step = pd.DataFrame()
  results_per_step["step"] = steps
  results_per_step["G_loss_step"] = g_loss_per_step
  results_per_step["D_loss_step"] = d_loss_per_step
  results_per_step.to_csv(f'{path_model}/{model_name}_G_D_loss_scores_per_step.csv')
  print(f"loss values PER STEP are saved in {path_model}_G_D_loss_scores_per_step.csv")




def Generate_images_for_FID(digit, model_path, save_path, num_images=100):
  # Set up the device
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  # Set up the generator and load the saved weights
  checkpoint = torch.load(f"{model_path}/model.tar")

  generator = Generator().to(device)
  # generator.load_state_dict(torch.load("generator_weights.pth", map_location=device))
  generator.load_state_dict(checkpoint['G_state_dict'])

  # Set up the noise vector and condition vector
  noise = Variable(torch.randn(num_images, 100)).to(device)
  condition = Variable(torch.zeros(num_images, 10)).to(device)
  condition[:, int(digit)] = 1  # set the condition vector to generate only 0 digits

  directory_name = f"{save_path}/{digit}"
  
  if os.path.exists(directory_name):
      # If it exists, remove it recursively
      shutil.rmtree(directory_name)

  # Create a new directory
  os.mkdir(directory_name)
  # Generate and save the images
  for i in range(num_images):
      with torch.no_grad():
          fake_image = generator(noise[i].unsqueeze(0), condition[i].unsqueeze(0)).detach().cpu()
      image_path = f"{save_path}/{digit}/generated_image_{i}.png"
      vutils.save_image(fake_image, image_path, normalize=True)
      # # Open the saved image and display it
      # img = Image.open(image_path)
      # img.show()


def calculate_activation_statistics(data_loader, model):
    device = next(model.parameters()).device
    act = []
    for batch_idx, (images, _) in enumerate(data_loader):
        with torch.no_grad():
            images = images.to(device)
            features = model(images)[1]
            act.append(features.cpu().detach().numpy())

    # act = np.concatenate(act, axis=0)
    act = np.concatenate(act, axis=0).reshape(-1, 2048)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_fid_score(real_images_path, generated_images_path, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    real_images = ImageFolder(real_images_path, transform=transform)
    generated_images = ImageFolder(generated_images_path, transform=transform)

    real_loader = DataLoader(real_images, batch_size=batch_size, shuffle=True)
    generated_loader = DataLoader(generated_images, batch_size=batch_size, shuffle=True)
    model = inception_v3(pretrained=True, transform_input=False, aux_logits=True)
    model.fc = nn.Identity()
    model = nn.Sequential(nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False), model)
    model.to(device).eval()

    mu_real, sigma_real = calculate_activation_statistics(real_loader, model)
    mu_gen, sigma_gen = calculate_activation_statistics(generated_loader, model)

    # calculate FID score
    diff = mu_real - mu_gen

    # Add a small positive constant to the diagonal of the covariance matrices
    eps = 1e-6
    sigma_real += eps * np.eye(sigma_real.shape[0])
    sigma_gen += eps * np.eye(sigma_gen.shape[0])

    sqrtm_real = sqrtm(sigma_real)
    sqrtm_gen = sqrtm(sigma_gen)
    covmean, _ = scipy.linalg.sqrtm((sigma_real + sigma_gen) / 2.0, disp=False)


    fid = (diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_gen) - 2 * np.trace(covmean.real))

    return fid



def plot_f1_precison_recall(path_to_csv):
  df = pd.read_csv(path_to_csv)
  fig = px.line(df, x="epoch", y=["F1_score", "Precision", "Recall"],
              title="F1 Score, Precision, and Recall over Epochs")

  # Show the plot
  fig.show()
  
 
def plot_loss(path_to_csv):
  df = pd.read_csv(path_to_csv)
  fig = px.line(df, x="epoch", y=["G_loss", "D_loss"],
                title="G Loss and D Loss vs Epochs")

  # Show the plot
  fig.show()
  
  
def plot_loss_step(path_to_csv):
  df = pd.read_csv(path_to_csv)
  fig = px.line(df, x="step", y=["G_loss_step", "D_loss_step"],
                title="G Loss and D Loss per every 50 step")

  # Show the plot
  fig.show()