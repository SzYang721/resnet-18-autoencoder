
from scripts.config import BATCH_SIZE, EPOCHS, MODEL_FILENAME, EARLY_STOP_THRESH, LR, WORKERS, TARGET_EPOCHS
from classes.resnet_autoencoder import AE
from scripts.data_loading import get_cifar10_data_loaders
from scripts.utils import train_epoch, test_epoch, plot_ae_outputs, checkpoint, resume
import torch
import datetime
import numpy as np

from torchsummary import summary

from models.resnet import resnet18 as encoder
from models.resnet_decode import resnet18 as decoder
from models.Autoencoder_CNN import Autoencoder


if __name__=='__main__': 

    if torch.cuda.is_available():
         device = 'cuda' 
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("Is the current version of PyTorch built with MPS activated?",torch.backends.mps.is_built())
    else:
        device = 'cpu'

    print("Using device:", device)

    print("Defining model...")
    cae = AE('default')
    Encoder = encoder(num_classes=10,fc_bias=True,fixdim=True,ETF_fc=True,SOTA=False)
    # load_path = "/data5/model_weights/"+"Resnet18-design-SGD"+"/"
    # i = TARGET_EPOCHS
    # Encoder.load_state_dict(torch.load(load_path + 'epoch_' + str(i + 1).zfill(3) + '.pth',map_location=device))
    Decoder = decoder(fixdim=True,SOTA=False)
    cae = Autoencoder(Encoder, Decoder)
    summary(cae.encoder.to(device), input_size=(3, 32, 32), device = device)
    summary(cae.decoder.to(device), input_size=(512, 1, 1), device = device)
    # Define the training parameters
    params_to_optimize = [
        {'params': cae.parameters()}
    ]
    parameters = filter(lambda p: p.requires_grad, cae.parameters())
    print('# of model parameters: ' + str(sum([np.prod(p.size()) for p in parameters])))
    # Define the loss function
    loss_fn = torch.nn.MSELoss()
    # Define the optimizer
    optim = torch.optim.Adam(params_to_optimize, lr=LR, weight_decay=1e-05) 
    # Move model to the selected device
    cae.to(device)

    print('Loading data...')
    train_loader, test_loader, train_dataset, test_dataset = get_cifar10_data_loaders(download=True, batch_size=BATCH_SIZE, num_workers=WORKERS)
   
    # Initialize varialbes
    best_val_loss = 1000000
    best_epoch = 0
    #Training loop 
    t1 = datetime.datetime.now()
    print("Start training..")
    for epoch in range(EPOCHS):
        print('> Epoch ' + str(epoch + 1))
        train_loss =train_epoch(cae,device,train_loader,loss_fn,optim)
        print("Evaluating on test set...")
        val_loss = test_epoch(cae,device,test_loader,loss_fn)
        print('\n EPOCH {}/{} \t train loss {} \t val loss {}.'.format(epoch + 1, EPOCHS,train_loss,val_loss))
        print('Plotting results...')
        plot_ae_outputs(cae,"train_dataset", epoch, train_dataset, device,n=10)
        plot_ae_outputs(cae,"test_dataset", epoch, test_dataset, device,n=10)
        if val_loss <= best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                checkpoint(cae, best_epoch, best_val_loss, MODEL_FILENAME)
        elif epoch - best_epoch > EARLY_STOP_THRESH:
            print(f"Early stopped training at epoch {epoch}")
            model, _, _ = resume(cae, MODEL_FILENAME)
            break  # terminate the training loop
    print("Total training time:",datetime.datetime.now()-t1)


