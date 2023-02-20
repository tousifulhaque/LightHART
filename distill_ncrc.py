import torch
import numpy as np
from Make_Dataset import Poses3d_Dataset
import PreProcessing_ncrc
from Model.model_crossview_fusion import ActTransformerMM
from Model.model_acc_only import ActTransformerAcc
from Tools.visualize import get_plot
from tqdm import tqdm
import pickle
from asam import ASAM, SAM
from timm.loss import LabelSmoothingCrossEntropy
import os



exp = 'myexp-1' #Assign an experiment id

if not os.path.exists('exps/'+exp+'/'):
    os.makedirs('exps/'+exp+'/')
PATH='exps/'+exp+'/'

#CUDA for PyTorch
print("Using CUDA....")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# KDL loss function 
def distillation(y, labels, teacher_scores, T, alpha):
    # Implementing alpha * Temp ^2 * crossEn(Q_s, Q_t) + (1-alpha)* crossEn(Q_s, y_true)
    pred_soft = F.log_softmax(y/T, dim = 1)
    # print(f'Student pred has Nan : {torch.isnan(pred_soft).any()}')
    teacher_scores_soft = F.log_softmax(teacher_scores/T, dim = 1)
    # print(f'Teacher pred has Nan : {torch.isnan(teacher_scores_soft).any()}')
    kl_div = nn.KLDivLoss(reduction= "batchmean", log_target=True)(pred_soft, teacher_scores_soft) * ( alpha * T * T * 2.0)
    # print(f'KlDiv pred has Nan : {torch.isnan(kl_div).any()}')
    loss_y_label = F.cross_entropy(y, labels) * (1.0 - alpha)
    # print(f'Y loss has Nan : {torch.isnan(loss_y_label).any()}')
    return kl_div + loss_y_label

# Parameters
print("Creating params....")
params = {'batch_size':8,
          'shuffle': True,
          'num_workers': 0}(num_epochs)= 250

# Generators
#pose2id,labels,partition = PreProcessing_ncrc_losocv.preprocess_losocv(8)
pose2id, labels, partition = PreProcessing_ncrc.preprocess()

print("Creating Data Generators...")
mocap_frames = 600
acc_frames = 150

training_set = Poses3d_Dataset( data='ncrc',list_IDs=partition['train'], labels=labels, pose2id=pose2id, mocap_frames=mocap_frames, acc_frames=acc_frames, normalize=False)
training_generator = torch.utils.data.DataLoader(training_set, **params) #Each produced sample is  200 x 59 x 3

validation_set = Poses3d_Dataset(data='ncrc',list_IDs=partition['test'], labels=labels, pose2id=pose2id, mocap_frames=mocap_frames, acc_frames=acc_frames ,normalize=False)
validation_generator = torch.utils.data.DataLoader(validation_set, **params) #Each produced sample is 6000 x 229 x 3

#Define model
print("Initiating Model...")
teacher_model = ActTransformerMM(device)
studetn_model = ActTransformerAcc(device)
model = model.to(device)


print("-----------TRAINING PARAMS----------")
#Define loss and optimizer
lr=0.0025
wt_decay=5e-4

#Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,weight_decay=wt_decay)
#optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wt_decay)

#ASAM
rho=0.5
eta=0.01
minimizer = ASAM(optimizer, model, rho=rho, eta=eta)

#Learning Rate Scheduler
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(minimizer.optimizer,(num_epochs)
#print("Using cosine")

#TRAINING AND VALIDATING
epoch_loss_train=[]
epoch_loss_val=[]
epoch_acc_train=[]
epoch_acc_val=[]

#Label smoothing
#smoothing=0.1
#criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
#print("Loss: LSC ",smoothing)

best_accuracy = 0.


def train(epoch, num_epochs, student_model, teacher_model, loss_fn):
    teacher_model.eval()
    with tqdm(total  = len(training_generator), desc = f'Epoch {epoch+1}/{num_epochs}',ncols = 128) as pbar:
        # Train
        student_model.train()
        loss = 0.
        accuracy = 0.
        cnt = 0.
        for inputs, acc_input, targets in training_generator:
            inputs = inputs.to(device); #print("Input batch: ",inputs)
            targets = targets.to(device)
            acc_input = acc_input.to(device)

            optimizer.zero_grad()

            # Ascent Step
            #print("labels: ",targets)
            predictions = student_model(acc_input.float())
            teacher_output = teacher_model(inputs.float())
            detached_pred = predictions.detach()
            teacher_output = teacher_output.detach()
            #print("predictions: ",torch.argmax(predictions, 1) )
            loss = loss_fn(detached_pred, target, teacher_output, T=2.0, alpha = 0.7)
            loss.mean().backward()
            minimizer.ascent_step()

            # Descent Step
            loss_fn(detached_pred, target, teacher_output, T=2.0, alpha = 0.7).mean().backward()
            minimizer.descent_step()

            with torch.no_grad():
                loss += loss.sum().item()
                accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
            cnt += len(targets)
        loss /= cnt
        accuracy *= 100. / cnt
        print(f"Epoch: {epoch}, Train accuracy: {accuracy:6.2f} %, Train loss: {loss:8.5f}")
        epoch_loss_train.append(loss)
        epoch_acc_train.append(accuracy)
        #scheduler.step()

        #accuracy,loss = validation(model,validation_generator)
        #Test
        model.eval()
        loss = 0.
        accuracy = 0.
        cnt = 0.
        model=model.to(device)
        with torch.no_grad():
            for inputs, targets in validation_generator:

                b = inputs.shape[0]
                inputs = inputs.to(device); #print("Validation input: ",inputs)
                targets = targets.to(device)
                
                predictions = model(inputs.float())
                
                with torch.no_grad():
                    loss += batch_loss.sum().item()
                    accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
                cnt += len(targets)
            loss /= cnt
            accuracy *= 100. / cnt
            
        
            if best_accuracy < accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(),PATH+exp+'_best_ckpt.pt'); print("Check point "+PATH+exp+'_best_ckpt.pt'+ ' Saved!')

        print(f"Epoch: {epoch},Test accuracy:  {accuracy:6.2f} %, Test loss:  {loss:8.5f}")


        epoch_loss_val.append(loss)
        epoch_acc_val.append(accuracy)


print(f"Best test accuracy: {best_accuracy}")
print("TRAINING COMPLETED :)")

#Save visualization
get_plot(PATH,epoch_acc_train,epoch_acc_val,'Accuracy-'+exp,'Train Accuracy','Val Accuracy','Epochs','Acc')
get_plot(PATH,epoch_loss_train,epoch_loss_val,'Loss-'+exp,'Train Loss','Val Loss','Epochs','Loss')
