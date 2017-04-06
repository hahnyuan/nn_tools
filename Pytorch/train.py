import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch
from .eval import eval_classification_net

def train_classification_net(net,trainloader,testloader=None,save_path=None,base_lr=0.01,num_epoch=5,use_cuda=True,optimizer=None,lr_change=None):
    # train a classification net
    # the trainloader should return (input, labels(not one-hot version))
    optimizer=optim.Adam(net.parameters(),lr=base_lr)
    # TODO optimizer passed in
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epoch):
        running_loss=0.0
        for i, (inputs,labels) in enumerate(trainloader,0):
            # TODO learning rate change function
            if use_cuda:
                inputs=inputs.cuda()
                labels=labels.cuda()
            inputs_var,labels_var=Variable(inputs),Variable(labels)
            optimizer.zero_grad()
            outputs=net(inputs_var)
            loss=criterion(outputs,labels_var)
            loss.backward()
            optimizer.step()

            running_loss+=loss.data[0]
            print_iter=200
            if i%print_iter==print_iter-1:
                print(epoch+1,i+1,running_loss/print_iter)
                running_loss=0
        if testloader:
            acc=eval_classification_net(net,testloader)
            print epoch,'Accurancy:', acc
    if save_path:
        torch.save(net.state_dict(),save_path)