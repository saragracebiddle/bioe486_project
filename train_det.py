import time
import os
from tqdm import tqdm
import config
from torch.utils.data import DataLoader

import torch

def train(objectDetector, bboxLossFunc, classLossFunc, trainDS, testDS, opt):
    print("[INFO] training the network...")
    H = {"total_train_loss": [], "total_val_loss": [], "train_class_acc": [],
	 "val_class_acc": []}
    
    # calculate steps per epoch for training and validation set   
    trainSteps = len(trainDS) // config.BATCH_SIZE
    valSteps = len(testDS) // config.BATCH_SIZE
    # create data loaders
    trainLoader = DataLoader(trainDS, batch_size=config.BATCH_SIZE,
	    shuffle=True, num_workers=os.cpu_count(), pin_memory=config.PIN_MEMORY)
    testLoader = DataLoader(testDS, batch_size=config.BATCH_SIZE,
	    num_workers=os.cpu_count(), pin_memory=config.PIN_MEMORY)
    
    startTime = time.time()

    

    for e in tqdm(range(config.NUM_EPOCHS)):
	    # set the model in training mode
        objectDetector.train()
	    # initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0
	    # initialize the number of correct predictions in the training
	    # and validation step
        trainCorrect = 0
        valCorrect = 0

        	# loop over the training set
        for (images, labels, bboxes) in trainLoader:
		    # send the input to the device
            (images, labels, bboxes) = (images.to(config.DEVICE),
                labels.to(config.DEVICE), bboxes.to(config.DEVICE))
		    # perform a forward pass and calculate the training loss
            predictions = objectDetector(images)
            bboxLoss = bboxLossFunc(predictions[0], bboxes)
            classLoss = classLossFunc(predictions[1], labels)
            totalLoss = (config.BBOX * bboxLoss) + (config.LABELS * classLoss)
		    # zero out the gradients, perform the backpropagation step,
		    # and update the weights
            opt.zero_grad()
            totalLoss.backward()
            opt.step()
		    # add the loss to the total training loss so far and
		    # calculate the number of correct predictions
            totalTrainLoss += totalLoss
            trainCorrect += (predictions[1].argmax(1) == labels).type(
			    torch.float).sum().item()

    	# switch off autograd
        with torch.no_grad():
		# set the model in evaluation mode
            objectDetector.eval()
		# loop over the validation set
            for (images, labels, bboxes) in testLoader:
			# send the input to the device
                (images, labels, bboxes) = (images.to(config.DEVICE),
				labels.to(config.DEVICE), bboxes.to(config.DEVICE))
			# make the predictions and calculate the validation loss
                predictions = objectDetector(images)
                bboxLoss = bboxLossFunc(predictions[0], bboxes)
                classLoss = classLossFunc(predictions[1], labels)
                totalLoss = (config.BBOX * bboxLoss) + \
				(config.LABELS * classLoss)
                totalValLoss += totalLoss
			# calculate the number of correct predictions
                valCorrect += (predictions[1].argmax(1) == labels).type(
				torch.float).sum().item()
            
        	# calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
	    # calculate the training and validation accuracy
        trainCorrect = trainCorrect / len(trainDS)
        valCorrect = valCorrect / len(testDS)
	    # update our training history
        H["total_train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["train_class_acc"].append(trainCorrect)
        H["total_val_loss"].append(avgValLoss.cpu().detach().numpy())
        H["val_class_acc"].append(valCorrect)
	    # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
        print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
		    avgTrainLoss, trainCorrect))
        print("Val loss: {:.6f}, Val accuracy: {:.4f}".format(
		    avgValLoss, valCorrect))
          
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
	    endTime - startTime))

    return H