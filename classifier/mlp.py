import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import time 
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle 
import os
from torch.nn.functional import pad


class two_layers(nn.Module):
    def __init__(self, input_size, output_size):
        super(two_layers , self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 2)
        self.conv2 = nn.Conv2d(6, 12, 2)

        self.layer1 = nn.Linear(192, 216, bias=True)
        self.layer2 = nn.Linear(216, 6, bias=True)
    
    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.layer1(x))
        y = self.layer2(x)

        return y

def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)

def train_network(net, optimizer, criterion, scheduler, trainloader, valloader, input_size):
    net.train()
    max_acc = 0
    for epoch in range(50):  # loop over the dataset multiple times

        train_loss = 0.0
        for i, data in enumerate(trainloader):

            data_tensors = data['tensor']
            data_labels = data['class']

            inp_len = data_tensors.shape[0]

            #inputs = data_tensors.reshape((inp_len, input_size))
            inputs = data_tensors.permute(0,3,1,2)


            labels = torch.LongTensor(data_labels).to(device='cuda:0')

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data_tensors.size(0)
        scheduler.step()

            # print statistics
        train_loss = train_loss/len(trainloader.dataset)

        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch+1, 
            train_loss
            ))
        accuracy = test_network(net, optimizer, criterion, valloader, input_size=input_size)
        if accuracy > max_acc :
            max_acc = accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, './cls_ckpt/test.pt')
        print("Max Accuracy: ", max_acc)
    print('Finished Training')

def test_network(net, optimizer, criterion, valloader, input_size):
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    net.eval() # prep model for *evaluation*
    
    timings = []
    for i, data_tensor in enumerate(valloader):
        #print(i, data['tensor'].shape)
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        data_tensors = data_tensor['tensor']
        data_labels = data_tensor['class']
        inp_len = data_tensors.shape[0]

        data = data_tensors.permute(0,3,1,2)
        data = data_tensors.reshape((inp_len, input_size))

        target = torch.LongTensor(data_labels).to(device='cuda:0')
        # forward pass: compute predicted outputs by passing inputs to the model
        starter
        output = net(data)
        ender
        
        curr_time = starter.(ender)
        timings.append(curr_time)
        # calculate the loss
        loss = criterion(output, target)
        # update test loss 
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(inp_len):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # calculate and print avg test loss
    test_loss = test_loss/len(valloader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(10):
        if class_total[i] > 0:
            print('Test Recall of %5s: %2d%% (%2d/%2d)' % (
                str(i), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))    
    print(np.mean(timings))
    return np.sum(class_correct) / np.sum(class_total)    

class CroppedTensorDataset(Dataset):
    def __init__(self, tensors_dir, output_file, slice_size):
        self.tensors_dir = tensors_dir
        self.output_file = output_file
        self.slice_size = slice_size
        self.elem_dict = {}

        self.tensor_files = [f for f in os.listdir(self.tensors_dir) if os.path.isfile(os.path.join(self.tensors_dir, f))]

        with open(self.output_file, 'rb') as f:
            output = pickle.load(f)

        obj_counter = 0

        counter_0 = 0
        counter_1 = 0
        counter_2 = 0
        counter_3 = 0
        counter_4 = 0
        counter_5 = 0
        counter_else = 0

        for frame_name in output: 
            frame_index = int(frame_name.replace('seq_0_frame_', '').replace('.pkl', ''))
            #frame_index = frame_name-1 
            sep_head_file = os.path.join(self.tensors_dir,'sep_head' + str(frame_index) +'.pt')
            shared_conv_file = os.path.join(self.tensors_dir,'shared_conv' + str(frame_index) + '.pt')

            boxes = output[frame_name]['box3d_lidar']
            classes = output[frame_name]['classes']

            for obj_index in range(len(boxes)) : 
                box = boxes[obj_index]
                obj_class = classes[obj_index]
                
                #if classes[obj_index] == 4:
                #    obj_class = 2
                #elif classes[obj_index] == 5:
                #    obj_class = 3


                if obj_class == 0:
                    counter_0 += 1
                elif obj_class == 1:
                    counter_1 += 1
                elif obj_class == 2:
                    counter_2 += 1
                elif obj_class == 3:
                    counter_3 += 1
                elif obj_class == 4:
                    counter_4 += 1
                elif obj_class == 5:
                    counter_5 += 1
                


                if (abs(box[0]) <= 70 and abs(box[1]) <= 70): 
                    if obj_class == 3: 
                        if counter_3 < 5100 :
                            old_xy = box[:2]
                            new_xy = torch.tensor([234, 234]) + (torch.round(old_xy * (468 / (2 * 74.88))).to(torch.int))

                            obj_data = [new_xy, sep_head_file, shared_conv_file, obj_class]
                            self.elem_dict[obj_counter] = obj_data
                            obj_counter += 1
                    elif obj_class == 0: 
                        if counter_0 < 5100 :
                            old_xy = box[:2]
                            new_xy = torch.tensor([234, 234]) + torch.round(old_xy * (468 / (2 * 74.88))).to(torch.int)

                            obj_data = [new_xy, sep_head_file, shared_conv_file, obj_class]
                            self.elem_dict[obj_counter] = obj_data
                            obj_counter += 1
                    elif obj_class == 2: 
                        if counter_2 < 5100 :
                            old_xy = box[:2]
                            new_xy = torch.tensor([234, 234]) + torch.round(old_xy * (468 / (2 * 74.88))).to(torch.int)

                            obj_data = [new_xy, sep_head_file, shared_conv_file, obj_class]
                            self.elem_dict[obj_counter] = obj_data
                            obj_counter += 1
                    elif obj_class == 1: 
                        if counter_1 < 5100 :
                            old_xy = box[:2]
                            new_xy = torch.tensor([234, 234]) + torch.round(old_xy * (468 / (2 * 74.88))).to(torch.int)

                            obj_data = [new_xy, sep_head_file, shared_conv_file, obj_class]
                            self.elem_dict[obj_counter] = obj_data
                            obj_counter += 1
                    elif obj_class == 5: 
                        if counter_5 < 5100 :
                            old_xy = box[:2]
                            new_xy = torch.tensor([234, 234]) + torch.round(old_xy * (468 / (2 * 74.88))).to(torch.int)

                            obj_data = [new_xy, sep_head_file, shared_conv_file, obj_class]
                            self.elem_dict[obj_counter] = obj_data
                            obj_counter += 1
                    else :
                        old_xy = box[:2]
                        new_xy = torch.tensor([234, 234]) + torch.round(old_xy * (468 / (2 * 74.88))).to(torch.int)

                        obj_data = [new_xy, sep_head_file, shared_conv_file, obj_class]
                        self.elem_dict[obj_counter] = obj_data
                        obj_counter += 1

        counter0 = 0
        counter1 = 0
        counter2 = 0
        counter3 = 0
        counter4 = 0
        counter5 = 0
        counterelse = 0

        for key in self.elem_dict :
            if self.elem_dict[key][3] == 0:
                counter0 += 1
            elif self.elem_dict[key][3] == 1:
                counter1 += 1
            elif self.elem_dict[key][3] == 2:
                counter2 += 1
            elif self.elem_dict[key][3] == 3:
                counter3 += 1
            elif self.elem_dict[key][3] == 4:
                counter4 += 1
            elif self.elem_dict[key][3] == 5:
                counter5 += 1
            else :
                counterelse += 1

        print("All tensors 0 class: ", counter0, ",1 class: ", counter1, ",2 class: ", counter2, ",3 class: ", counter3, ",4 class: ", counter4, ",5 class: ", counter5, ",all objects: ", obj_counter)

    def __len__(self):
        return len(self.elem_dict)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        box_data = self.elem_dict[idx]

        new_xy = box_data[0]
        sep_head_rot = torch.squeeze(torch.load(box_data[1])['hm']).permute(1,2,0)
        #shared_conv = torch.squeeze(torch.load(box[2]))

        sep_head_slice = sep_head_rot[new_xy[1]-self.slice_size : new_xy[1] + self.slice_size +1, new_xy[0] - self.slice_size : new_xy[0] + self.slice_size +1, :]
        #shared_conv_slice = shared_conv[:, new_xy[0]-self.slice_size : new_xy[0] + self.slice_size, new_xy[1] - self.slice_size : new_xy[1] + self.slice_size]

        sample = {'tensor': sep_head_slice, 'class': box_data[3]}

        return sample   

def process_output(output_dict, sep_head_tensor, slice_size, net) :
    new_out = output_dict.copy()
    sep_head_tensor = pad(torch.squeeze(sep_head_tensor), (0,0,7,7,7,7), "constant", 0)
    temp_boxes = new_out['box3d_lidar'].clone()
    temp_boxes = torch.round(temp_boxes[:,:2] * (468 / (2 * 74.88)) + 234).to(dtype=torch.int32)
    slices_list = []
    temp_boxes = temp_boxes.detach().tolist()
    slices_list = [sep_head_tensor[temp_boxes[i][1] + 7 - slice_size : temp_boxes[i][1] + 7 + slice_size, temp_boxes[i][0] + 7 - slice_size : temp_boxes[i][0] + 7 + slice_size, :] for i in range(len(temp_boxes))]
    data = torch.stack(slices_list).permute(0,3,1,2)
    output = net(data)
    _, pred = torch.max(output, 1)   
    filtered_boxes = np.argwhere(np.array(pred.cpu().detach()) % 2 == 0).tolist()
    filtered_boxes = [ int(x[0]) for x in filtered_boxes]
    new_out['label_preds'] = output_dict['label_preds'][filtered_boxes]
    new_out['scores'] = output_dict['scores'][filtered_boxes]
    new_out['box3d_lidar'] = output_dict['box3d_lidar'][filtered_boxes]
    if new_out['box3d_lidar'].shape[0] != 1 : 
        new_out['box3d_lidar'] = new_out['box3d_lidar'].squeeze()

    return new_out


