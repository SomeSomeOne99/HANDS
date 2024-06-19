#%%
import torch, torchvision, torch.nn as nn, json, os, matplotlib.pyplot as plt, numpy as np
from torch.utils.data import DataLoader, random_split
from PIL import Image
from random import randint
#%%
#Instantiating variables
BATCH_SIZE = 32 # Adjust the batch size depending on your GPU memory
IMAGE_SIZE = 448
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
torch.manual_seed(420)
#%%
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.layer1 = nn.Conv2d(3, 16, 5, padding=2)
        self.layer2 = nn.Conv2d(16, 32, 3, padding=1)
        self.layer3 = nn.Conv2d(32, 64, 3, padding=1)
        self.layer4 = nn.Linear(64*56*56, 1024)
        self.layer5 = nn.Linear(1024, 14)
    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = nn.functional.max_pool2d(x, 2, stride=2)
        x = self.layer2(x)
        x = torch.relu(x)
        x = nn.functional.max_pool2d(x, 2, stride=2)
        x = self.layer3(x)
        x = torch.relu(x)
        x = nn.functional.max_pool2d(x, 2, stride=2)
        x = x.view(-1, 64*56*56)
        x = self.layer4(x)
        x = torch.relu(x)
        x = self.layer5(x)
        x = torch.sigmoid(x)
        return x
    # def forward(self, x): return self.layer4(().view(-1, 1048576))
    # def forward(self, x): return self.layer2(torch.relu(self.layer1(x)).view(-1, 4194304))
#%%
from torch.utils.data import dataloader, Dataset
class gun(Dataset):
    def __init__(self, jsonPath, transforms = None):
        self.dataset = json.load(open(jsonPath))
        self.transforms = transforms
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        #print(index, type(index))
        #print(self.dataset[index], type(self.dataset[index]))
        image = Image.open(self.dataset[index]["imagePath"])
        if self.transforms != None:
            image = self.transforms(image)
        alltargetstuples = self.dataset[index]["targets"]
        alltargets = []
        for tuple in alltargetstuples:
            alltargets.append(tuple[0])
            alltargets.append(tuple[1])
        alltargets = torch.Tensor(alltargets)
        return image, alltargets
#%%
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                            #torchvision.transforms.RandomRotation(15),
                                            #torchvision.transforms.RandomHorizontalFlip(),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
dataset = gun("normalised3.json", transforms=transform)
def collate_fn(batch):
    image, labels = zip(*batch)
    image = torch.stack(image, dim=0)
    labels = torch.stack(labels, dim=0)
    #print(image.shape, labels.shape)
    return image, labels
trainset, testset = random_split(dataset, [int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)])
trainloader = DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True, collate_fn=collate_fn)
testloader = DataLoader(testset, batch_size = BATCH_SIZE, shuffle = True, collate_fn=collate_fn)
# %%
model = NN().to(device)
loss_fn = torch.nn.MSELoss()
LEARNING_RATE = 0.0001
optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) #0.0001
#%%
def testmodel(full_output = True):
    model.eval()
    with torch.no_grad():
        batch_loss = 0
        batch_count = 0
        untransform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(), torchvision.transforms.Resize((1080, 1080))])
        for image, labels in testloader:
            labels = labels.to(device)
            image = image.to(device)
            output = model(image)
            if full_output:
                print(output)
                image_example = untransform(image[0].to("cpu"))
                output_example = output[0].to("cpu")
                labels_example = labels[0].to("cpu")
                for i in range(0, len(labels_example), 2):
                    plt.scatter(labels_example[i] * 1080, labels_example[i+1] * 1080, c="black", marker="x")
                    plt.annotate("t" + str(int(i/2)), (labels_example[i] * 1080, labels_example[i+1] * 1080), textcoords="offset points", xytext=(0, 10))
                #for point in zip([output_example[i] for i in range(0, len(output_example), 2)], [output_example[i] for i in range(1, len(output_example), 2)]):
                    #pass #plt.scatter(point[0] * 1920, point[1] * 1080)
                for i in range(0, len(output_example), 2):
                    plt.scatter(output_example[i] * 1080, output_example[i+1] * 1080, marker="x")
                    plt.annotate("o" + str(int(i/2)), (output_example[i] * 1080, output_example[i+1] * 1080), textcoords="offset points", xytext=(0, 10))
                plt.imshow(image_example)
                plt.show()
            loss = loss_fn(output, labels)
            batch_loss += loss.item()
            batch_count += 1
        if full_output:
            print("test", batch_loss/batch_count)
    model.train()
    return batch_loss/batch_count
#%%
EPOCHS = 1000
epoch_losses = []
epoch_testlosses = []
EPOCHS_PER_CHECKPOINT = 5
last_checkpoint = float("inf")
MAXIMUM_CHECKPOINTFAIL_PATIENCE = 300
checkpointfail_tolerance = 0
last_checkpointpass_model = None
checkpointfail_points = []
#%%
for epoch_num in range(EPOCHS):
    epoch_loss = 0
    for image, labels in trainloader:
        labels = labels.to(device)
        optimiser.zero_grad()
        image = image.to(device)
        output = model(image)
        #print(output.shape)
        #print(len(labels), labels[0].shape)
        loss = loss_fn(output, labels)
        loss.backward()
        optimiser.step()
        epoch_loss += loss.item()
    print(epoch_num, epoch_loss/len(trainloader))
    epoch_losses.append(epoch_loss/len(trainloader))
    epoch_testlosses.append(testmodel(False))
    if epoch_num % EPOCHS_PER_CHECKPOINT == 0:
        new_checkpoint = testmodel(False)
        print("previous:", last_checkpoint)
        print(" current:", new_checkpoint)
        if new_checkpoint < last_checkpoint: # improvement occured
            print("checkpoint passed!")
            last_checkpoint = new_checkpoint
            checkpointfail_tolerance = 0
            last_checkpointpass_model = model.state_dict() # save the last model
        else: #stagnate or overfit
            checkpointfail_tolerance += 1
            checkpointfail_points.append((epoch_num, new_checkpoint))
            print("checkpoint fail - tolerance", checkpointfail_tolerance)
            if checkpointfail_tolerance == MAXIMUM_CHECKPOINTFAIL_PATIENCE:
                print("failed too many checkpoints!")
                model.load_state_dict(last_checkpointpass_model)
                break
#%%
plt.title("LR: " + str(LEARNING_RATE) + "; Final: " + str(last_checkpoint))
plt.plot(epoch_losses)
plt.plot(epoch_testlosses)
plt.scatter([epoch_num for epoch_num, checkpoint_value in checkpointfail_points], [checkpoint_value for epoch_num, checkpoint_value in checkpointfail_points], marker="x")
#plt.bar(list(range(1, len(epoch_losses))), [20*(epoch_losses[i] - epoch_losses[i-1]) for i in range(1, len(epoch_losses))])
#plt.bar(list(range(1, len(epoch_testlosses))), [20*(epoch_testlosses[i] - epoch_testlosses[i-1]) for i in range(1, len(epoch_testlosses))])
plt.show()
#%%
def preprocessing():
    json.dump([{"imagePath": file[:file.rindex(".")] + ".jpg", "targets": [(point[0] / 1920, point[1] / 1080) for shape in json.load(open(file))["shapes"] for point in shape["points"]]} for file in os.listdir("C:/Users/tahir/Desktop/HANDS") if file.endswith(".json")], open("normalised.json", "w"))
#%%
def plotdata():
    normalisedData = json.load(open("normalised.json"))
    selected = normalisedData[randint(0, len(normalisedData))]
    plt.imshow(plt.imread(selected["imagePath"]))
    for target in selected["targets"]:
        plt.scatter(target[0] * 1920, target[1] * 1080)
    plt.show()
plotdata()
# %%
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))
# %%
images, labels = next(iter(trainloader))
#print(all)