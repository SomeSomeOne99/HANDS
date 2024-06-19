import pygame.camera
import cv2, torchvision, torch.nn as nn, torch, pygame, numpy, time, os
from PIL import Image
IMAGE_SIZE = 448
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                            #torchvision.transforms.RandomRotation(15),
                                            #torchvision.transforms.RandomHorizontalFlip(),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
class NN(nn.Module): # from HANDS 0.2
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
## file list init
image_files = [file for file in os.listdir("C:/Users/tahir/Desktop/HANDS") if file.endswith(".jpg")]
current_file_i = 0
## erroneous classifications (using modelfile)
major_error_indices = [8,24,41,66,81,83,84,88,91,93,94,99,102,113,123,131]
minor_error_indices = [29,36,51,64,76,77,96,110,111]
## model init
model = NN()
model = model.to(device)
model.load_state_dict(torch.load("modelfile2"))
model.eval()
## pygame init
pygame.init()
IMAGE_RESOLUTION = (1920, 1080)
pygame.camera.init()
#camera = pygame.camera.Camera(0, IMAGE_RESOLUTION, "RGB")#cv2.VideoCapture(0)
#camera.start()
screen = pygame.display.set_mode(IMAGE_RESOLUTION)
clock = pygame.time.Clock()
FPS = 15
running = True
last_time = time.time()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                current_file_i += 1
                if current_file_i == len(image_files):
                    current_file_i = 0
            elif event.key == pygame.K_LEFT:
                current_file_i -= 1
                if current_file_i == -1:
                    current_file_i = len(image_files) - 1
            elif event.key == pygame.K_w: # W: next major error
                for i, x in enumerate(major_error_indices):
                    if x > current_file_i:
                        current_file_i = x
                        break
                    if i == len(major_error_indices) - 1:
                        current_file_i = major_error_indices[0] # cycle to beginning
            elif event.key == pygame.K_q: # Q: last major error
                for i, x in enumerate(major_error_indices[::-1]):
                    if x < current_file_i:
                        current_file_i = x
                        break
                    if i == len(major_error_indices) - 1:
                        current_file_i = major_error_indices[-1] # cycle to beginning
            elif event.key == pygame.K_s: # S: next minor error
                for i, x in enumerate(minor_error_indices):
                    if x > current_file_i:
                        current_file_i = x
                        break
                    if i == len(minor_error_indices) - 1:
                        current_file_i = minor_error_indices[0] # cycle to beginning
            elif event.key == pygame.K_a: # A: last minor error
                for i, x in enumerate(minor_error_indices[::-1]):
                    if x < current_file_i:
                        current_file_i = x
                        break
                    if i == len(minor_error_indices) - 1:
                        current_file_i = minor_error_indices[-1] # cycle to beginning
    #image_surface = camera.get_image()
    #image_bytes = camera.get_raw()
    if True: #result:
        #pygame.image.save(image_surface, "input.jpg")
        file_image = Image.open(image_files[current_file_i])
        image = transforms(file_image).unsqueeze(0)
        image = image.to(device)
        output = model(image)
        output = output.to("cpu")
        outputlist = output.tolist()[0]
        points = [(outputlist[i]*(IMAGE_RESOLUTION[0]), outputlist[i+1]*(IMAGE_RESOLUTION[1])) for i in range(0, len(outputlist), 2)]
        #print(points)
        # display to screen
        screen.fill((0,0,0))
        #image_array_rgba = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGBA)
        #print(image_array.shape)
        screen.blit(pygame.image.fromstring(file_image.tobytes(), IMAGE_RESOLUTION, "RGB"), pygame.Rect(0, 0, 100, 100))
        for point in points:
            pygame.draw.rect(screen, (255,255,255), pygame.Rect(point[0], point[1], 10, 10))
        screen.blit(pygame.font.Font("freesansbold.ttf", 15).render(str(points), True, (0,0,0)), pygame.Rect(0, 0, 100, 100))
        screen.blit(pygame.font.Font("freesansbold.ttf", 15).render(str(current_file_i) + " of " + str(len(image_files) - 1), True, (0,0,0)), pygame.Rect(0, 25, 100, 100))
        screen.blit(pygame.font.Font("freesansbold.ttf", 15).render(image_files[current_file_i], True, (0,0,0)), pygame.Rect(0, 50, 100, 100))
        screen.blit(pygame.font.Font("freesansbold.ttf", 15).render("MAJOR" if current_file_i in major_error_indices else ("MINOR" if current_file_i in minor_error_indices else ""), True, (0,0,0)), pygame.Rect(0, 75, 100, 100))
        pygame.display.flip()
    clock.tick(FPS)
    time_ = time.time()
    print(1/(time_ - last_time))#print(1/(time-last_time))
    last_time = time_