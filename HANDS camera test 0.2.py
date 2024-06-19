import pygame.camera
import cv2, torchvision, torch.nn as nn, torch, pygame
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
model = NN()
model = model.to(device)
model.load_state_dict(torch.load("modelfile"))
model.eval()
pygame.init()
IMAGE_RESOLUTION = (1920, 1080)
camera = cv2.VideoCapture(0)
screen = pygame.display.set_mode(IMAGE_RESOLUTION)
clock = pygame.time.Clock()
FPS = 5
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    result, bgr_image = camera.read()
    if result:
        cv2.imwrite("input.jpg", bgr_image)
        file_image = Image.open("input.jpg") # "C:/Users/tahir/Desktop/HANDS/WIN_20240524_20_44_42_Pro.jpg")
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
        pygame.display.flip()
    #clock.tick(0.5)