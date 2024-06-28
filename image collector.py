import pygame.camera
import pygame, time, datetime
from PIL import Image
pygame.init()
IMAGE_RESOLUTION = (1920, 1080)
pygame.camera.init()
camera = pygame.camera.Camera(0, IMAGE_RESOLUTION, "RGB")#cv2.VideoCapture(0)
camera.start()
screen = pygame.display.set_mode(IMAGE_RESOLUTION)
clock = pygame.time.Clock()
FPS = 0.5
running = True
last_time = time.time()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    #image_surface = camera.get_image()
    image_bytes = camera.get_raw()
    if True: #result:
        #pygame.image.save(image_surface, "input.jpg")
        file_image = Image.frombytes("RGB", (1920, 1080), image_bytes) #
        file_image = Image.merge("RGB", file_image.split()[::-1])
        file_image.save(open("C://Users//tahir//Desktop//HANDS//data//TODELTE" + datetime.date.strftime(datetime.date.today(), "%Y%m%d") + "_" + time.strftime("%H%M%S", time.localtime()) + ".jpg", "wb"), bitmap_format='jpg')
        # display to screen
        screen.fill((0,0,0))
        #image_array_rgba = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGBA)
        #print(image_array.shape)
        screen.blit(pygame.image.fromstring(file_image.tobytes(), IMAGE_RESOLUTION, "RGB"), pygame.Rect(0, 0, 100, 100))
        time_ = time.time()
        screen.blit(pygame.font.Font("freesansbold.ttf", 15).render("FPS:" + str(round(1/(time_ - last_time), 2)), True, (0,0,0)), pygame.Rect(0, 25, 100, 100))
        last_time = time_
        pygame.display.flip()