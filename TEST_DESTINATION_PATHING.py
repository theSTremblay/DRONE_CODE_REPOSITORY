"""\
Demo app for the ARDrone.
This simple application allows to control the drone and see the drone's video
stream.
Copyright (c) 2011 Bastian Venthur
The license and distribution terms for this file may be
found in the file LICENSE in this distribution.
"""

# THis demo is to prove the ability of the drone to move towards a destination, the parameters will bbe a point to move towards and the drone must do its best to get there.
import pygame
import subprocess
from pydrone import libardrone
import cv2

file = open('canRead.txt', 'w+')
canRead = file.write('0')
file.close()

def is_square(apositiveint):
  x = apositiveint // 2
  seen = set([x])
  while x * x != apositiveint:
    x = (x + (apositiveint // x)) // 2
    if x in seen: return False
    seen.add(x)
  return True

def crop2(infile,imgheight,imgwidth,imgheight2, imgwidth2):
    #im = Image.open(infile)
    #imgwidth, imgheight = im.size
    b= []
    for i in range(imgheight//imgheight2):
        for j in range(imgwidth//imgwidth2):
            crop_img = infile[i*imgheight2:(i+1)*imgheight2, j*imgwidth2:(j+1)*imgwidth2]
            b.append(crop_img)
    return b

def Pipe_Access_Initial():
    camera_process = subprocess.Popen(["python3", "DRONE_LENET_PREDICTION.py"],
                                      stdout=subprocess.PIPE)
    done_correct = camera_process.communicate()[0]

    collision = float(done_correct)
def Pipe_Access(message_filepath):
    camera_process = subprocess.Popen(["python3", "DRONE_LENET_PREDICTION.py"],
                                      stdout=subprocess.PIPE)
    done_correct = camera_process.communicate(message_filepath)[0]

    collision = float(done_correct)

def Folder_Deletion(path):
    import os, shutil
    folder = path
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

def crop(im, k):
    k2 = 9

    boolean_var = is_square(k)
    #im = Image.open(input)
    # 290 , 640
    imgheight, imgwidth, chan = im.shape

    imgheight = int(imgheight - (imgheight % sqrt(k)))
    imgwidth = int(imgwidth - (imgwidth % sqrt(k)))

    imgheight2 = int(int(imgheight - (imgheight % sqrt(k))) / 3)
    imgwidth2 = int(int(imgwidth - (imgwidth % sqrt(k))) / 3)

    rim = cv2.resize(im, (imgwidth, imgheight))

    M = rim.shape[0] // 2
    N = rim.shape[1] // 2

    # rimg = Image.fromarray(rim, 'L')

    # 96, 213 dimensions of tile

    tiles = crop2(rim, imgheight, imgwidth, imgheight2, imgwidth2)
    return tiles



if __name__ == '__main__':
    pygame.init()
    W, H = 320, 240
    screen = pygame.display.set_mode((W, H))
    drone = libardrone.ARDrone()
    clock = pygame.time.Clock()
    # test this for phi values which gives rotation
    test = drone.navdata[0]
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYUP:
                drone.hover()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    drone.reset()
                    running = False
                # takeoff / land
                elif event.key == pygame.K_RETURN:
                    drone.takeoff()
                elif event.key == pygame.K_SPACE:
                    drone.land()
                # emergency
                elif event.key == pygame.K_BACKSPACE:
                    drone.reset()
                # forward / backward
                elif event.key == pygame.K_w:
                    drone.move_forward()
                elif event.key == pygame.K_s:
                    drone.move_backward()
                # left / right
                elif event.key == pygame.K_a:
                    drone.move_left()
                elif event.key == pygame.K_d:
                    drone.move_right()
                # up / down
                elif event.key == pygame.K_UP:
                    drone.move_up()
                elif event.key == pygame.K_DOWN:
                    drone.move_down()
                # turn left / turn right
                elif event.key == pygame.K_LEFT:
                    drone.turn_left()
                elif event.key == pygame.K_RIGHT:
                    drone.turn_right()
                # speed
                elif event.key == pygame.K_1:
                    drone.speed = 0.1
                elif event.key == pygame.K_2:
                    drone.speed = 0.2
                elif event.key == pygame.K_3:
                    drone.speed = 0.3
                elif event.key == pygame.K_4:
                    drone.speed = 0.4
                elif event.key == pygame.K_5:
                    drone.speed = 0.5
                elif event.key == pygame.K_6:
                    drone.speed = 0.6
                elif event.key == pygame.K_7:
                    drone.speed = 0.7
                elif event.key == pygame.K_8:
                    drone.speed = 0.8
                elif event.key == pygame.K_9:
                    drone.speed = 0.9
                elif event.key == pygame.K_0:
                    drone.speed = 1.0

        try:
            try:

                image_grid_array = []
                size_of_grid = 9
                grid_length = 3

                try:
                    frame = None

                    while running2:
                        # get current frame of video
                        running2, frame = cam.read()
                        if running2:
                            cv2.imshow('frame', frame)
                            image_grid_array.extend(crop(frame, size_of_grid))

                            Ps = 0
                            Pl = .5
                            Pr = .2

                            i = 1

                            folder_name = "Drone_Tesselations"

                            for image in image_grid_array:

                                cv2.imwrite("test_img.jpeg", image)

                                canRead = '0'
                                # Test for I/O file close permissions
                                number_variable = str(20)
                                camera_process = subprocess.Popen(["python3", "DRONE_LENET_PREDICTION.py"],
                                                                  stdout=subprocess.PIPE)
                                done_correct = camera_process.communicate()[0]

                                collision = float(done_correct)

                                print("Drone Read inactive")

                                if i % grid_length < (size_of_grid / 2 + 1):
                                    Pl += collision
                                elif i % grid_length == (size_of_grid / 2 + 1):
                                    Ps += collision
                                else:
                                    Pr += collision

                            event = 1

                            # forward / backward
                            if Pr and Pl > Ps:

                                drone.move_forward()
                            else:
                                if Pr <= Pl:

                                    drone.move_left()
                                else:
                                    drone.move_right()
                                    # turn left / turn right
                        else:
                            # error reading frame
                            print 'error reading video feed'
                except Exception as e:
                    print(e)

                hud_color = (255, 0, 0) if drone.navdata.get('drone_state', dict()).get('emergency_mask', 1) else (
                10, 10, 255)
                bat = drone.navdata.get(0, dict()).get('battery', 0)
                f = pygame.font.Font(None, 20)
                hud = f.render('Battery: %i%%' % bat, True, hud_color)
                screen.blit(hud, (10, 10))
            except Exception as e:
                print(str(e))
                pass
            hud_color = (10, 10, 255)
            if drone.navdata.get('drone_state', dict()).get('emergency_mask', 1):
                hud_color = (255, 0, 0)
            bat = drone.navdata.get(0, dict()).get('battery', 0)
            f = pygame.font.Font(None, 20)
            hud = f.render('Battery: %i%%' % bat, True, hud_color)
            screen.blit(surface, (0, 0))
            screen.blit(hud, (10, 10))
        except:
            pass

        pygame.display.flip()
        clock.tick()
        pygame.display.set_caption("FPS: %.2f" % clock.get_fps())

    print "Shutting down...",
    drone.halt()
    print "Ok."