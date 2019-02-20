#!/usr/bin/env python

# Copyright (c) 2011 Bastian Venthur
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


"""Demo app for the AR.Drone.

This simple application allows to control the drone and see the drone's video
stream.
"""


import pygame

import pygame



from pydrone import libardrone

import cv2
import numpy as np
from math import sqrt
import time
from filelock import FILELOCK

def write_to_filelock(file, written_value, read_or_write):
    canRead = 0
    with FileLock(file):
        if read_or_write == 'r':
            file = open('canRead.txt', 'r+')
            canRead = file.readline()
            file.close()
        else:
            try:
                file = open(file, 'w+')
                file.write(written_value)
                file.close()
                canRead = 1
            except Exception as e:
                print(e)
                canRead = 0
        # work with the file as it is now locked
        print("Lock acquired.")
    return canRead


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



def main():
    pygame.init()

    cam = cv2.VideoCapture('tcp://192.168.1.1:5555')
    running2 = True
    W2, H2 = 320,240
    W, H = 640, 290
    screen = pygame.display.set_mode((W, H))
    drone = libardrone.ARDrone()
    clock = pygame.time.Clock()

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

                        canRead = write_to_filelock('droneRead.txt', "1", 'w')

                        for image in image_grid_array:

                            cv2.imwrite("test_img.jpeg", image)

                            canRead = 0
                            # Test for I/O file close permissions
                            while canRead == 0:

                                canRead = write_to_filelock('canRead.txt', "1", 'w')


                                # To run this in python 3.5 we will need to run a script that uses the "env"
                                # wrapper
                                # Once the pipeline is established, have the thread continuousely generate new samples
                                #
                                # can definitely recursion this size_of grid /2 + 1
                            canRead = 0

                            while (canRead == 0):
                                canRead = write_to_filelock('DroneRead.txt', "0", 'r')
                            canRead = 0

                            while (canRead == 0):
                                try:

                                    canRead = write_to_filelock('out.txt', "0", 'r')
                                    collision = float(canRead)
                                except:
                                    collision = 0
                                    print("COLLISION: -1")
                            canRead = 0
                            while (canRead == 0):
                                canRead = write_to_filelock('DroneRead.txt', "0", 'w')

                            print("Drone Read active")

                            if i % grid_length < (size_of_grid / 2 + 1):
                                Pl += collision
                            elif i % grid_length == (size_of_grid / 2 + 1):
                                Ps += collision
                            else:
                                Pr += collision

                        event = 1

                        print("we got to the good part: all images from grid read")
                        # Will need to comment out the other parts

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



            hud_color = (255, 0, 0) if drone.navdata.get('drone_state', dict()).get('emergency_mask', 1) else (10, 10, 255)
            bat = drone.navdata.get(0, dict()).get('battery', 0)
            f = pygame.font.Font(None, 20)
            hud = f.render('Battery: %i%%' % bat, True, hud_color)
            screen.blit(hud, (10, 10))
        except Exception as e:
            print(str(e))
            pass

        pygame.display.flip()
        clock.tick(50)
        pygame.display.set_caption("FPS: %.2f" % clock.get_fps())

    print "Shutting down...",
    drone.halt()
    print "Ok."

if __name__ == '__main__':
    main()
