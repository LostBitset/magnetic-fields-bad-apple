#!/bin/bash

ffmpeg -r 30 -i outframes/BadAppleBField_%d.jpg -i BadApple.mp4 -c:v libx264 -c:a aac -strict -2 -shortest output.mp4

