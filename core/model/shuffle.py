import os
from os.path import join
import random

# Directories
TRAIN_PATH = '../../data/train'
TEST_PATH = '../../data/test'

# Image filenames gathering
rock_filenames = [f for f in os.listdir(join(TRAIN_PATH, 'rock'))]
paper_filenames = [f for f in os.listdir(join(TRAIN_PATH, 'paper'))]
scissor_filenames = [f for f in os.listdir(join(TRAIN_PATH, 'scissor'))]

# Shuffling
random.shuffle(rock_filenames)
random.shuffle(paper_filenames)
random.shuffle(scissor_filenames)

# Test size
test_percentage = 0.2
class_size = 250
test_size = int(class_size * test_percentage)

# Rock transfer
for filename in rock_filenames[:test_size]:
    train_rock_path = join(TRAIN_PATH, 'rock')
    os.system('move ' + join(train_rock_path, filename) + ' ' + join(TEST_PATH, 'rock'))
   
# Paper transfer
for filename in paper_filenames[:test_size]:
    train_rock_path = join(TRAIN_PATH, 'paper')
    os.system('move ' + join(train_rock_path, filename) + ' ' + join(TEST_PATH, 'paper'))
    
# Scissor transfer
for filename in scissor_filenames[:test_size]:
    train_rock_path = join(TRAIN_PATH, 'scissor')
    os.system('move ' + join(train_rock_path, filename) + ' ' + join(TEST_PATH, 'scissor'))
   
print('Success')