import numpy as np
from neurodynex.hopfield_network import network
import matplotlib.image as mpimg
import glob
import os
import cv2
from shutil import rmtree


def read_images(image_folder):
    src_folder = image_folder + '/*'
    image_list = glob.glob(src_folder)
    names = [os.path.basename(adr).split('.')[0] for adr in image_list]
    imgs = [];
    for src_image_adr in image_list:
        im = mpimg.imread(src_image_adr)
        im = cv2.resize(im, (32, 32))
        imgs.append(im)
    return imgs, names

def image_to_binary(image):
    col = np.reshape(image, (-1, 1))
    binary = np.zeros((col.shape[0], 8), dtype=int)
    for i in range(binary.shape[0]):
        b = [int(j) for j in bin(col[i][0])[2:]]
        binary[i, -len(b):] = b
        
    binary[binary == 0] = -1
    return binary


def binary_to_image(binary, size):
    binary[binary == -1] = 0
    p = 2 ** np.array([i for i in range(7, -1, -1)])
    col = np.sum(binary * p, axis=1)
    image = np.reshape(col, size)
    return image


def train_hopfield(imgs):
    bin_imgs = [image_to_binary(im) for im in imgs]
    hop_nets = []
    
    for i in range(8):
        patterns = []
        net = network.HopfieldNetwork(nr_neurons = 1024)
        for j in range(len(bin_imgs)):
            patterns.append(bin_imgs[j][:, i])
        net.store_patterns(patterns)
        hop_nets.append(net)
    return hop_nets


def test_hopfield(hop_nets, imgs):
    img_32 = [];
    bin_imgs = [image_to_binary(im) for im in imgs]
    
    final_img = np.zeros((1024, 8))
    for img in bin_imgs:
        for i in range(8):
            hop_nets[i].set_state_from_pattern(img[:, i])
            states = hop_nets[i].run_with_monitoring(nr_steps = 5)
            final_img[:, i] = states[-1]
        img_32.append(binary_to_image(final_img, (32, 32)))
    
    return img_32

def find_closest_subject(trn_imgs, trn_img_names, ret_imgs):
    match_names = []
    match_imgs = []
    for i in range(len(ret_imgs)):
        min_val = float('inf')
        for j in range(len(trn_imgs)):
            d = np.sum(np.abs(ret_imgs[i] - trn_imgs[j]))
            if d < min_val:
                min_val = d
                min_i = j
        match_names.append(trn_img_names[min_i])
        match_imgs.append(trn_imgs[min_i])
    return match_names, match_imgs


trn_imgs, trn_img_names = read_images('yaleface/TrainData/')
hop_nets = train_hopfield(trn_imgs)
tst_imgs, tst_img_names = read_images('yaleface/TestData/')
ret_imgs = test_hopfield(hop_nets, tst_imgs)

match_names, match_imgs = find_closest_subject(trn_imgs, trn_img_names, ret_imgs)

if os.path.exists('results'):
        rmtree('results')
        
os.mkdir('results')
for i, (trn, ret, match, mt_name, ac_name) in enumerate(zip(tst_imgs, ret_imgs, match_imgs, match_names, tst_img_names)):
    adr = 'results/' + str(i)
    os.mkdir(adr)
    mpimg.imsave(adr + '/actual_' + ac_name + '.bmp', trn, cmap='gray')
    mpimg.imsave(adr + '/retrived.bmp', ret, cmap='gray')
    mpimg.imsave(adr + '/matched_' + mt_name + '.bmp', match, cmap='gray')

s = np.zeros((70,))
for i in range(len(match_names)):
    s[i] = match_names[i] == tst_img_names[i]
    
acc = np.zeros((7,))
k = 0
for i in range(0, 60, 10):
    acc[k] = np.mean(s[i : i + 10])
    k += 1
    
all_acc = s.mean()
print('overall accuracy: {}'.format(all_acc))

for i in range(7):
    print('accuracy for class {} is {}'.format(i + 1, acc[i]))




