from provider import CIFAR10
import sys
import argparse
import tensorflow as tf
import numpy as np

from helpers import *
import sys
import cPickle as pickle
import time

npop = 1200     # population size
sigma = 0.1    # noise standard deviation
alpha = 0.025  # learning rate
# alpha = 0.001  # learning rate
boxmin = 0
boxmax = 1
boxplus = (boxmin + boxmax) / 2.
boxmul = (boxmax - boxmin) / 2.
folder = './liclipadvImages/'
epsi = 0.015

flags = tf.app.flags
flags.DEFINE_string('data_set', 'cifar10',
                    """Choose between mnist, cifar10.""")
flags.DEFINE_string('model_name', 'resnet',
                    """Name of CNN model"""
                    """Choose between lenet and resnet.""")
flags.DEFINE_integer('resnet_n', 18,
                     """n for RESNET {3:20, 5:32, 7:44, 9:56, 18:110}.""")
flags.DEFINE_integer("embedding_at", 5,
                     """position of embedding [2]""")

# Training Mode
flags.DEFINE_boolean("is_train", False,
                     """True for training, False for testing [False]""")
flags.DEFINE_boolean("restore", False,
                     """True for restoring variables from the checkpoint_dir"""
                     """ [False]""")
flags.DEFINE_boolean("restore_inplace", True,
                     """True for restoring variables from the train_dir"""
                     """ [False]""")
flags.DEFINE_boolean("use_saved_images", False,
                     """True for testing with saved_images stored in"""
                     """ FLAGS.saved_dir [False]""")
flags.DEFINE_string('saved_data_dir', '/tmp/mnist_train',
                    """Directory where to load adversarial images for test.""")
flags.DEFINE_boolean("cascade", False,
                     """True for using saved iter_fgsm images for training"""
                     """ [False]""")
flags.DEFINE_boolean("ensemble", False,
                     """True for ensemble training"""
                     """ [False]""")
flags.DEFINE_string('saved_iter_fgsm_dir', '/tmp/mnist_train',
                    """Directory which contains iter_fgsm images"""
                    """ for training.""")
flags.DEFINE_boolean('log_device_placement', False,
                     """Whether to log device placement.""")
flags.DEFINE_boolean('add_noise_inference', False,
                     """True for adding random noise for inference.""")


# Define Directories
flags.DEFINE_string('data_dir', '../cifar10_data',
                    """Path to the CIFAR-10 data directory.""") ###################### change cifar10 dir
flags.DEFINE_string('eval_dir', '/tmp/mnist_eval',
                    """Directory where to write event logs.""")
flags.DEFINE_string('eval_data', 'test',
                    """Either 'test' or 'train_eval'.""")
flags.DEFINE_string('checkpoint_dir', '../../cascade_checkpoint/resnet110_cifar10_pivot_cascade',
                    """Directory where to read model checkpoints.""")
flags.DEFINE_string('train_dir', '/checkpoint/resnet110_cifar10_pivot',
                    """Directory where to write event logs """
                    """and checkpoint.""")

# Training Hyper Parameters
flags.DEFINE_integer('max_epochs', 240,
                     """Number of batches to run.""")
flags.DEFINE_integer('batch_size', 128,
                     """Number of images to process in a batch.""")
flags.DEFINE_integer('adver_batch_size', 64,
                     """Number of adversarial images to process in a batch.""")
flags.DEFINE_float('max_e', 16,
                   """Maximum magnitude for adversarial noise""")
flags.DEFINE_boolean('use_fp16', False,
                     """Train the model using fp16.""")

# Loss Parameters
flags.DEFINE_float("samebatch_loss_factor", 1.,
                   """samebatch loss factor [0.5]""")
flags.DEFINE_float("adver_loss_factor", 0.3,
                   """adver loss factor [0.3]""")
flags.DEFINE_float("similarity_loss_factor", 0.0,
                   """similarity loss factor [0.0001]""")
flags.DEFINE_float("pivot_loss_factor", 0.0,
                   """pivot loss factor [0.0001]""")
flags.DEFINE_string('distance', 'l2',
                    """Choose between l1 and l2.""")
flags.DEFINE_boolean('normalize_embedding', False,
                     """True for embedding normalization.""")

# Input preprocessing, adversarial training or not
flags.DEFINE_integer('norm_option', 0,
                     """0: per pixel, 1: per channel, 2: no normalization""")
flags.DEFINE_boolean("per_image_standard", False,
                     """True for per_image_standardization [False]""")
flags.DEFINE_boolean("rand_crop", False,
                     """True for random crop for inputs [True]""")
flags.DEFINE_boolean("adversarial", True,
                     """True for Adversarial training [True]""")
flags.DEFINE_integer('adver_option', 2,
                     """0: max_e, 1: uniform random, """
                     """2: truncated normal, 3: reverse truncated normal""")
flags.DEFINE_integer('sparsity', 100,
                     """represents the portion of non-zeros in adver_noise """)
flags.DEFINE_boolean("per_pixel_rand", False,
                     """True for per pixel random e for adversarial training"""
                     """ [False]""")
flags.DEFINE_string('method', 'step_ll',
                    """Choose between step_ll, step_fgsm, step_both and """
                    """ step_rand.""")

# For analysis
flags.DEFINE_boolean("save_adver_images", False,
                     """True for saving adversarial images during test."""
                     """Saved images can be used for transfer analysis.""")
flags.DEFINE_boolean("save_iter_fgsm_images", False,
                     """True for saving iter_fgsm adversarial images."""
                     """This is for cascade adversarial training.""")
flags.DEFINE_string("test_data_from", 'validation',
                    """Choose between train, validation and test""")
FLAGS = flags.FLAGS

from utils import *
import model

def main():

    parser = argparse.ArgumentParser()
#     parser.add_argument('--cifar-path', type=str, required=True,
#             help='path to the test_batch file from http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100)
    parser.add_argument('--debug', action='store_true')
    #parser.add_argument('--checkpoint_dir', type=str, default = './ref/resnet110_cifar10', help = 'path to checkpoint file')
    parser.add_argument('--checkpoint_dir', type=str, default = './checkpoint/resnet110_cifar10_pivot_cascade/', help = 'path to checkpoint file')
    #hardlist = [11, 27, 38, 43, 51, 93, 98, 122, 132, 133, 198] # hardlist for cascade pivot
    #hardlist = [67, 71, 100, 110, 140, 150, 156, 161, 197, 200, 234, 235, 252, 254, 258, 295, 297, 316, 329, 346, 354, 362, 395, 421, 424, 448, 492, 503, 536, 563, 575, 583, 584, 585, 594, 633, 657, 704, 706, 708, 712, 714, 731, 752, 755, 757, 777, 843, 846, 886, 890, 895, 917, 936, 938, 939, 949, 1047, 1052, 1054, 1061, 1083, 1085, 1133, 1211, 1233, 1243, 1254, 1265, 1356, 1444, 1452, 1490, 1494, 1497, 1585, 1599, 1633, 1638, 1712, 1789, 1836, 1847, 1853, 1859, 1880, 1917, 1943, 1954, 1970, 1972, 2010, 2024, 2025, 2051, 2108, 2122, 2134, 2173, 2195, 2240, 2271, 2288, 2303, 2307, 2323, 2369, 2373, 2385, 2390, 2412, 2438, 2449, 2500, 2546, 2572, 2575, 2613, 2627, 2649, 2665, 2674, 2676, 2735, 2745, 2748, 2785, 2792, 2800, 2818, 2825, 2834, 2852, 2894, 2933, 2934, 2945, 2985, 2996, 3003, 3012, 3044, 3061, 3074, 3095, 3102, 3112, 3114, 3119, 3133, 3186, 3187, 3199, 3201, 3226, 3246, 3262, 3269, 3270, 3294, 3300, 3371, 3436, 3445, 3465, 3468, 3471, 3473, 3487, 3498, 3499, 3506, 3529, 3538, 3546, 3553, 3581, 3636, 3644, 3668, 3676, 3681, 3689, 3692, 3744, 3759, 3793, 3797, 3806, 3813, 3830, 3835, 3851, 3874, 3876, 3878, 3880, 3885, 3886, 3888, 3902, 3914, 3937, 3946, 3973, 4004, 4029, 4047, 4068, 4069, 4075, 4098, 4103, 4127, 4150, 4153, 4197, 4198, 4279, 4283, 4308, 4317, 4346, 4356, 4372, 4379, 4380, 4397, 4443, 4473, 4477, 4496, 4497, 4511, 4512, 4566, 4591, 4598, 4612, 4653, 4657, 4665, 4670, 4677, 4679, 4700, 4717, 4765, 4772, 4789, 4792, 4799, 4828, 4834, 4877, 4883, 4908, 4911, 4916, 4929, 4935, 4944, 4947, 4995, 5012, 5034, 5036, 5052, 5055, 5089, 5095, 5110, 5158, 5169, 5178, 5183, 5223, 5237, 5250, 5260, 5318, 5323, 5325, 5334, 5344, 5364, 5383, 5399, 5411, 5431, 5434, 5443, 5449, 5479, 5496, 5520, 5550, 5554, 5556, 5575, 5581, 5589, 5615, 5649, 5663, 5668, 5676, 5687, 5688, 5709, 5742, 5764, 5780, 5812, 5844, 5871, 5887, 5920, 5932, 5945, 5951, 5953, 5957, 5959, 5987, 5993, 6057, 6066, 6084, 6085, 6087, 6093, 6114, 6170, 6194, 6200, 6216, 6222, 6252, 6292, 6304, 6327, 6362, 6363, 6370, 6382, 6387, 6396, 6403, 6407, 6414, 6415, 6422, 6446, 6451, 6466, 6469, 6496, 6552, 6556, 6563, 6589, 6614, 6673, 6682, 6720, 6734, 6743, 6747, 6763, 6766, 6771, 6780, 6785, 6817, 6819, 6834, 6850, 6891, 6900, 6909, 6923, 6929, 6937, 6957, 6963, 6988, 7040, 7072, 7087, 7094, 7100, 7109, 7152, 7153, 7160]
    # hardlist for cascade pivot 10000
    #hardlist = [11, 13, 21, 98, 103, 111, 198, 222, 235, 252, 297, 298, 362, 447, 498, 554, 582, 584, 604, 609, 759, 858, 864, 871, 896, 915, 918, 934, 949, 1010, 1078, 1229, 1234, 1238, 1288, 1347, 1378, 1410, 1473, 1490, 1497, 1517, 1540, 1566, 1591, 1596, 1627, 1651, 1677, 1694,1725, 1729, 1743, 1744, 1784, 1819, 1904, 1917, 1943, 2004, 2039, 2048, 2122, 2139, 2188, 2189, 2220, 2238, 2322, 2341, 2345, 2373, 2388, 2400, 2402, 2450, 2484, 2541, 2578, 2624, 2661, 2745, 2792, 2839, 2897,
#            2947, 2980, 2993, 3058, 3102, 3112, 3134, 3152, 3153, 3277, 3294, 3348, 3372, 3394, 3407, 3426, 3445, 3528, 3535, 3641, 3672, 3691, 3722, 3740, 3802, 3803, 3823, 3825, 3876, 3937, 3946, 3976, 3992, 4024, 4037, 4040, 4083, 4115, 4153, 4178, 4194, 4278, 4305, 4314, 4327, 4356, 4419, 4426, 4461, 4466, 4496, 4497, 4543, 4584, 4657, 4663, 4670, 4769, 4771, 4773, 4854, 4901, 4968, 5036, 5055, 5095, 5124, 5223, 5333, 5383, 5445, 5457, 5503, 5543, 5583, 5604, 5653, 5663, 5668, 5716,
#            5764, 5780, 5844, 5879, 5880, 5920, 5951, 5953, 5964, 5970, 5993, 6057, 6058, 6067, 6105, 6106, 6112, 6159, 6221, 6222, 6294, 6304, 6556, 6583, 6618, 6738, 6747, 6775, 6797, 6923, 6940, 6971, 6990, 7020, 7060, 7098, 7115, 7139, 7196, 7209, 7284, 7335, 7338, 7373, 7426, 7428, 7432, 7523, 7575, 7607, 7696, 7704, 7715, 7724, 7742, 7748, 7767, 7774, 7803, 7835, 7876,7886, 7922, 7925, 8042, 8064, 8088, 8089, 8146, 8210, 8235, 8259, 8311, 8453, 8485, 8522, 8537, 8608, 8643, 8693,
#            8750, 8756, 8795, 8800, 8812, 8909, 8975, 9000, 9008, 9017, 9063, 9089, 9097, 9151, 9175, 9291, 9356, 9458, 9539, 9565, 9572, 9662, 9719, 9800, 9802,  9841, 9882, 9890, 9899, 9904, 9936, 9946, 9951, 9954, 9980]
    hardlist = [11, 13, 21, 98, 111, 235, 252, 297, 298, 350, 362, 447, 490, 582, 584, 604, 759, 858, 864, 871, 896, 915, 934, 949]
    args = parser.parse_args()

    test_loss = 0
    correct = 0
    total = 0
    totalImages = 0
    succImages = 0
    faillist = []
    print('npop {} , sigma {}, alpha {}, runstep {}', npop, sigma, alpha,800)



    # set up TensorFlow session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    phase = tf.convert_to_tensor(False,dtype=tf.bool)
    input_xs = tf.placeholder(tf.float32, [None, 24,24, 3])
    _, real_logits = model.inference(input_xs, phase,
                                         is_for_adver=tf.constant(False))
    real_logits = tf.nn.softmax(real_logits)

    # initialize a model

    ckpt_path = tf.train.get_checkpoint_state(args.checkpoint_dir)
    optimistic_restore(sess, ckpt_path.model_checkpoint_path)
    # initialize an attack (it's a white box attack, and it's allowed to look
    # at the internals of the model in any way it wants)
    # attack = BPDA(sess, model, epsilon=model.threat_model.epsilon, debug=args.debug)
    # attack = Attack(sess, model.model, epsilon=model.threat_model.epsilon)

    # initialize a data provider for CIFAR10 images
    provider =  CIFAR10(FLAGS.data_dir +'/test_batch')


    start = 0
    end = 10000
    total = 0
    runsteplist = []
    start_time = time.time()

    for i in range(start, end):
        success = False
        if i not in hardlist:
            continue
        print('evaluating %d of [%d, %d)' % (i, start, end))
        inputs, targets= provider[i]
        inputs = inputs.reshape((1,32,32,3))
        inputs0 = copy_crop_images(inputs,24,True)
        modify = np.random.randn(1,3,32,32) * 0.01
        perturbpath ='perturb/cascade_'+str(i)+'.pkl'
        ##### thermometer encoding
        inputs0 = inputs0[0][0]
        logits = sess.run(real_logits, feed_dict={input_xs: [inputs0]})
        print(logits)
        if np.argmax(logits) != targets:
            print('skip the wrong example ', i)
            continue
        totalImages += 1

        for runstep in range(1500):
            Nsample = np.random.randn(npop, 3,32,32)

            modify_try = modify.repeat(npop,0) + sigma*Nsample

            newimg = inputs.transpose(0,3,1,2)

            inputimg = (newimg+modify_try)
            if runstep % 10 == 0:
#                 modify = pickle.load(open(perturbpath, 'rb'))
                realinputimg =(newimg+modify)
                realdist = realinputimg - newimg
                realclipdist = np.clip(realdist, -epsi, epsi)
                realclipinput = realclipdist + (newimg)
                realclipinput = np.clip(realclipinput, 0, 1)
#                 l2real =  np.sum((realclipinput - (np.tanh(newimg) * boxmul + boxplus))**2)**0.5
                #l2real =  np.abs(realclipinput - inputs.numpy())
                #print(inputs.shape)
                realclipinput = copy_crop_images(realclipinput.transpose(0,2,3,1), 24, True)
                realclipinput = realclipinput[0]
                outputsreal = sess.run(real_logits, feed_dict={input_xs: realclipinput})[0]
                print('probs:',np.sort(outputsreal)[-1:-4:-1])

                sys.stdout.flush()
#                 print('l2real: '+str(l2real.max()))
#                print(outputsreal)
                if (np.argmax(outputsreal) != targets) and (np.abs(realclipdist).max() <= epsi):
                    #outputpkl = open(perturbpath, 'wb')
                    print('save perturb to path: ',perturbpath)
                    #pickle.dump(modify,open(perturbpath, 'wb'), -1)
                    succImages += 1
                    success = True
                    runsteplist.append(runstep)
                    print('clipimage succImages: '+str(succImages)+'  totalImages: '+str(totalImages))
                    print('lirealsucc: '+str(realclipdist.max()))
                    sys.stdout.flush()
#                     imsave(folder+classes[targets[0]]+'_'+str("%06d" % batch_idx)+'.jpg',inputs.transpose(1,2,0))
                    break
            dist = inputimg - (newimg)
            clipdist = np.clip(dist, -epsi, epsi)
            clipinput = (clipdist + (newimg) ).reshape(npop,3,32,32)
            target_onehot =  np.zeros((1,10))


            target_onehot[0][targets]=1.
            clipinput = copy_crop_images(clipinput.transpose(0,2,3,1),24,True)
            clipinput = clipinput[0]
            clipinput = np.clip(clipinput,0,1)

            outputs = sess.run(real_logits, feed_dict={input_xs: clipinput})
            outputs = np.log(outputs)

            target_onehot = target_onehot.repeat(npop,0)



            real = (target_onehot * outputs).sum(1)
            other = ((1. - target_onehot) * outputs - target_onehot * 10000.).max(1)[0]

            loss1 = np.clip(real - other, 0.,1000)

            Reward = 0.5 * loss1
#             Reward = l2dist

            Reward = -Reward

            A = (Reward - np.mean(Reward)) / (np.std(Reward)+1e-7)

            modify = modify + (alpha/(npop*sigma)) * ((np.dot(Nsample.reshape(npop,-1).T, A)).reshape(3,32,32))
        if(i%1000==0):
            print('runsteps:',runsteplist)
            sys.stdout.flush()

        if not success:
            faillist.append(i)
            print('faillist: ',faillist)
        sys.stdout.flush()
    print(faillist)
    end_time = time.time()
    print('all time :', end_time - start_time)
    success_rate = succImages/float(totalImages)
    #np.savez('runstep',runsteplist)
    sys.stdout.flush()


#     print('attack success rate: %.2f%% (over %d data points)' % (success_rate*100, args.end-args.start))

if __name__ == '__main__':
    main()
