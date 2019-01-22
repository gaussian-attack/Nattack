import time
import pickle
import robustml
from robustml_model import Thermometer, LEVELS
from discretization_utils import discretize_uniform
import sys
import argparse
import tensorflow as tf
import numpy as np
from helpers import *



npop = 500    # population size
sigma = 0.1    # noise standard deviation
alpha = 0.008  # learning rate
# alpha = 0.001  # learning rate
boxmin = 0
boxmax = 1
boxplus = (boxmin + boxmax) / 2.
boxmul = (boxmax - boxmin) / 2.
folder = './liclipadvImages/'
epsi = 0.031

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--cifar-path', type=str, default='../cifar10_data/test_batch',
            help='path to the test_batch file from http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    test_loss = 0
    correct = 0
    total = 0
    totalImages = 0
    succImages = 0
    faillist = []



    # set up TensorFlow session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)


    # initialize a model
    model = Thermometer(sess)

    print(model.threat_model.targeted)
    # initialize an attack (it's a white box attack, and it's allowed to look
    # at the internals of the model in any way it wants)
    # attack = BPDA(sess, model, epsilon=model.threat_model.epsilon, debug=args.debug)
    # attack = Attack(sess, model.model, epsilon=model.threat_model.epsilon)

    # initialize a data provider for CIFAR-10 images
    provider = robustml.provider.CIFAR10(args.cifar_path)
    input_xs = tf.placeholder(tf.float32, [None, 32, 32, 3])
    hardlist = [ 1412, 1432, 1449, 1465, 1466, 1468, 1473, 1483, 1485, 1494, 1500, 1511, 1517, 1528, 1560, 1562, 1563, 1567, 1585, 1591, 1599, 1614, 1617, 1627, 1629, 1650, 1651,      1655, 1665, 1677, 1692, 1694, 1699, 1706, 1707, 1734, 1743, 1744, 1748, 1776, 1781, 1784, 1792, 1814, 1815, 1825, 1839, 1851, 1853, 1856, 1864, 1875, 1881, 1893, 1901, 1904, 1912, 1913,      1927, 1930, 1949, 1968, 1970, 1974, 1980, 1987, 1988, 1992, 2004, 2012, 2020, 2031, 2040, 2052, 2071, 2084, 2089, 2097, 2138, 2143, 2148, 2158, 2164, 2179, 2184, 2188, 2189, 2200, 2210,      2218, 2228, 2234, 2238, 2249, 2258, 2268, 2274, 2277, 2290, 2326, 2332, 2333, 2344, 2345, 2368, 2373, 2382, 2399, 2400, 2402, 2410, 2445, 2450, 2456, 2457, 2462, 2470, 2478, 2484, 2485,      2492, 2513, 2537, 2541, 2561, 2583, 2601, 2602, 2633, 2645, 2647, 2661, 2663, 2667, 2684, 2693, 2698, 2710, 2732, 2754, 2757, 2761, 2792, 2794, 2799, 2820, 2827, 2830, 2850, 2885, 2887,      2896, 2908, 2936, 3001, 3024, 3040, 3045, 3058, 3066, 3074, 3081, 3089, 3099, 3103, 3104, 3126, 3128, 3134, 3141, 3144, 3152, 3153, 3157, 3162, 3176, 3177, 3187, 3198, 3210, 3212, 3215, 3217, 3223, 3225,     3241, 3269, 3286, 3287, 3294, 3307, 3317, 3337, 3362, 3368, 3370, 3372, 3374, 3377, 3394, 3409, 3432, 3435, 3438, 3445, 3454, 3466, 3480, 3485, 3492, 3515, 3532, 3542, 3545, 3547, 3556,      3558, 3582, 3590, 3598, 3604, 3621, 3633, 3634, 3638, 3642, 3661, 3663, 3672, 3675, 3688, 3691, 3699, 3713, 3714, 3722, 3723, 3730, 3732, 3742, 3745, 3756, 3760, 3784, 3794, 3802, 3805,      3818, 3823, 3831, 3836, 3837, 3841, 3846, 3852, 3860, 3908, 3919, 3926, 3946, 3958, 3982, 4018, 4039, 4040, 4060, 4072, 4105, 4113, 4120, 4123, 4128, 4131, 4153, 4178, 4181, 4183, 4218,      4231, 4232, 4233, 4234, 4235, 4257, 4263, 4278, 4284, 4297, 4311, 4314, 4318, 4323, 4324, 4327, 4330, 4336, 4338, 4339, 4356, 4359, 4372, 4378, 4379, 4390, 4395, 4396, 4417, 4418, 4419,      4433, 4434, 4451, 4453, 4466, 4475, 4480, 4496, 4502, 4511, 4518, 4534, 4543, 4546, 4563, 4591, 4594, 4602, 4631, 4638, 4657, 4658, 4663, 4680, 4698, 4702, 4711, 4713, 4727, 4728, 4738,      4743, 4753, 4759, 4763, 4766, 4767, 4768, 4773, 4774, 4777, 4781, 4800, 4835, 4836, 4840, 4841, 4842, 4848, 4853, 4891, 4896, 4905, 4912, 4915, 4968, 4971, 4992, 5000, 5015, 5027, 5032,      5036, 5054, 5055, 5064, 5083, 5087, 5091, 5094, 5097, 5114, 5137, 5157, 5170, 5178, 5181, 5209, 5223, 5226, 5247, 5248, 5251, 5259, 5263, 5264, 5312, 5315, 5323, 5326, 5336, 5350, 5353,      5360, 5382, 5397, 5400, 5423, 5440, 5442, 5447, 5448, 5449, 5451, 5470, 5475, 5485, 5494, 5543, 5545, 5553, 5568, 5572, 5574, 5583, 5584, 5585, 5601, 5606, 5622, 5624, 5645, 5668, 5683,      5712, 5719, 5720, 5723, 5727, 5728, 5735, 5737, 5746, 5768, 5772, 5780, 5801, 5818, 5823, 5828, 5838, 5845, 5846, 5895, 5918, 5920, 5922, 5947, 5951, 5953, 5964, 5967, 5980, 5998, 6002,      6019, 6045, 6048, 6057, 6058, 6059, 6099, 6104, 6110, 6117, 6120, 6133, 6149, 6159, 6173, 6176, 6188, 6192, 6198, 6206, 6207, 6211, 6221, 6222, 6223, 6226, 6232, 6233, 6234, 6248, 6266,      6274, 6289, 6323, 6325, 6331, 6350, 6363, 6377, 6390, 6404, 6410, 6435, 6449, 6450, 6471, 6503, 6509, 6510, 6527, 6546, 6555, 6563, 6571, 6585, 6600, 6618, 6639, 6643, 6649, 6672, 6675,      6676, 6711, 6738, 6744, 6749, 6787, 6789, 6821, 6838, 6840, 6846, 6847, 6864, 6882, 6886, 6891, 6925, 6927, 6932, 6945, 6956, 6971, 6984, 7013, 7020, 7049, 7058, 7063, 7071, 7075, 7086,      7098, 7105, 7115, 7116, 7118, 7122, 7136, 7139, 7158, 7166, 7192, 7208, 7225, 7234, 7276, 7282, 7293, 7301, 7303, 7335, 7339, 7342, 7345, 7359, 7372, 7385, 7405, 7415, 7424, 7432, 7438,      7448, 7477, 7479, 7483, 7500, 7501, 7512, 7516, 7523, 7525, 7532, 7540, 7563, 7564, 7565, 7581, 7607, 7647, 7656, 7678, 7681, 7684, 7691, 7700, 7703, 7710, 7715, 7719, 7748, 7752, 7753,      7759, 7764, 7767, 7774, 7779, 7811, 7814, 7825, 7835, 7836, 7837, 7840, 7844, 7854, 7880, 7936, 7944, 7992, 8000, 8004, 8005, 8017, 8061, 8089, 8094, 8101, 8116, 8144, 8147, 8149, 8155,      8178, 8188, 8207, 8211, 8217, 8221, 8225, 8228, 8235, 8288, 8293, 8330, 8355, 8360, 8365, 8393, 8417, 8426, 8434, 8453, 8455, 8458, 8459, 8485, 8506, 8510, 8515, 8517, 8522, 8537, 8539,      8552, 8560, 8571, 8579, 8593, 8594, 8595, 8621, 8639, 8643, 8675, 8676, 8693, 8697, 8709, 8718, 8745, 8747, 8756, 8763, 8768, 8770, 8772, 8788, 8789, 8793, 8800, 8812, 8820, 8829, 8839,      8845, 8846, 8871, 8881, 8903, 8909, 8927, 8929, 8931, 8954, 8975, 8980, 8982, 8988, 9000, 9014, 9017, 9020, 9032, 9045, 9055, 9063, 9072, 9075, 9077, 9091, 9115, 9138, 9152, 9156, 9164,      9166, 9173, 9180, 9181, 9199, 9202, 9211, 9220, 9236, 9241, 9243, 9244, 9245, 9249, 9256, 9270, 9284, 9312, 9323, 9340, 9354, 9367, 9372, 9379, 9382, 9392, 9394, 9395, 9399, 9408, 9409,      9410, 9424, 9432, 9444, 9455, 9458, 9482, 9496, 9504, 9516, 9519, 9530, 9544, 9545, 9564, 9565, 9566, 9577, 9581, 9588, 9592, 9599, 9614, 9618, 9623, 9628, 9660, 9674, 9677, 9698, 9702,      9724, 9754, 9766, 9800, 9802, 9814, 9817, 9822, 9828, 9848, 9854, 9871, 9876, 9890, 9899, 9905, 9916, 9920, 9921, 9935, 9936, 9938, 9951, 9980]
    start = 0
    end = 10000
    total = 0
    uniform = discretize_uniform(input_xs, levels=LEVELS, thermometer=True)
    real_logits = tf.nn.softmax(model.model(uniform))
    successlist = []
    printlist = []
    outpath = 'perturb/theradv_'
    start_time = time.time()


    for i in range(start, end):
        if i not in hardlist:
            continue
        success = False
        print('evaluating %d of [%d, %d)' % (i, start, end), file=sys.stderr)
        inputs, targets= provider[i]
        modify = np.random.randn(1,3,32,32) * 0.001
        outpath_pkl = outpath + str(i)+'.pkl'
        ##### thermometer encoding

        logits = sess.run(real_logits,feed_dict={input_xs: [inputs]})
        if np.argmax(logits) != targets:
            print('skip the wrong example ', i)
            continue
        totalImages += 1

        for runstep in range(1000):
            Nsample = np.random.randn(npop, 3,32,32)

            modify_try = modify.repeat(npop,0) + sigma*Nsample

            newimg = torch_arctanh((inputs-boxplus) / boxmul).transpose(2,0,1)
            

#             inputimg = np.tanh(newimg+modify_try) * boxmul + boxplus
            if runstep % 10 == 0:
#                 realinputimg = np.tanh(newimg+modify) * boxmul + boxplus
#                 realdist = realinputimg - (np.tanh(newimg) * boxmul + boxplus)
                realclipdist = np.clip(modify, -epsi, epsi)
                realclipinput = realclipdist + inputs.transpose(2,0,1)
                realclipinput = np.clip(realclipinput,0,1)
#                 l2real =  np.sum((realclipinput - inputs.transpose(2,0,1 )**2)**0.5)
                #l2real =  np.abs(realclipinput - inputs.numpy())
                print(inputs.shape)
                outputsreal = sess.run(real_logits, feed_dict={input_xs: realclipinput.transpose(0,2,3,1)})
                print(outputsreal)

                print(np.abs(realclipdist).max())
#                 print('l2real: '+str(l2real.max()))
                print(outputsreal)
                if (np.argmax(outputsreal) != targets) and (np.abs(realclipdist).max() <= epsi):
                    succImages += 1
                    success = True
                    print('clipimage succImages: '+str(succImages)+'  totalImages: '+str(totalImages))
                    print('lirealsucc: '+str(realclipdist.max()))
                    successlist.append(i)
                    printlist.append(runstep)
                    pickle.dump(modify, open(outpath_pkl,'wb'),-1)
#                     imsave(folder+classes[targets[0]]+'_'+str("%06d" % batch_idx)+'.jpg',inputs.transpose(1,2,0))

                    break
#             dist = inputimg - inputs.transpose(2,0,1 )
            clipdist = np.clip(modify_try, -epsi, epsi)
            
            clipinput = (clipdist + inputs.transpose(2,0,1)).reshape(npop,3,32,32)
            clipinput = np.clip(clipinput,0,1)
            target_onehot =  np.zeros((1,10))


            target_onehot[0][targets]=1.


            outputs = sess.run(real_logits, feed_dict={input_xs: clipinput.transpose(0,2,3,1)})

            target_onehot = target_onehot.repeat(npop,0)



            real = (target_onehot * outputs).sum(1)
            other = ((1. - target_onehot) * outputs - target_onehot * 10000.).max(1)[0]

            loss1 = np.clip(real - other, 0.,1000)

            Reward = 0.5 * loss1
#             Reward = l2dist

            Reward = -Reward

            A = (Reward - np.mean(Reward)) / (np.std(Reward)+1e-7)


            modify = modify + (alpha/(npop*sigma)) * ((np.dot(Nsample.reshape(npop,-1).T, A)).reshape(3,32,32))
        if not success:
            faillist.append(i)
            print('failed:',faillist)
        else:
            print('successed:',successlist)
    print(faillist)
    print('all_time: ',time.time()-start_time)
    success_rate = succImages/float(totalImages)
    print('succ rate', success_rate)
    np.savez('ther_adv_runstep',printlist)



    print('attack success rate: %.2f%% (over %d data points)' % (success_rate*100, args.end-args.start))

if __name__ == '__main__':
    main()
