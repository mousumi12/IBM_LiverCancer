'''
 * @author [Mousumi Roy]
 * @email [mousumi.roy@stonybrook.edu]
 * @desc [description]
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

SEED=0 # set set to allow reproducing runs
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.set_random_seed(SEED)

import os, shutil
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="7" #model will be trained on GPU 0

from model import UNet
from utils import dice_coef
from loader import dataLoader
from utils import VIS, mean_IU
# configure args
from opts import *
from opts import dataset_mean, dataset_std # set them in opts

# save and compute metrics
vis = VIS(save_path=opt.checkpoint_path)

# configuration session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


''' Users define data loader (with train and test) '''
img_shape = [opt.imSize, opt.imSize]
train_generator, train_samples = dataLoader(opt.data_path+'/train/', opt.batch_size,img_shape, mean=dataset_mean, std=dataset_std)
test_generator, test_samples = dataLoader(opt.data_path+'/val/', 1,  img_shape, train_mode=False,mean=dataset_mean, std=dataset_std)

opt.iter_epoch = int(train_samples) 
# define input holders
label = tf.placeholder(tf.int32, shape=[None]+img_shape)
# define model
with tf.name_scope('unet'):
    model = UNet().create_model(img_shape=img_shape+[3], num_class=opt.num_class)
    img = model.input
    pred = model.output
# define loss
with tf.name_scope('cross_entropy'):
    cross_entropy_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=pred))
# define optimizer
global_step = tf.Variable(0, name='global_step', trainable=False)
with tf.name_scope('learning_rate'):
    learning_rate = tf.train.exponential_decay(opt.learning_rate, global_step,
                                           opt.iter_epoch, opt.lr_decay, staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss, global_step=global_step)

# compute dice score for simple evaluation during training
# with tf.name_scope('dice_eval'):
#     dice_evaluator = tf.reduce_mean(dice_coef(label, pred))

''' Tensorboard visualization '''
# cleanup pervious info
if opt.load_from_checkpoint == '':
    cf = os.listdir(opt.checkpoint_path)
    for item in cf: 
        if 'event' in item: 
            os.remove(os.path.join(opt.checkpoint_path, item))
# define summary for tensorboard
tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)
tf.summary.scalar('learning_rate', learning_rate)
summary_merged = tf.summary.merge_all()
# define saver
train_writer = tf.summary.FileWriter(opt.checkpoint_path, sess.graph)
saver = tf.train.Saver() # must be added in the end

''' Main '''
tot_iter = opt.iter_epoch * opt.epoch
init_op = tf.global_variables_initializer()
sess.run(init_op)

with sess.as_default():
    # restore from a checkpoint if exists
    # the name_scope can not change 
    if opt.load_from_checkpoint != '':
        try:
            saver.restore(sess, opt.load_from_checkpoint)
            print ('--> load from checkpoint '+opt.load_from_checkpoint)
        except:
                print ('unable to load checkpoint ...' + str(e))
    # debug
    start = global_step.eval()
    for it in range(start, tot_iter):
        if it % opt.iter_epoch == 0 or it == start:
            
            saver.save(sess, opt.checkpoint_path+'model', global_step=global_step)
            print ('save a checkpoint at '+ opt.checkpoint_path+'model-'+str(it))
            print ('start testing {} samples...'.format(test_samples))
            for ti in range(test_samples):
                x_batch, y_batch = next(test_generator)
                # tensorflow wants a different tensor order
                feed_dict = {   
                                img: x_batch,
                                label: y_batch,
                            }
                loss, pred_logits = sess.run([cross_entropy_loss, pred], feed_dict=feed_dict)
                pred_map_batch = np.argmax(pred_logits, axis=3)
                # import pdb; pdb.set_trace()
                for pred_map, y in zip(pred_map_batch, y_batch):
                    #print(y)
                    #print(pred_map)
                    score = vis.add_sample(pred_map, y)
                    #print(score)
            vis.compute_scores(suffix=it)
        
        x_batch, y_batch = next(train_generator)
        #print(y_batch)
        feed_dict = {   img: x_batch,
                        label: y_batch
                    }
        _, loss, summary, lr, pred_logits = sess.run([train_step, 
                                    cross_entropy_loss, 
                                    summary_merged,
                                    learning_rate,
                                    pred
                                    ], feed_dict=feed_dict)
        global_step.assign(it).eval()
        train_writer.add_summary(summary, it)
        
        pred_map = np.argmax(pred_logits[0], axis=2)
        score, _ = mean_IU(pred_map, y_batch[0])

       
        if it % 20 == 0 : 
            print ('[iter %d, epoch %.3f]: lr=%f loss=%f, mean_IU=%f' % (it, float(it)/opt.iter_epoch, lr, loss, score))
        
