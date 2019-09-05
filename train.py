#Author by Cqy2019
import argparse
import logging
import torch
import torch.nn as nn
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from model import Image_Model, Text_Model
from dataset import IPAR_TC_12, Iteration
from utils import *
from optim import get_optimizer
from tensorboardX import SummaryWriter

def get_parser():
    '''
    Generate a parameters parse
    '''
    parser = argparse.ArgumentParser(description='Cross modal retrieve')
    parser.add_argument('--dump_path', type=str, default='./dumped/',
                        help='Experiment dump path')
    parser.add_argument('--exp_name', type=str, default='',
                        help='Experiment name')
    parser.add_argument('--exp_id', type=str, default='',
                        help='Experiment ID')
    parser.add_argument('--dataset_name', type=str, default='ipar_tc_12',
                        help='only support ipar_tc_12 currently')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size for training')
    parser.add_argument('--root', type=str, default='/home/disk1/chengqinyuan/datasets',
                        help='location of data')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='epochs for training')
    parser.add_argument('--eval_only', type=bool_flag, default=False,
                        help='decide to train or evaluate')
    parser.add_argument('--optimizer', type=str, default='adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.001',
                        help='choose which optimizer to use')
    parser.add_argument('--clip_grad_norm', type=float, default=5,
                        help='Clip gradients norm (0 to disable)')
    parser.add_argument('--code_size', type=int, default=32,
                        help='size of binary code')
    parser.add_argument('--load_model', type=str, default='',
                        help='location of saved model')
    parser.add_argument('--triplet_margin', type=int, default=8,
                        help='margin when calculate triplet loss')
    parser.add_argument('--lang', type=str, default='en',
                        help='choose language of the text')
    parser.add_argument('--bertmodel_dir', type=str, default='/home/disk1/chengqinyuan/pretrain_bert/bert-base-uncased-pytorch_model.bin',
                        help='bert pretrain model location')
    
    return parser

def main(params):

    logger = initialize_exp(params)
    writer = SummaryWriter(os.path.join(params.dump_path, 'runs'))

    logger.info('loading %s dataset' % params.dataset_name)
    dataset = IPAR_TC_12

    #select querys and database
    indexes = [i for i in range(19999)]
    random.shuffle(indexes)
    test_indexes = indexes[:2000]
    database_indexes = indexes[2000:]

    dataloaders = {
        'train': Iteration(params, dataset(params.root, if_train=True, lang=params.lang)),
        'test': Iteration(params, dataset(params.root, if_train=False, lang=params.lang, sample_index=test_indexes)),
        'database': Iteration(params, dataset(params.root, if_train=False, lang=params.lang, sample_index=database_indexes))
    }

    #create neural network
    Image_encoder = Image_Model(params=params)
    Text_encoder = Text_Model(params=params)
    Image_encoder.cuda()
    Text_encoder.cuda()

    #create optimizer
    image_optimizer = get_optimizer(Image_encoder.parameters(), params.optimizer)
    text_optimizer = get_optimizer(Text_encoder.parameters(), params.optimizer)
    # optimizer = torch.optim.Adam(training_network.parameters())
    # optimizer = torch.optim.SGD(training_network.parameters(), lr=0.001, momentum=0.9)

    if params.eval_only is False:
        logger.info('Start Training')
        max_map = 0
        for epoch in range(params.epochs):
            Image_encoder.train()
            Text_encoder.train()
            total_batches = len(dataloaders['train'])
            logger.info('============ Starting epoch %i ... ============' % epoch)
            count = 0
            total_loss_value = 0
            total_loss_i_i = 0
            total_loss_t_t = 0
            total_loss_i_t = 0
            total_loss_t_i = 0

            for index, (images, tokens, segment, mask, labels) in enumerate(dataloaders['train']):
                image_optimizer.zero_grad()
                text_optimizer.zero_grad()
                image_embeddings = Image_encoder(images)
                text_embeddings = Text_encoder(tokens, segment, mask)
                triplet_loss_i_i, triplet_loss_t_t, triplet_loss_i_t, triplet_loss_t_i = triplet_hashing_loss(image_embeddings, text_embeddings, labels, margin=params.triplet_margin)
                logger.info('Batch %i/%i: loss_image_image: %f\t||\tloss_text_text: %f\t||\tloss_image_text: %f\t||\tloss_text_image: %f' % (index + 1, 
                total_batches, triplet_loss_i_i.item(), triplet_loss_t_t.item(), triplet_loss_i_t.item(), triplet_loss_t_i.item()))
                total_loss = triplet_loss_i_i + triplet_loss_i_t + triplet_loss_t_i + triplet_loss_i_t
                total_loss_value += total_loss.item()
                total_loss_i_i += triplet_loss_i_i.item()
                total_loss_t_t += triplet_loss_t_t.item()
                total_loss_i_t += triplet_loss_i_t.item()
                total_loss_t_i += triplet_loss_t_i.item()

                count +=1 
                # writer.add_scalar('train_image_image', triplet_loss.item(), index + epoch * params.batch_size)
                # triplet_loss_i_i.backward(retain_graph=True)
                # image_optimizer.step()
                # text_optimizer.step()
                # triplet_loss_t_t.backward(retain_graph=True)
                # image_optimizer.step()
                # text_optimizer.step()
                # triplet_loss_i_t.backward(retain_graph=True)
                # image_optimizer.step()
                # text_optimizer.step()
                # triplet_loss_t_i.backward(retain_graph=False)
                total_loss.backward()
                image_optimizer.step()
                text_optimizer.step()

            logger.info('============ End of epoch %i ============' % epoch)
            
            if epoch % 10 == 0 and epoch != 0:
                
                avg_mAP = eval_all(dataloaders['database'], dataloaders['test'], Image_encoder, Text_encoder, params, logger)
                
                if avg_mAP > max_map:
                    max_map = avg_mAP
                    save_model(Image_encoder, Text_encoder, params.save_best)
                save_model(Image_encoder, Text_encoder, params.save_path)
            logger.info('\nEpoch %i: total_avg_loss: %f||\tavg loss_i_i: %f||\tavg loss_t_t: %f||\tavg loss_i_t: %f||\tavg loss_t_i: %f\n' % (epoch, total_loss_value / count, total_loss_i_i / count, total_loss_t_t / count, total_loss_i_t / count, total_loss_t_i / count))
    
    if params.eval_only:
        logger.info('Start Testing')
        assert os.path.isfile(params.load_model)
        load_model(Image_encoder, Text_encoder, params.load_model)
        avg_mAP = eval_all(dataloaders['database'], dataloaders['test'], Image_encoder, Text_encoder, params, logger)

    writer.close()

if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()
    main(params)