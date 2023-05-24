from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from progress.bar import Bar
import time
import torch

from models.model import create_model, load_model
# from utils.debugger import Debugger

class BaseDetector(object):
    def __init__(self, args):
        if args.gpus[0] >= 0:
            args.device = torch.device('cuda')
        else:
            args.device = torch.device('cpu')
        self.cop_only = args.cop_only
        if not self.cop_only:
            print('Creating model...')
            self.model = create_model(args)
            self.model = load_model(self.model, args.load_model, device = args.device)
            self.model = self.model.to(args.device)
            self.model.eval()
        
        self.args = args
        self.pause = True

    def run(self, graph):
        net_time, post_time = 0, 0
        tot_time = 0
        # debugger = Debugger(dataset=self.args.dataset, ipynb=(self.args.debug == 3),
        #                     theme=self.args.debugger_theme)
        start_time = time.time()
        
        graph = graph.to(self.args.device)

        output, forward_time = self.process(graph, return_time=True)
        net_time += forward_time - start_time
        # if self.opt.debug >= 2:
        #     self.debug(debugger, graph, output)
        post_process_time = time.time()
        post_time += post_process_time - forward_time

        tot_time += post_process_time - start_time

        if self.args.debug == 1:
            # self.show_results(debugger, graph, output, self.args.debug_dir)
            self.show_results(graph, output, self.args.debug_dir)

        return {'results': output, 'tot': tot_time, 'net': net_time,
                'post': post_time}

    def process(self, graph, return_time=False):

        with torch.no_grad():
            output = self.model(graph)[0]
        
        forward_time = time.time()

        
        if return_time:
            return output, forward_time
        else:
            return output



    def pre_process(self, graph, meta=None):
        raise NotImplementedError

    def post_process(self, output, graph):
        if self.args.predict_diff:
            output = output / self.args.diff_multiplier + graph.c1.to(self.args.device)
            output = torch.clamp(output, min=-1., max=1.)
        else:
            output = torch.clamp(output, min=0., max=1.)
        return output

    def merge_outputs(self, detections):
        raise NotImplementedError

    def debug(self, debugger, graph, dets, output, scale=1):
        raise NotImplementedError

    # def show_results(self, debugger, image, results):
    def show_results(self, graph, output, path):
        # c1
        file_name = os.path.join(path, '{}_c1.png'.format(graph.name))
        x = graph.gt[:, 0].cpu().numpy()
        y = graph.c1[:, 0].cpu().numpy()
        
        plt.scatter(x,y)

        plt.title('{} - gt vs c1'.format(graph.name))
        plt.xlabel("gt")
        plt.ylabel("c1")

        plt.savefig(file_name)
        plt.close()

        if not self.cop_only:
            # pred
            file_name = os.path.join(path, '{}_pred.png'.format(graph.name))
            x = graph.gt[:, 0].cpu().numpy()
            y = output[:, 0].cpu().numpy()

            plt.scatter(x,y)

            plt.title('{} - gt vs pred'.format(graph.name))
            plt.xlabel("gt")
            plt.ylabel("pred")

            plt.savefig(file_name)
            plt.close()

    