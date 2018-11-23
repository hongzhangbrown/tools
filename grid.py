from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import multiprocessing
import os
import random
import subprocess
import sys
import time

class GridJobLuncher(object):

    @staticmethod
    def fetch_from_hdfs(hdfs_dir, local_dir=None):
        fnull = open(os.devnull, 'w')
        if local_dir:
            subprocess.call(['hdfs','dfs','-get', hdfs_dir, local_dir], stdout=fnull, stderr=fnull)
        else:
            subprocess.call(['hdfs','dfs','-get',hdfs_dir], stdout=fnull, stderr=fnull)

    @staticmethod
    def copy_to_hdfs(local_dir, hdfs_dir):
        if os.path.exists(local_dir):
            fnull = open(os.devnull,'w')
            subprocess.call(["hadoop","fs","-rm","-r","-skipTrash",hdfs_dir], stdout = fnull, stderr=fnull)
            subprocess.call(["hadoop","fs","-copyFromLocal", local_dir, hdfs_dir], stdout=fnull, stderr=fnull)

    @staticmethod
    def cleanup_dir(local_dir):
        if os.path.exists(local_dir):
            fnull = open(os.devnull, 'w')
            subprocess.call(["rm","-rf",local_dir], stdout=fnull, stderr=fnull)

    @staticmethod
    def debug_directory(dir):
        import socket
        print("on host {}".format(socket.gethostname()))
        for root,dirs, files in os.walk(dir,topdown=False):
            for name in files:
                print(os.path.join(root,name))
            for name in dirs:
                print(os.path.join(root,name))


    @staticmethod
    def start(argv):
        raise NotImplementedError()

    @classmethod
    def worker_main(cls,argv):

        print("worker_main")
        sys.argv = argv
        parser = argparse.ArgumentParser()
        parser.register("type","bool",lambda v:v.lower()== "true")
        parser.add_argument("--hdfs_src",type = str, help="input data dir on hdfs")
        parser.add_argument("--hdfs_dst",type=str,help="output data dir on hdfs")
        parser.add_argument("--log_dir",type=str,help="local logs directory")

        args,unknown = parser.parse_known_args()

        if args.hdfs_src is not None:
            print("---------------Fetch data from HDFS path {}-------------".format(args.hdfs_src))
            cls.fetch_from_hdfs("{}/*".format(args.hdfs_src))

        print("-----------------debug local files ------------------")
        cls.debug_directory(".")
        cls.start(argv)


        if args.hdfs_dst is not None:
            print("-----------persist model and logs to HDFS path: {}-------------".format(args.hdfs_dst))
            cls.copy_to_hdfs(args.log_dir,'{}/{}'.format(args.hdfs_dst,args.log_dir))

        print("--------------clean up local files-----------------------")
        cls.cleanup_dir(".")


    @classmethod
    def driver_main(cls,sc):
        parser = argparse.ArgumentParser()
        parser.add_argument("--num-executors",default=1, type=int,help="num of executors")

        args, unknown = parser.parse_known_args()
        num_executors = args.num_executors
        rdd = sc.parallelize([sys.argv for i in range(num_executors)],num_executors)
        rdd.map(cls.worker_main).collect()

    @classmethod
    def worker_eval_main(clscls,it,argv):
        raise NotImplementedError()



    @classmethod
    def driver_eval_main(cls,sc):

        parser = argparse.ArgumentParser()
        parser.add_argument("--inputs",required=True, type=str, help="preprocess input file on HDFS")
        parser.add_argument("--outputs",required=True,type=str,help="preprocess output file on HDFS")

        args,unknown = parser.parse_known_args()
        rdd = sc.textFile(args.inputs)

        argv = sys.argv
        print("argv:",argv)
        rdd.mapPartitions(lambda it:cls.worker_eval_main(it,argv)).saveAsTextFile(args.outputs)




