#!/bin/bash
aws s3 sync /home/ec2-user/efs-1/AlgoTrading s3://s3-buck-1/AlgoTrading --delete
aws s3 sync /home/ec2-user/efs-1/Data s3://s3-buck-1/Data --delete
