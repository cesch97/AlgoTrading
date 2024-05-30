#! /bin/sh

rm -rf /data/AlgoTrading/strategies
scp -r ec2-3:/home/ec2-user/efs-1/Data/strategies /data/AlgoTrading

rm -rf /data/AlgoTrading/trad_systems
scp -r ec2-3:/home/ec2-user/efs-1/Data/trad_systems /data/AlgoTrading

rm -rf /data/AlgoTrading/evals
scp -r ec2-3:/home/ec2-user/efs-1/Data/evals /data/AlgoTrading

rm -rf /data/AlgoTrading/cleaning
scp -r ec2-3:/home/ec2-user/efs-1/Data/cleaning /data/AlgoTrading

rm -rf /data/AlgoTrading/trad_logs
scp -r ec2-3:/home/ec2-user/efs-1/Data/trad_logs /data/AlgoTrading
