#!/bin/bash

# get sshd port
# sshport=$(lsof -i | grep sshd | awk '{print $9}' | sed s/\*://)

hostfile=${TRAIN_WORKSPACE}/hostfile
hostlist=$(cat $hostfile | awk '{print $1}' | xargs)
for host in ${hostlist[@]}; do
    if [ -n "$2" ]; then
        TARGET_DIR="$host:$2"
        echo "scp $1 to $TARGET_DIR"
        ssh $host "mkdir -p $TARGET_DIR"
        scp -r $1 $TARGET_DIR
    else
        #ssh $host "ls $PWD"
        #echo -e "${host},\c"
        #echo "${host}"
        echo "scp $1 to $host"
        ssh $host "mkdir -p $PWD"
        scp -r $1 ${host}:${PWD}/
    fi
done
