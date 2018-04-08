@echo off
echo Creating leveldb...
if exist cifar10-leveldb (
    rmdir cifar10-leveldb /s /q
 ) else (
 	mkdir cifar10-leveldb
 )
SET DATA="./cifar-10-batches-bin"
SET GLOG_logtostderr=1
"convert_cifar_data.exe" %DATA% ./cifar10-leveldb
SET GLOG_logtostderr=1
echo Computing image mean...
"compute_image_mean.exe" ./cifar10-leveldb/cifar-train-leveldb mean.binaryproto
echo Done.
pause