#!/bin/bash
scp $@ scuiaa@cluster1.math.ust.hk:./MachineLearning
/usr/bin/expect <<EOD
expect "password"
send "dPao7g9mn"
