#!/usr/bin/env bash

mkdir -p ~/.opennre/pretrain/glove
wget -P ~/.opennre/pretrain/glove https://thunlp.oss-cn-qingdao.aliyuncs.com/opennre/pretrain/glove/glove.6B.50d_mat.npy
wget -P ~/.opennre/pretrain/glove https://thunlp.oss-cn-qingdao.aliyuncs.com/opennre/pretrain/glove/glove.6B.50d_word2id.json

mkdir -p ~/.opennre/benchmark/wiki80/
wget -P ~/.opennre/benchmark/wiki80/ https://thunlp.oss-cn-qingdao.aliyuncs.com/opennre/benchmark/wiki80/wiki80_rel2id.json
wget -P ~/.opennre/benchmark/wiki80/ https://thunlp.oss-cn-qingdao.aliyuncs.com/opennre/benchmark/wiki80/wiki80_train.txt
wget -P ~/.opennre/benchmark/wiki80/ https://thunlp.oss-cn-qingdao.aliyuncs.com/opennre/benchmark/wiki80/wiki80_val.txt
