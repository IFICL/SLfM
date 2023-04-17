#!/bin/bash
{
mkdir -p checkpoints
prefix='checkpoints'
file_name='pretrained_models.tar.gz'

wget https://www.dropbox.com/s/6qfwfix7to1nzzl/slfm_pretrained_models.tar.gz?dl=0 -O $prefix/$file_name
cd checkpoints && tar -zxvf $file_name

}
