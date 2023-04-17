#!/bin/bash

{
source ~/.bashrc
conda activate SLfM

CUDA=3
batch_size=96
num_workers=8
vision_backbone='resnet18'
audio_backbone='resnet18'
loss='L1'
input_nc=2
output_nc=2
N_source=1
N_view=2
pretext_flag='--mono2binaural'
shared_flag=''
remove_input=''
model=''
generative_loss_ratio=0
geometric_loss_ratio=1
binaural_loss_ratio=1
symmetric_loss_ratio=0
synthetic_loss_ratio=0
activation='tanh'
audiobase_path='data/AI-Habitat/data-split/FMA'


# --------------------------- HM3D-4View-Rotation: LibriSpeech ----------------------------------- #

EXP='HM3D-4View-Rotation/SLfM-Geometric/EXP1.0-LibriSpeech-M2B-N=1-3View-Freeze-PIT-FinerRot-g5b1s1'
setting='slfm_hm3d'
pretext_flag='--mono2binaural --freeze_audio --freeze_camera --freeze_generative --sound_permutation'
shared_flag='--finer_rotation'
generative_loss_ratio=0
geometric_loss_ratio=5
binaural_loss_ratio=1
symmetric_loss_ratio=1
N_view=3
N_source=1
epochs=20
model='HM3D-4View-Rotation/SLfM-Pretext/EXP1.0-LibriSpeech-M2B-N=1-3View-CondLength=2.55/Pretext-binauralization/pretext_best.pth.tar'
audiobase_path='data/AI-Habitat/data-split/LibriSpeech'





# -------------------------- Training -------------------------- #
CUDA_VISIBLE_DEVICES=$CUDA python main.py --exp=$EXP --epochs=$epochs --setting=$setting --vision_backbone=$vision_backbone --audio_backbone=$audio_backbone --batch_size=$batch_size --num_workers=$num_workers --save_step=1 --valid_step=1 --lr=0.0001 --optim='AdamW' --repeat=1 --schedule='cos' --unet_input_nc=$input_nc --unet_output_nc=$output_nc --loss_type=$loss --n_source=$N_source --n_view=$N_view --online_render --add_geometric --imagenet_pretrain --color_jitter --generative_loss_ratio=$generative_loss_ratio  --geometric_loss_ratio=$geometric_loss_ratio  --binaural_loss_ratio=$binaural_loss_ratio --symmetric_loss_ratio=$symmetric_loss_ratio --activation=$activation --audiobase_path=$audiobase_path $pretext_flag $shared_flag $remove_input --weights=$model

# -------------------------- Evaluation -------------------------- #
N_view=2

mkdir -p ./results/${EXP}

CUDA_VISIBLE_DEVICES=$CUDA python evaluation/evaluate_angle.py --exp=$EXP --setting=$setting --vision_backbone=$vision_backbone --audio_backbone=$audio_backbone --batch_size=$batch_size --num_workers=$num_workers --n_source=$N_source --n_view=$N_view --online_render  --add_geometric --ssl_flag --activation=$activation --audiobase_path=$audiobase_path $shared_flag --input='audio' --weights="${EXP}/audio_best.pth.tar" > "./results/${EXP}/audio_res.txt"

CUDA_VISIBLE_DEVICES=$CUDA python evaluation/evaluate_angle.py --exp=$EXP --setting=$setting --vision_backbone=$vision_backbone --audio_backbone=$audio_backbone --batch_size=$batch_size --num_workers=$num_workers --n_source=$N_source --n_view=$N_view --online_render  --add_geometric --activation=$activation --audiobase_path=$audiobase_path $shared_flag --input='vision' --weights="${EXP}/vision_best.pth.tar" > "./results/${EXP}/vision_res.txt"


}