#!/bin/bash
{
source ~/.bashrc
conda activate SLfM

CUDA=0,1
batch_size=96
num_workers=16
vision_backbone='resnet18'
audio_backbone='resnet18'
loss='L1'
input_nc=2
output_nc=2
N_source=1
N_view=2
pretext_flag='--mono2binaural'
shared_flag=''

# select audio database changing audiobase_path variable
# audiobase_path='data/AI-Habitat/data-split/FMA'
audiobase_path='data/AI-Habitat/data-split/LibriSpeech'
remove_input=''


# ---------------------------- hm3d-4view-rotation: LibriSpeech M2B --------------------------------- #

EXP='HM3D-4View-Rotation/SLfM-Pretext/EXP1.0-LibriSpeech-M2B-N=1-3View-CondLength=2.55'
batch_size=64
pretext_flag='--mono2binaural'
setting='slfm_hm3d'
shared_flag='--cond_clip_length=2.55'
audiobase_path='data/AI-Habitat/data-split/LibriSpeech'
N_source=1
N_view=3


# EXP='HM3D-4View-Rotation/SLfM-Pretext/EXP1.1-LibriSpeech-M2B-N=2(dominant)-3View-CondLength=2.55'
# batch_size=64
# pretext_flag='--mono2binaural'
# setting='slfm_hm3d'
# shared_flag='--cond_clip_length=2.55 --with_dominant_sound'
# audiobase_path='data/AI-Habitat/data-split/LibriSpeech'
# N_source=2
# num_workers=24
# N_view=3


# ---------------------------- hm3d-4view-rotation: LibriSpeech L2R --------------------------------- #
# EXP='HM3D-4View-Rotation/SLfM-Pretext/EXP2.0-LibriSpeech-L2R-N=1-3View-CondLength=2.55'
# batch_size=64
# pretext_flag=''
# setting='slfm_hm3d'
# shared_flag='--cond_clip_length=2.55'
# audiobase_path='data/AI-Habitat/data-split/LibriSpeech'
# N_source=1
# N_view=3


# EXP='HM3D-4View-Rotation/SLfM-Pretext/EXP2.1-LibriSpeech-L2R-N=2(dominant)-3View-CondLength=2.55'
# batch_size=64
# pretext_flag=''
# setting='slfm_hm3d'
# shared_flag='--cond_clip_length=2.55  --with_dominant_sound'
# audiobase_path='data/AI-Habitat/data-split/LibriSpeech'
# N_source=2
# num_workers=16
# N_view=3


# -------------------------- Training: Pretext -------------------------- #
echo "Start training pretext task...";
pretext_EXP="${EXP}/Pretext-binauralization"
epoch=500
lr=0.0001

CUDA_VISIBLE_DEVICES=$CUDA python main.py --exp=$pretext_EXP --epochs=$epoch --setting=$setting --vision_backbone=$vision_backbone --audio_backbone=$audio_backbone --batch_size=$batch_size --num_workers=$num_workers --save_step=1 --valid_step=1 --lr=$lr --optim='AdamW' --repeat=1 --schedule='cos' --unet_input_nc=$input_nc --unet_output_nc=$output_nc --loss_type=$loss --n_source=$N_source --n_view=$N_view --online_render --color_jitter --imagenet_pretrain --audiobase_path=$audiobase_path $pretext_flag $shared_flag $remove_input --eval



# -------------------------- Training: Downstream (including evaluation)-------------------------- #
batch_size=96

echo "Start training camera pose downstream task...";
camera_EXP="${EXP}/Downstream-CamRot-LinCls"
lr=0.0001
epoch=100

CUDA_VISIBLE_DEVICES=$CUDA python main_camera.py --exp=$camera_EXP --epochs=$epoch --setting=$setting --vision_backbone=$vision_backbone --batch_size=$batch_size --num_workers=$num_workers --save_step=1 --valid_step=1 --lr=$lr --optim='AdamW' --repeat=1 --schedule='cos' --n_source=$N_source --n_view=2 --time_sync --not_load_audio --color_jitter --freeze_camera --audiobase_path=$audiobase_path --weights_vision="${pretext_EXP}/vision_best.pth.tar"  $shared_flag --eval



echo "Start training sound localization downstream task...";
audio_EXP="${EXP}/Downstream-AudLoc-LinCls"
lr=0.001
epoch=100

CUDA_VISIBLE_DEVICES=$CUDA python main_audio.py --exp=$audio_EXP --epochs=$epoch --setting=$setting --audio_backbone=$audio_backbone --batch_size=$batch_size --num_workers=$num_workers --save_step=1 --valid_step=1 --lr=$lr --optim='AdamW' --repeat=1 --schedule='cos' --n_source=$N_source --n_view=1 --time_sync --online_render --freeze_audio --ssl_flag --audiobase_path=$audiobase_path --weights_audio="${pretext_EXP}/audio_best.pth.tar" --azimuth_loss_type='classification' $shared_flag --eval




}