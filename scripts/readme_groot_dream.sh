# This script covers the whole story for groot dream.
##### step 1: generate synthetic videos
# world model inference, save to xxx folder as mp4 files
# install lapa env
cd /home/users/yufanh/LAPA/laq
conda create -n laq python=3.10 -y
conda activate laq
cd laq
pip install -e .
apt update
apt install -y libgl1
# save to individual png files, modify the path. Used for training LAPA
python data_process_medbot.py
accelerate launch train_sthv2.py
# extract latent codes directly from mp4 files
python inference_sthv2_clean.py --video_folder /home/projects/healthcareeng_monai/datasets/medbot_pick_handover_121x1280x704_24fps_singleview/cam_high_images_1010 \
--output_folder /home/projects/healthcareeng_monai/datasets/medbot_pick_handover_121x1280x704_24fps_singleview/cam_high_images_1010_lapa

python inference_sthv2_clean.py --video_folder /home/projects/healthcareeng_monai/datasets/medbot_pick_handover_121x1280x704_24fps_singleview/cam_high_images_initial_frames_aug \
--output_folder /home/projects/healthcareeng_monai/datasets/medbot_pick_handover_121x1280x704_24fps_singleview/cam_high_images_infer_aug_lapa


python inference_sthv2_clean.py --video_folder /home/projects/healthcareeng_monai/datasets/medbot_pick_handover_121x1280x704_24fps_singleview/cam_high_images_initial_frames_aug \
--output_folder /home/projects/healthcareeng_monai/datasets/medbot_pick_handover_121x1280x704_24fps_singleview/cam_high_images_infer_aug_lapa_originallaq \
--laq_checkpoint ./vae.100000/model.safetensors

##### step 2: convert data to lerobot format.
## install lerobot
cd /home/users/yufanh/Isaac-GR00T/
cd lerobot
conda create -y -n lerobot python=3.10
conda activate lerobot
apt update
apt install -y libgl1
pip install simplejpeg
conda install ffmpeg=7.1.1 -c conda-forge
pip install -e .
cd ../scripts
# optional, split training and testing data
python create_train_test_split.py \
    --folder-a /home/projects/healthcareeng_monai/datasets/medbot_pick_handover \
    --folder-b /home/projects/healthcareeng_monai/datasets/medbot_pick_handover_new_1010 \
    --output medbot_split.json
### convert kinematic data. Do it twice if splitting training/testing
# modify repo_id and data_dir in the script
python convert_data_medbot.py \
    --split_file medbot_split.json \
    --splits train1 train2 \
    --repo_id /home/projects/healthcareeng_monai/datasets/medbot_pick_handover_54combined_lerobot

python convert_data_medbot.py \
    --split_file medbot_split.json \
    --splits test2 \
    --repo_id /home/projects/healthcareeng_monai/datasets/medbot_pick_handover_new_1010_test2_lerobot

# copy a modality file to the dataset
cp /home/projects/healthcareeng_monai/datasets/medbot_pick_handover_lerobot/meta/modality.json /home/projects/healthcareeng_monai/datasets/medbot_pick_handover_new_1010_lerobot/meta/
vim /lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/datasets/medbot_pick_handover_new_1010_lerobot/meta/modality.json

## convert lapa to lerobot. the video name is "image.top" which differs from kinematics but it won't matter if we match groot data config with image.top
# convert the lapa code datasets into lerobot. Copy the codebooks from laq to here.
cd /home/users/yufanh/Isaac-GR00T/scripts
python convert_lapa_to_lerobot.py \
--data_dir /home/projects/healthcareeng_monai/datasets/medbot_pick_handover_121x1280x704_24fps_singleview/cam_high_images_1010_lapa \
--repo_id /home/projects/healthcareeng_monai/datasets/medbot_pick_handover_new_1010_lerobot_offset1 \
--task_prompt "The left arm of the surgical robot is picking up a needle over a red rubber pad and handing it over to the right arm." \
--codebook_path 'vae.4000_codebooks.json'
# copy a modality file to the dataset
cp /home/projects/healthcareeng_monai/datasets/medbot_pick_handover_121x1280x704_24fps_singleview_lerobot_offset1/meta/modality.json /home/projects/healthcareeng_monai/datasets/medbot_pick_handover_new_1010_lerobot_offset1/meta/


##### step3 run baseline groot
cd /home/users/yufanh/Isaac-GR00T
conda activate gr00t
apt update
apt install -y libgl1
pip install -e .[base]

# run baseline model, change data_config MedBotdataconfig max_action_dim to 32
python scripts/gr00t_finetune.py --dataset-path /home/projects/healthcareeng_monai/datasets/medbot_pick_handover_lerobot/ --num-gpus 8 --video-backend torchvision_av --data-config medbot --output-dir ./medbot_tunev_singleview --batch_size 16


##### step4 run lapa model
# run lapa model, change data config max_action_dim to 512
python scripts/gr00t_finetune.py \
--dataset-path /home/projects/healthcareeng_monai/datasets/medbot_pick_handover_lerobot/ /home/projects/healthcareeng_monai/datasets/medbot_pick_handover_new_1010_lerobot_offset1 \
--batch_size 16 --num-gpus 8 --video-backend torchvision_av --action_dim 512 --data-config medbot lapa \
--output-dir ./medbot_joint10k_1010lapa_only --embodiment_tag new_embodiment lapa --max_steps 10000
# copy the meta file 
cp medbot_joint10k/checkpoint-10000/experiment_cfg/metadata.json medbot_joint10k_1010lapa/checkpoint-10000/experiment_cfg/metadata.json 

##### step 5 eval mse
python scripts/eval_policy.py --plot \
    --dataset-path /home/projects/healthcareeng_monai/datasets/medbot_pick_handover_new_1010_lerobot/ \
    --embodiment-tag new_embodiment \
    --data-config medbot \
    --save_plot_path ./medbot_joint10k_1010lapa/eval_new1010.png \
    --model_path ./medbot_joint10k_1010lapa/checkpoint-10000 \
    --action_horizon 16 \
    --trajs 11 \
    --video-backend torchvision_av \
    --steps 400

    python scripts/eval_policy.py --plot \
    --dataset-path /home/projects/healthcareeng_monai/datasets/medbot_pick_handover_lerobot_test/ \
    --embodiment-tag new_embodiment \
    --data-config medbot \
    --save_plot_path ./medbot_joint10k_aug_originallaq/eval.png \
    --model_path ./medbot_joint10k_aug_originallaq/checkpoint-10000 \
    --action_horizon 16 \
    --trajs 5 \
    --video-backend torchvision_av \
    --steps 400