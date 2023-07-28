#!/bin/sh
#SBATCH --nodes=1
#SBATCH --qos=long
#SBATCH --time=10-0
##SBATCH --nodelist=gpu003
#
#srun -o aug.out -t 5-0 -G4 -n1 -J aug python train_multiple_gpu_augmentation.py
#python /home/spotter5/cnn_mapping/train_multiple_gpu_augmentation.py
#python /home/spotter5/cnn_mapping/train_multiple_gpu_effb7.py
#python /home/spotter5/cnn_mapping/unet_3d.py
#python /home/spotter5/cnn_mapping/effb7_just_dnbr.py
#python /home/spotter5/cnn_mapping/train_nbac_sent_1985.py
#python /home/spotter5/cnn_mapping/train_nbac_1985.py
#python /home/spotter5/cnn_mapping/train_nbac_threshold_1985.py
#python /home/spotter5/cnn_mapping/split_to_chunk_sent.py
#python /home/spotter5/cnn_mapping/apply_dnbr_threshold.py
#python /home/spotter5/cnn_mapping/train_nbac_sent_1985.py
#python /home/spotter5/cnn_mapping/train_nbac_sent_1985.py
#python /home/spotter5/cnn_mapping/train_nbac_sent_threshold_1985.py
#python /home/spotter5/cnn_mapping/train_mtbs_nbac_sent_threshold_1985.py
#python /home/spotter5/cnn_mapping/train_mtbs_nbac_sent_threshold_1985_schedule.py
#python /home/spotter5/cnn_mapping/train_all_ca_nbac_no_thresh.py
#python /home/spotter5/cnn_mapping/train_nbac_sent_threshold_1985_aug.py
#python /home/spotter5/cnn_mapping/train_all_ca_nbac.py
#python /home/spotter5/cnn_mapping/train_nbac_sent_084_1985.py
#python /home/spotter5/cnn_mapping/train_nbac_no_7_1985.py
#python /home/spotter5/cnn_mapping/train_nbac_sent_1985_negative.py
#python /home/spotter5/cnn_mapping/train_nbac_sent_threshold_1985_aug_contrast.py
#python /home/spotter5/cnn_mapping/train_nbac_modis.py
#python /home/spotter5/cnn_mapping/train_nbac_modis_0.py
#python /home/spotter5/cnn_mapping/train_l8_c2_128.py
#python /home/spotter5/cnn_mapping/train_l8_c2_256.py
#python /home/spotter5/cnn_mapping/train_l8_c2_0_128.py
#python /home/spotter5/cnn_mapping/train_l8_c2_512.py
#python /home/spotter5/cnn_mapping/train_l8_c2_0_256.py
#python /home/spotter5/cnn_mapping/train_l8_c2_0_512.py
#python /home/spotter5/cnn_mapping/train_ak_ca_no_threshold.py
#python /home/spotter5/cnn_mapping/train_all_0_128_global_norm.py
#python /home/spotter5/cnn_mapping/train_all_0_128_just_3_standardize.py
#python /home/spotter5/cnn_mapping/train_all_0_128_just_3_norm_cutoff.py
#python /home/spotter5/cnn_mapping/train_all_0_128_just_3_global_std.py
#python /home/spotter5/cnn_mapping/train_all_0_128_just_3_keep_direction.py
#python /home/spotter5/cnn_mapping/train_all_0_512_just_3_norm_cutoff.py
#python /home/spotter5/cnn_mapping/train_all_0_256_just_3_norm_cutoff.py
#python /home/spotter5/cnn_mapping/train_all_0_128_just_3_norm_cutoff.py

# python /home/spotter5/cnn_mapping/v3/train_128_0_just_dnbr.py

python /home/spotter5/cnn_mapping/v3/train_128_0_just_dnbr_proj.py

# python /home/spotter5/cnn_mapping/v3/train_128_0_just_dnbr_proj_w_mtbs.py



# python /home/spotter5/cnn_mapping/v3/train_128_0_just_dnbr_negative.py


# python /home/spotter5/cnn_mapping/v2/train_128_0_just_dnbr.py
# python /home/spotter5/cnn_mapping/v2/train_128_0_just_dnbr_wgs_crop.py

# python /home/spotter5/cnn_mapping/v3/train_128_0_modis.py

# python /home/spotter5/cnn_mapping/v3/train_128_0_modis_aug.py


# python /home/spotter5/cnn_mapping/v3/train_128_0_just_VI.py
#python /home/spotter5/cnn_mapping/v2/train_256_0_just_dnbr.py
#python /home/spotter5/cnn_mapping/v2/train_128_0_just_dnbr_neg_test.py
#python /home/spotter5/cnn_mapping/v2/train_128_0_just_VI_negative.py
# python /home/spotter5/cnn_mapping/v2/train_128_0_just_dnbr_w_negative.py
#python /home/spotter5/cnn_mapping/v2/train_128_0_just_dnbr_w_negative_single_norm.py
# python /home/spotter5/cnn_mapping/v2/train_128_0_just_dnbr_wgs.py
#python /home/spotter5/cnn_mapping/v2/train_128_0_just_dnbr_w_negative_weight.py
#python /home/spotter5/cnn_mapping/v2/train_128_0_just_dnbr_w_negative_aug.py
#python /home/spotter5/cnn_mapping/v2/train_128_0_just_VI_negative_aug.py
