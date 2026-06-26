python train.py --dataset /home/kzh/dataset/N3DV --scenes coffee_martini --configs arguments/n3dv/default.py --compre_config fsd_gop60.yaml --coarse_postfix default_lang --postfix default_lang --compres_thres 28.5
echo "coffee_martini Done"

python train.py --dataset /home/kzh/dataset/N3DV --scenes cook_spinach --configs arguments/n3dv/default.py --compre_config fsd_gop60.yaml --coarse_postfix default_lang --postfix default_lang --compres_thres 32.9
echo "cook_spinach Done"

python train.py --dataset /home/kzh/dataset/N3DV --scenes cut_roasted_beef --configs arguments/n3dv/default.py --compre_config fsd_gop60.yaml --coarse_postfix default_lang --postfix default_lang --compres_thres 32.5
echo "cut_roasted_beef Done"

python train.py --dataset /home/kzh/dataset/N3DV --scenes flame_salmon --configs arguments/n3dv/default.py --compre_config fsd_gop60.yaml --coarse_postfix default_lang --postfix default_lang --compres_thres 28.5
echo "flame_salmon Done"

python train.py --dataset /home/kzh/dataset/N3DV --scenes flame_steak --configs arguments/n3dv/default.py --compre_config fsd_gop60.yaml --coarse_postfix default_lang --postfix default_lang --compres_thres 32
echo "flame_steak Done"

python train.py --dataset /home/kzh/dataset/N3DV --scenes sear_steak --configs arguments/n3dv/default.py --compre_config fsd_gop60.yaml --coarse_postfix default_lang --postfix default_lang --compres_thres 33.5
echo "sear_steak Done"

# gop 30   --gop 30 --gopids 0 1 2 3 4 5 6 7 8 9
# gop 100  --gop 100 --gopids 0 1 2

#
#python train.py --dataset /home/kzh/dataset/MeetRoom --scenes discussion --configs arguments/meeting/default.py --compre_config fsd_gop60.yaml --postfix default --compres_thres 27.3
#echo "discussion Done"

#python train.py --dataset /home/kzh/dataset/MeetRoom --scenes trimming --configs arguments/meeting/default.py --compre_config fsd_gop60.yaml --postfix default --compres_thres 27.3
#echo "trimming Done"

#python train.py --dataset /home/kzh/dataset/MeetRoom --scenes vrheadset --configs arguments/meeting/vrheadset.py --compre_config fsd_gop60.yaml --postfix default --compres_thres 26.1
#echo "vrheadset Done"
