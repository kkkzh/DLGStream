

python train.py --dataset /home/kzh/dataset/N3DV --work_path /home/kzh/output/4DLangSplat/hac --scenes coffee_martini --configs arguments/n3dv/default_hac_1.py --compre_config hac_gop60.yaml --coarse_postfix default_lang --postfix default_lang --type hac --language
echo "coffee_martini Done"

python train.py --dataset /home/kzh/dataset/N3DV --work_path /home/kzh/output/4DLangSplat/hac --scenes cook_spinach --configs arguments/n3dv/default_hac.py --compre_config hac_gop60.yaml --coarse_postfix default_lang --postfix default_lang --type hac --language
echo "cook_spinach Done"

python train.py --dataset /home/kzh/dataset/N3DV --work_path /home/kzh/output/4DLangSplat/hac --scenes cut_roasted_beef --configs arguments/n3dv/default_hac.py --compre_config hac_gop60.yaml --coarse_postfix default_lang --postfix default_lang --type hac --language
echo "cut_roasted_beef Done"

python train.py --dataset /home/kzh/dataset/N3DV --work_path /home/kzh/output/4DLangSplat/hac --scenes flame_salmon --configs arguments/n3dv/default_hac.py --compre_config hac_gop60.yaml --coarse_postfix default_lang --postfix default_lang --type hac --language
echo "flame_salmon Done"

python train.py --dataset /home/kzh/dataset/N3DV --work_path /home/kzh/output/4DLangSplat/hac --scenes flame_steak --configs arguments/n3dv/default_hac.py --compre_config hac_gop60.yaml --coarse_postfix default_lang --postfix default_lang --type hac --language
echo "flame_steak Done"

python train.py --dataset /home/kzh/dataset/N3DV --work_path /home/kzh/output/4DLangSplat/hac --scenes sear_steak --configs arguments/n3dv/default_hac.py --compre_config hac_gop60.yaml --coarse_postfix default_lang --postfix default_lang --type hac --language
echo "sear_steak Done"

# gop 30   --gop 30 --gopids 0 1 2 3 4 5 6 7 8 9
# gop 60   --gopids 0 1 2 3 4
# gop 100  --gop 100 --gopids 0 1 2

#
#python train.py --dataset /home/kzh/dataset/MeetRoom --work_path /home/kzh/output/4DLangSplat/hac --scenes discussion --configs arguments/meeting/default_hac.py --compre_config hac_gop60.yaml --coarse_postfix default --postfix default  --compres_thres 32.5 --type hac --lmbda 0.001
#echo "discussion Done"
#python train.py --dataset /home/kzh/dataset/MeetRoom --work_path /home/kzh/output/4DLangSplat/hac --scenes trimming --configs arguments/meeting/default_hac.py --compre_config hac_gop60.yaml --coarse_postfix default --postfix default  --compres_thres 32.5 --type hac --lmbda 0.001
#echo "trimming Done"
#python train.py --dataset /home/kzh/dataset/MeetRoom --work_path /home/kzh/output/4DLangSplat/hac --scenes vrheadset --configs arguments/meeting/default_hac.py --compre_config hac_gop60.yaml --coarse_postfix default --postfix default  --compres_thres 32.5 --type hac --lmbda 0.001
#echo "vrheadset Done"
