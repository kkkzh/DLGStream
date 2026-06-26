python test_fdsd.py -s /home/kzh/dataset/N3DV/coffee_martini --configs arguments/n3dv/default.py --compre_config fsd_gop60.yaml --checkpoint /home/kzh/output/4DLangSplat/sog/coffee_martini --postfix default_lang --qp 6 --all --gopids 0 1 2 3 4 --language
python eval_langsplat_iou_acc.py --checkpoint /home/kzh/output/4DLangSplat/sog  --postfix default_lang --qp 6 --visualize_results --mask_threshold 0.4 --timescale 0 300  --chose_mask_strategy mean --scene cook_spinach
echo "coffee_martini Done"

python test_fdsd.py -s /home/kzh/dataset/N3DV/cook_spinach --configs arguments/n3dv/default.py --compre_config fsd_gop60.yaml --checkpoint /home/kzh/output/4DLangSplat/sog/cook_spinach --postfix default_lang --qp 6 --all --gopids 0 1 2 3 4 --language
python eval_langsplat_iou_acc.py --checkpoint /home/patrickdd/kzh/runtime/4DLangSplat/sog  --postfix default_lang --qp 6 --visualize_results --mask_threshold 0.4 --timescale 0 300  --chose_mask_strategy mean --scene cook_spinach
echo "cook_spinach Done"

python test_fdsd.py -s /home/kzh/dataset/N3DV/cut_roasted_beef --configs arguments/n3dv/default.py --compre_config fsd_gop60.yaml --checkpoint /home/kzh/output/4DLangSplat/sog/cut_roasted_beef --postfix default_lang --qp 6 --all --gopids 0 1 2 3 4 --language
python eval_langsplat_iou_acc.py --checkpoint /home/patrickdd/kzh/runtime/4DLangSplat/sog  --postfix default_lang --qp 6 --visualize_results --mask_threshold 0.4 --timescale 0 300  --chose_mask_strategy mean --scene cut_roasted_beef
echo "cut_roasted_beef Done"

python test_fdsd.py -s /home/kzh/dataset/N3DV/flame_salmon --configs arguments/n3dv/default.py --compre_config fsd_gop60.yaml --checkpoint /home/kzh/output/4DLangSplat/sog/flame_salmon --postfix default_lang --qp 6 --all --gopids 0 1 2 3 4 --language
python eval_langsplat_iou_acc.py --checkpoint /home/patrickdd/kzh/runtime/4DLangSplat/sog  --postfix default_lang --qp 6 --visualize_results --mask_threshold 0.4 --timescale 0 300  --chose_mask_strategy mean --scene flame_salmon
echo "flame_salmon Done"

python test_fdsd.py -s /home/kzh/dataset/N3DV/flame_steak --configs arguments/n3dv/default.py --compre_config fsd_gop60.yaml --checkpoint /home/kzh/output/4DLangSplat/sog/flame_steak --postfix default_lang --qp 6 --all --gopids 0 1 2 3 4 --language
python eval_langsplat_iou_acc.py --checkpoint /home/patrickdd/kzh/runtime/4DLangSplat/sog  --postfix default_lang --qp 6 --visualize_results --mask_threshold 0.4 --timescale 0 300  --chose_mask_strategy mean --scene flame_steak
echo "flame_steak Done"

python test_fdsd.py -s /home/kzh/dataset/N3DV/sear_steak --configs arguments/n3dv/default.py --compre_config fsd_gop60.yaml --checkpoint /home/kzh/output/4DLangSplat/sog/sear_steak --postfix default_lang --qp 6 --all --gopids 0 1 2 3 4 --language
python eval_langsplat_iou_acc.py --checkpoint /home/patrickdd/kzh/runtime/4DLangSplat/sog  --postfix default_lang --qp 6 --visualize_results --mask_threshold 0.4 --timescale 0 300  --chose_mask_strategy mean --scene sear_steak
echo "sear_steak Done"


python test_hac.py -s /home/kzh/dataset/N3DV/coffee_martini --configs arguments/n3dv/default_hac.py --compre_config hac_gop60.yaml --checkpoint /home/kzh/output/4DLangSplat/hac/coffee_martini --postfix default_lang --qp 6 --all --gopids 0 1 2 3 4 --language
python eval_langsplat_iou_acc.py --checkpoint /home/kzh/output/4DLangSplat/hac  --postfix default_lang --qp 6 --visualize_results --mask_threshold 0.4 --timescale 0 300  --chose_mask_strategy mean --scene cook_spinach
echo "coffee_martini Done"

python test_hac.py -s /home/kzh/dataset/N3DV/cook_spinach --configs arguments/n3dv/default_hac.py --compre_config hac_gop60.yaml --checkpoint /home/kzh/output/4DLangSplat/hac/cook_spinach --postfix default_lang --qp 6 --all --gopids 0 1 2 3 4 --language
python eval_langsplat_iou_acc.py --checkpoint /home/patrickdd/kzh/runtime/4DLangSplat/hac  --postfix default_lang --qp 6 --visualize_results --mask_threshold 0.4 --timescale 0 300  --chose_mask_strategy mean --scene cook_spinach
echo "cook_spinach Done"

python test_hac.py -s /home/kzh/dataset/N3DV/cut_roasted_beef --configs arguments/n3dv/default_hac.py --compre_config hac_gop60.yaml --checkpoint /home/kzh/output/4DLangSplat/hac/cut_roasted_beef --postfix default_lang --qp 6 --all --gopids 0 1 2 3 4 --language
python eval_langsplat_iou_acc.py --checkpoint /home/patrickdd/kzh/runtime/4DLangSplat/hac  --postfix default_lang --qp 6 --visualize_results --mask_threshold 0.4 --timescale 0 300  --chose_mask_strategy mean --scene cut_roasted_beef
echo "cut_roasted_beef Done"

python test_hac.py -s /home/kzh/dataset/N3DV/flame_salmon --configs arguments/n3dv/default_hac.py --compre_config hac_gop60.yaml --checkpoint /home/kzh/output/4DLangSplat/hac/flame_salmon --postfix default_lang --qp 6 --all --gopids 0 1 2 3 4 --language
python eval_langsplat_iou_acc.py --checkpoint /home/patrickdd/kzh/runtime/4DLangSplat/hac  --postfix default_lang --qp 6 --visualize_results --mask_threshold 0.4 --timescale 0 300  --chose_mask_strategy mean --scene flame_salmon
echo "flame_salmon Done"

python test_hac.py -s /home/kzh/dataset/N3DV/flame_steak --configs arguments/n3dv/default_hac.py --compre_config hac_gop60.yaml --checkpoint /home/kzh/output/4DLangSplat/hac/flame_steak --postfix default_lang --qp 6 --all --gopids 0 1 2 3 4 --language
python eval_langsplat_iou_acc.py --checkpoint /home/patrickdd/kzh/runtime/4DLangSplat/hac  --postfix default_lang --qp 6 --visualize_results --mask_threshold 0.4 --timescale 0 300  --chose_mask_strategy mean --scene flame_steak
echo "flame_steak Done"

python test_hac.py -s /home/kzh/dataset/N3DV/sear_steak --configs arguments/n3dv/default_hac.py --compre_config hac_gop60.yaml --checkpoint /home/kzh/output/4DLangSplat/hac/sear_steak --postfix default_lang --qp 6 --all --gopids 0 1 2 3 4 --language
python eval_langsplat_iou_acc.py --checkpoint /home/patrickdd/kzh/runtime/4DLangSplat/hac  --postfix default_lang --qp 6 --visualize_results --mask_threshold 0.4 --timescale 0 300  --chose_mask_strategy mean --scene sear_steak
echo "sear_steak Done"
