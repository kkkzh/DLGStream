import os
import sys
import shutil
import argparse

def do_system(arg, exit=True):
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        if exit:
            sys.exit(err)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", type=str, default=['coffee_martini', 'cook_spinach', 'cut_roasted_beef', 'flame_steak', 'flame_salmon'])
    parser.add_argument("--work_path", type=str, default="/home/kzh/dataset/N3DV/")
    parser.add_argument("--lang_feat_path", type=str, default="/home/kzh/dataset2/N3DV/")
    args = parser.parse_args()

    root_output_path = args.lang_feat_path

    for idx, scene in enumerate(args.scenes):
        print(f"processing scene: {scene}")
        source_path = os.path.join(args.work_path, scene)
        output_path = os.path.join(root_output_path, scene)

        # segment images
        for camid in range(0, 21):
            camera_path = os.path.join(source_path, f'cam{str(camid).zfill(2)}')
            if os.path.exists(camera_path):
                image_path = os.path.join(camera_path, "images")
                target_path = os.path.join(root_output_path, scene, f'cam{str(camid).zfill(2)}', "sam_features")
                if not os.path.exists(target_path):
                    for level in ['default', 'small', 'middle', 'large']:
                        if not os.path.exists(os.path.join(target_path, level)):
                            os.environ['LEVEL']=level
                            print('sam level =',os.getenv("LEVEL",'default'))
                            cmd = (f"python submodules/4d-langsplat-tracking-anything-with-deva/demo/demo_automatic.py --chunk_size 4 --img_path {image_path} --amp --temporal_setting semionline --size 480 --output {os.path.join(target_path, level)}")
                            do_system(cmd)
                    do_system(f"python submodules/4d-langsplat-tracking-anything-with-deva/concat_npy.py --base_dir {target_path}")
                else:
                    print(f"sam features of camera {camid} exists, skipping!")
            else:
                print(f"camera {camid} not exists, skipping!")

        # generate clip features
        cmd = (f"python scripts/generate_clip_features.py --dataset_path {source_path} --dataset_type dynerf --precompute_seg {output_path} --output_name clip_features")
        do_system(cmd)

        clean_dir = True
        if clean_dir:
            for camid in range(0, 21):
                sam_feature_dir = os.path.join(root_output_path, scene, f"cam{str(camid).zfill(2)}", "sam_features")
                if os.path.exists(sam_feature_dir):
                    shutil.rmtree(sam_feature_dir)


        # generate video features
        # clean_dir = False
        # for camid in range(0, 21):
        #     camera_path = os.path.join(source_path, f'cam{str(camid).zfill(2)}')
        #     if os.path.exists(camera_path):
        #         image_path = os.path.join(camera_path, "images")
        #         target_path = os.path.join(root_output_path, scene, f"cam{str(camid).zfill(2)}", "sam_features", "large", "origin_mask_large")
        #         output_dir = os.path.join(root_output_path, scene, f'cam{str(camid).zfill(2)}', "mllm_features")
        #         target_dir = os.path.join(root_output_path, scene, f"cam{str(camid).zfill(2)}", "video_features")
        #         if not os.path.exists(target_dir):
        #             cmd = (f"python preprocess/generate_image_prompt.py --mask_dir {target_path} --image_dir {image_path} --output_dir {output_dir} --end_str png ")
        #             # do_system(cmd)
        #
        #             cmd = f"python preprocess/generate_video_captions.py --output_base {output_dir} --video_file {output_dir} --segmentation_dir {target_path} --mode video"
        #             # do_system(cmd)
        #             cmd = f"python preprocess/generate_video_captions.py --output_base {output_dir} --video_file {output_dir} --segmentation_dir {target_path} --mode image"
        #             do_system(cmd)
        #
        #             cmd = f"python preprocess/generate_video_features.py --caption_dir {output_dir}/output --segmentation_dir {target_path} "
        #             do_system(cmd)
        #
        #             src = os.path.join(output_dir, "output", "final_features")
        #             dst = target_dir
        #             os.rename(src, dst)
        #
        #             if clean_dir:
        #                 sam_feature_dir = os.path.join(root_output_path, scene, f"cam{str(camid).zfill(2)}", "sam_features")
        #                 shutil.rmtree(sam_feature_dir)
        #                 mllm_feature_dir = output_dir
        #                 shutil.rmtree(mllm_feature_dir)
