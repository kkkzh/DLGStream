import argparse
import csv
import glob
import os
import json
from collections import defaultdict
from typing import Dict, Union

import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from autoencoder.model import Autoencoder
from utils.colormaps import apply_colormap, ColormapOptions
from utils.eval_utils import smooth, smooth_cuda, colormap_saving, vis_mask_save, polygon_to_mask, stack_mask
from utils.openclip_encoder import OpenCLIPNetwork


def eval_gt_lerfdata(json_folder: Union[str, Path] = None, output_path: Path = None, prompts=None, replace_prompts=None, dataset_type=None):
    """
    prompts: if pompts is None, check all words else check only words in prompts
    replace_prompts: dict() replace the prompt for query
    Organize lerf's gt annotations
    gt format:
        file name: frame_xxxxx.json
        file content: labelme format
    return:
        gt_ann: dict()
            keys: str(int(idx))
            values: dict()
                keys: str(label)
                values: dict() which contain 'bboxes' and 'mask'
    """
    # Load the COCO format json file
    with open(os.path.join(json_folder, '_annotations.coco.json'), 'r') as f:
        data = json.load(f)

    gt_ann = {}
    img_paths = []
    id2name = {}
    name2id = {}
    im_id2imidx = {}

    for item in data['categories']:
        idx = item['id']
        id2name[int(idx)] = item['name']
        name2id[item['name']] = int(idx)
    for img_data in data['images']:
        img_ann = defaultdict(dict)
        idx = img_data['id']
        img_name = img_data['file_name']
        img_paths.append(os.path.join(json_folder, img_name))
        h, w = img_data['height'], img_data['width']

        for annotation in data['annotations']:
            if annotation['image_id'] == idx:
                label = id2name[annotation['category_id']]
                if prompts is not None and label not in prompts:
                    continue

                box = np.asarray(annotation['bbox']).reshape(-1)  # x1, y1, width, height
                box[2] += box[0]
                box[3] += box[1]
                segmentation = annotation['segmentation'][0]
                assert len(segmentation) % 2 == 0
                point_segmentation = []
                for i in range(0, len(segmentation), 2):
                    point_segmentation.append([segmentation[i], segmentation[i + 1]])

                mask = polygon_to_mask((h, w), point_segmentation)
                if replace_prompts is not None and label in replace_prompts.keys():
                    label_list = replace_prompts[label]
                    label_list.append(label)
                else:
                    label_list = [label]
                for label in label_list:
                    if img_ann[label].get('mask', None) is not None:
                        mask = stack_mask(img_ann[label]['mask'], mask)
                        img_ann[label]['bboxes'] = np.concatenate(
                            [img_ann[label]['bboxes'].reshape(-1, 4), box.reshape(-1, 4)], axis=0)
                    else:
                        img_ann[label]['bboxes'] = box
                    img_ann[label]['mask'] = mask

                    # Save for visualization
                    save_path = output_path / 'gt' / img_name.split('.')[0] / f'{label}.jpg'
                    if args.visualize_results:
                        save_path.parent.mkdir(exist_ok=True, parents=True)
                        vis_mask_save(mask, save_path)

        gt_ann[f'{idx}'] = img_ann

    for item in data['images']:
        idx = item['id']
        filename = item['file_name']

        im_id2imidx[idx] = int(filename.split('_')[0])  # 减一是为了对齐npy文件序号和image idx

    return gt_ann, (h, w), img_paths, id2name, name2id, im_id2imidx


def activate_stream(sem_map,
                    image,
                    clip_model,
                    image_name: Path = None,
                    # img_ann: Dict = None,
                    thresh: float = 0.5,
                    colormap_options=None,
                    name2id=None,
                    scale=30,
                    chose_mask_strategy='point',
                    imageid=None,
                    visualize_results=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_map = clip_model.get_max_across(sem_map)  # 3xkx832x1264
    n_head, n_prompt, h, w = valid_map.shape

    # positive prompts
    chosen_iou_list, chosen_lvl_list = [], []
    prompt_iou_lvl_dict = {}
    mask_dict = {}
    mask_for_video_dict = {}
    for k in range(n_prompt):
        iou_lvl = torch.zeros(n_head).to(device)
        mask_lvl = torch.zeros((n_head, h, w)).to(device)
        mask_for_video = torch.zeros((n_head, h, w)).to(device)
        output_list = []
        thresh_list = []
        for i in range(n_head):
            avg_pool = torch.nn.AvgPool2d(kernel_size=scale, stride=1, padding=14, count_include_pad=False).to(device)
            avg_filtered = avg_pool(valid_map[i][k].unsqueeze(0).unsqueeze(0))
            valid_map[i][k] = 0.5 * (avg_filtered.squeeze(0).squeeze(0) + valid_map[i][k])

            if visualize_results:
                output_path_relev = image_name / 'heatmap' / f'{clip_model.positives[k]}_{i}'
                output_path_relev.parent.mkdir(exist_ok=True, parents=True)
                colormap_saving(valid_map[i][k].unsqueeze(-1), colormap_options, output_path_relev)

            # truncate the heatmap into mask
            output = valid_map[i][k]
            output = output - torch.min(output)
            output = output / (torch.max(output) + 1e-9)
            output = output * (1.0 - (-1.0)) + (-1.0)
            output = torch.clip(output, 0, 1)
            output_list.append(output)

            thresh_list.append(thresh)

            if visualize_results:
                p_i = torch.clip(valid_map[i][k] - 0.5, 0, 1).unsqueeze(-1)
                valid_composited = apply_colormap(p_i / (p_i.max() + 1e-6), ColormapOptions("turbo"))
                mask = (valid_map[i][k] < 0.5).squeeze()
                valid_composited[mask, :] = image[mask, :] * 0.6
                output_path_compo = image_name / 'composited' / f'{clip_model.positives[k]}_{i}'
                output_path_compo.parent.mkdir(exist_ok=True, parents=True)
                colormap_saving(valid_composited, colormap_options, output_path_compo)

            if i == 0 and visualize_results:
                # background_only = image.clone()
                # background_only = background_only * 0.6
                # output_path_background = image_name / 'background' / f'{clip_model.positives[k]}_{i}'
                # output_path_background.parent.mkdir(exist_ok=True, parents=True)
                # colormap_saving(background_only, colormap_options, output_path_background)

                overlay_color = torch.tensor([128 / 255, 0.0, 128 / 255]).cuda()
                promtp_name = clip_model.positives[k]
                mask_gt = img_ann[promtp_name]['mask'].astype(np.uint8)
                overlay_layer = overlay_color * 0.5
                annotated_image = image.clone()
                annotated_image[mask_gt.squeeze() > 0] = annotated_image[mask_gt.squeeze() > 0] * 0.5 + overlay_layer * 255

                output_path_annotation = image_name / 'annotation' / f'{clip_model.positives[k]}_{i}'
                output_path_annotation.parent.mkdir(exist_ok=True, parents=True)
                colormap_saving(annotated_image, colormap_options, output_path_annotation)

            mask_pred = (output > thresh).type(torch.uint8)
            mask_for_video[i] = mask_pred
            mask_pred = smooth_cuda(mask_pred)
            mask_lvl[i] = mask_pred

            promtp_name = clip_model.positives[k]
            mask_gt = torch.from_numpy(img_ann[promtp_name]['mask'].astype(np.uint8)).to(device)

            intersection = torch.sum(torch.logical_and(mask_gt, mask_pred))
            union = torch.sum(torch.logical_or(mask_gt, mask_pred))
            iou = torch.sum(intersection) / torch.sum(union)
            iou_lvl[i] = iou

        score_lvl = torch.zeros((n_head,), device=valid_map.device)

        for i in range(n_head):
            if chose_mask_strategy == "point":
                score = valid_map[i, k].max()
                score_lvl[i] = score
            elif chose_mask_strategy == "mean":
                # Calculate the average score within the thresholded mask
                thresh = thresh_list[i]
                chose_mask_area = (output_list[i].cpu().numpy() > thresh).astype(np.uint8)

                chose_mask_area_after_smooth = chose_mask_area

                if np.sum(chose_mask_area_after_smooth) > 0:
                    score = valid_map[i, k][chose_mask_area_after_smooth].mean().item()
                    # print("score:", score)
                else:
                    score = 0

                score_lvl[i] = score
            else:
                raise NotImplementedError

        chosen_lvl = torch.argmax(score_lvl)
        chosen_iou_list.append(iou_lvl[chosen_lvl])
        chosen_lvl_list.append(chosen_lvl.cpu().numpy())

        if visualize_results:
            save_path = image_name / f'chosen_{clip_model.positives[k]}.png'
            vis_mask = mask_lvl[chosen_lvl].cpu().numpy()
            vis_mask_save(vis_mask, save_path)
            save_path = image_name / f'chosen_for_video_{clip_model.positives[k]}.png'
            vis_mask = mask_for_video[chosen_lvl].cpu().numpy()
            vis_mask_save(vis_mask, save_path)

        prompt_iou_lvl_dict[clip_model.positives[k]] = (iou_lvl[chosen_lvl], chosen_lvl.cpu().numpy(), score_lvl.cpu().numpy(), thresh_list)
        mask_dict[clip_model.positives[k]] = mask_lvl[chosen_lvl]
        mask_for_video_dict[clip_model.positives[k]] = [mask_for_video[chosen_lvl]]

    return chosen_iou_list, chosen_lvl_list, prompt_iou_lvl_dict, mask_dict, mask_for_video_dict


def activate_stream_2(sem_map,
                    image,
                    clip_model,
                    image_name: Path = None,
                    # img_ann: Dict = None,
                    thresh: float = 0.5,
                    colormap_options=None,
                    name2id=None,
                    scale=30,
                    chose_mask_strategy='point',
                    imageid=None,
                    visualize_results=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_map_o = clip_model.get_max_across(sem_map)  # 3xkx832x1264
    valid_map = valid_map_o.clone()
    n_head, n_prompt, h, w = valid_map.shape

    # positive prompts
    chosen_iou_list, chosen_lvl_list = [], []
    prompt_iou_lvl_dict = {}
    mask_dict = {}
    mask_for_video_dict = {}
    output_list_dict = {}
    for k in range(n_prompt):
        iou_lvl = torch.zeros(n_head).to(device)
        mask_lvl = torch.zeros((n_head, h, w)).to(device)
        mask_for_video = torch.zeros((n_head, h, w)).to(device)
        output_list = []
        thresh_list = []
        for i in range(n_head):
            avg_pool = torch.nn.AvgPool2d(kernel_size=scale, stride=1, padding=14, count_include_pad=False).to(device)
            avg_filtered = avg_pool(valid_map[i][k].unsqueeze(0).unsqueeze(0))
            valid_map[i][k] = 0.5 * (avg_filtered.squeeze(0).squeeze(0) + valid_map[i][k])

            if visualize_results:
                output_path_relev = image_name / 'heatmap' / f'{clip_model.positives[k]}_{i}'
                output_path_relev.parent.mkdir(exist_ok=True, parents=True)
                colormap_saving(valid_map[i][k].unsqueeze(-1), colormap_options, output_path_relev)

            # truncate the heatmap into mask
            output = valid_map[i][k]
            output = output - torch.min(output)
            output = output / (torch.max(output) + 1e-9)
            output = output * (1.0 - (-1.0)) + (-1.0)
            output = torch.clip(output, 0, 1)
            output_list.append(output)

            thresh_list.append(thresh)

            if visualize_results:
                p_i = torch.clip(valid_map[i][k] - 0.5, 0, 1).unsqueeze(-1)
                valid_composited = apply_colormap(p_i / (p_i.max() + 1e-6), ColormapOptions("turbo"))
                mask = (valid_map[i][k] < 0.5).squeeze()
                valid_composited[mask, :] = image[mask, :] * 0.6
                output_path_compo = image_name / 'composited' / f'{clip_model.positives[k]}_{i}'
                output_path_compo.parent.mkdir(exist_ok=True, parents=True)
                colormap_saving(valid_composited, colormap_options, output_path_compo)

            if i == 0 and visualize_results:
                # background_only = image.clone()
                # background_only = background_only * 0.6
                # output_path_background = image_name / 'background' / f'{clip_model.positives[k]}_{i}'
                # output_path_background.parent.mkdir(exist_ok=True, parents=True)
                # colormap_saving(background_only, colormap_options, output_path_background)

                overlay_color = torch.tensor([128 / 255, 0.0, 128 / 255]).cuda()
                promtp_name = clip_model.positives[k]
                mask_gt = img_ann[promtp_name]['mask'].astype(np.uint8)
                overlay_layer = overlay_color * 0.5
                annotated_image = image.clone()
                annotated_image[mask_gt.squeeze() > 0] = annotated_image[mask_gt.squeeze() > 0] * 0.5 + overlay_layer * 255

                output_path_annotation = image_name / 'annotation' / f'{clip_model.positives[k]}_{i}'
                output_path_annotation.parent.mkdir(exist_ok=True, parents=True)
                colormap_saving(annotated_image, colormap_options, output_path_annotation)

            mask_pred = (output > thresh).type(torch.uint8)
            mask_for_video[i] = mask_pred
            mask_pred = smooth_cuda(mask_pred)
            mask_lvl[i] = mask_pred

            promtp_name = clip_model.positives[k]
            mask_gt = torch.from_numpy(img_ann[promtp_name]['mask'].astype(np.uint8)).to(device)

            intersection = torch.sum(torch.logical_and(mask_gt, mask_pred))
            union = torch.sum(torch.logical_or(mask_gt, mask_pred))
            iou = torch.sum(intersection) / torch.sum(union)
            iou_lvl[i] = iou

        output_list_dict[k] = output_list
        score_lvl = torch.zeros((n_head,), device=valid_map.device)

        for i in range(n_head):
            if chose_mask_strategy == "point":
                score = valid_map[i, k].max()
                score_lvl[i] = score
            elif chose_mask_strategy == "mean":
                # Calculate the average score within the thresholded mask
                thresh = thresh_list[i]
                chose_mask_area = (output_list[i].cpu().numpy() > thresh).astype(np.uint8)

                chose_mask_area_after_smooth = chose_mask_area

                if np.sum(chose_mask_area_after_smooth) > 0:
                    score = valid_map[i, k][chose_mask_area_after_smooth].mean().item()
                    # print("score:", score)
                else:
                    score = 0

                score_lvl[i] = score
            else:
                raise NotImplementedError

        chosen_lvl = torch.argmax(score_lvl)
        chosen_iou_list.append(iou_lvl[chosen_lvl])
        chosen_lvl_list.append(chosen_lvl.cpu().numpy())

        if visualize_results:
            save_path = image_name / f'chosen_{clip_model.positives[k]}.png'
            vis_mask = mask_lvl[chosen_lvl].cpu().numpy()
            vis_mask_save(vis_mask, save_path)
            save_path = image_name / f'chosen_for_video_{clip_model.positives[k]}.png'
            vis_mask = mask_for_video[chosen_lvl].cpu().numpy()
            vis_mask_save(vis_mask, save_path)

        prompt_iou_lvl_dict[clip_model.positives[k]] = (iou_lvl[chosen_lvl], chosen_lvl.cpu().numpy(), score_lvl.cpu().numpy(), thresh_list)
        mask_dict[clip_model.positives[k]] = mask_lvl[chosen_lvl]
        mask_for_video_dict[clip_model.positives[k]] = [mask_for_video[chosen_lvl]]

    # positive prompts
    valid_map = valid_map_o.clone()
    acc_num = 0
    positives = list(img_ann.keys())
    for k in range(len(positives)):
        select_output = valid_map[:, k]

        # NOTE 平滑后的激活值图中找最大值点
        scale = 30
        kernel = np.ones((scale, scale)) / (scale ** 2)
        np_relev = select_output.cpu().numpy()
        avg_filtered = cv2.filter2D(np_relev.transpose(1, 2, 0), -1, kernel)
        # print(f"o_avg_filtered shape: {avg_filtered.shape}")

        # avg_pool = torch.nn.AvgPool2d(kernel_size=scale, stride=1, padding=14, count_include_pad=False).to(device)
        # avg_filtered = avg_pool(select_output.unsqueeze(0))
        # avg_filtered = avg_filtered.squeeze().permute(1,2,0)
        # avg_filtered = avg_filtered.cpu().numpy()

        score_lvl = np.zeros((n_head,))
        coord_lvl = []
        for i in range(n_head):
            # if chose_mask_strategy == "point":
            #     score = avg_filtered[..., i].max()
            # elif chose_mask_strategy == "mean":
            #     # Calculate the average score within the thresholded mask
            #     chose_mask_area = (avg_filtered[..., i].cpu().numpy() > thresh).astype(np.uint8)
            #     chose_mask_area_after_smooth = chose_mask_area
            #
            #     if np.sum(chose_mask_area_after_smooth) > 0:
            #         score = avg_filtered[..., i][chose_mask_area_after_smooth].mean().item()
            #     else:
            #         score = 0
            # else:
            #     raise NotImplementedError
            # print(f"score: {score}")

            score = avg_filtered[..., i].max()
            coord = np.nonzero(avg_filtered[..., i] == score)
            # print(f"o_coord: {coord}")
            # avg_filtered = avg_filtered.cpu().numpy()
            # d = np.abs(avg_filtered[..., i] - score)
            # min_d = d.min()
            # coord = np.nonzero(d == min_d)
            # print(f"m_coord: {coord}")

            score_lvl[i] = score
            coord_lvl.append(np.asarray(coord).transpose(1, 0)[..., ::-1])

        selec_head = np.argmax(score_lvl)
        coord_final = coord_lvl[selec_head]

        for box in img_ann[positives[k]]['bboxes'].reshape(-1, 4):
            flag = 0
            x1, y1, x2, y2 = box
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            for cord_list in coord_final:
                if (x_min <= cord_list[0] <= x_max and
                        y_min <= cord_list[1] <= y_max):
                    acc_num += 1
                    flag = 1
                    break
            if flag != 0:
                break

    return chosen_iou_list, chosen_lvl_list, prompt_iou_lvl_dict, mask_dict, mask_for_video_dict, acc_num

def lerf_localization(sem_map,
                      image,
                      clip_model,
                      image_name,
                      img_ann,
                      thresh: float = 0.5,
                      chose_mask_strategy='point'):
    output_path_loca = image_name / 'localization'
    output_path_loca.mkdir(exist_ok=True, parents=True)

    valid_map = clip_model.get_max_across(sem_map)  # 3xkx832x1264
    n_head, n_prompt, h, w = valid_map.shape

    # positive prompts
    acc_num = 0
    positives = list(img_ann.keys())
    for k in range(len(positives)):
        select_output = valid_map[:, k]

        # NOTE 平滑后的激活值图中找最大值点
        scale = 30
        kernel = np.ones((scale, scale)) / (scale ** 2)
        np_relev = select_output.cpu().numpy()
        avg_filtered = cv2.filter2D(np_relev.transpose(1, 2, 0), -1, kernel)

        score_lvl = np.zeros((n_head,))
        coord_lvl = []
        for i in range(n_head):
            score = avg_filtered[..., i].max()
            coord = np.nonzero(avg_filtered[..., i] == score)
            score_lvl[i] = score
            coord_lvl.append(np.asarray(coord).transpose(1, 0)[..., ::-1])

        selec_head = np.argmax(score_lvl)
        coord_final = coord_lvl[selec_head]

        for box in img_ann[positives[k]]['bboxes'].reshape(-1, 4):
            flag = 0
            x1, y1, x2, y2 = box
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            for cord_list in coord_final:
                if (x_min <= cord_list[0] <= x_max and
                        y_min <= cord_list[1] <= y_max):
                    acc_num += 1
                    flag = 1
                    break
            if flag != 0:
                break

        # NOTE 将平均后的结果与原结果相加，抑制噪声并保持激活边界清晰
        # avg_filtered = torch.from_numpy(avg_filtered[..., selec_head]).unsqueeze(-1).to(select_output.device)
        # torch_relev = 0.5 * (avg_filtered + select_output[selec_head].unsqueeze(-1))
        # p_i = torch.clip(torch_relev - 0.5, 0, 1)
        # valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
        # mask = (torch_relev < 0.5).squeeze()
        # valid_composited[mask, :] = image[mask, :] * 0.3
        #
        # save_path = output_path_loca / f"{positives[k]}.png"
        # show_result(valid_composited.cpu().numpy(), coord_final,
        #             img_ann[positives[k]]['bboxes'], save_path)
    return acc_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation script parameters")
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--postfix", type=str, default=None)
    parser.add_argument("--qp", type=int, default=6)
    parser.add_argument("--output",type=str, default="language_query")
    parser.add_argument("--annotation_folder",type=str, default='/home/patrickdd/kzh/dataset/N3DV/4DLangSplat')
    parser.add_argument("--timescale", nargs="+", type=int, default=[0, 300])
    parser.add_argument('--feat_dim', type=int, default=3)
    parser.add_argument('--encoder_hidden_dims', nargs="+", type=int, default=[256, 128, 64, 32, 3])
    parser.add_argument('--decoder_hidden_dims', nargs="+", type=int, default=[16, 32, 64, 128, 256, 512])
    parser.add_argument("--ae_ckpt_path", type=str, default='/home/patrickdd/kzh/projects/Work4/autoencoder/ckpt')
    parser.add_argument('--mask_threshold', type=float, default=0.4)
    parser.add_argument('--scale', type=int, default=29)
    parser.add_argument('--chose_mask_strategy', choices=['point', 'mean'], default="mean")
    parser.add_argument('--visualize_results', action='store_true', help='Whether to save visualization results')
    args = parser.parse_args()

    mask_thresh = args.mask_threshold
    args.annotation_folder = os.path.join(args.annotation_folder, args.scene)
    args.ae_ckpt_path = os.path.join(args.ae_ckpt_path, f'{args.scene}_clip', 'best_ckpt.pth')

    output_path = os.path.join(args.checkpoint, args.output)
    os.makedirs(output_path, exist_ok=True)

    colormap_options = ColormapOptions(
        colormap="turbo",
        normalize=True,
        colormap_min=-1.0,
        colormap_max=1.0,
    )

    # load label
    json_folder = os.path.join(args.annotation_folder, 'train')
    gt_ann, image_shape, image_paths, id2name, name2id, im_id2imidx = eval_gt_lerfdata(Path(json_folder), Path(output_path), prompts=None, replace_prompts=None)

    # eval_index_list = [int(idx) for idx in list(gt_ann.keys())]  # range(1, frame_num+1)
    eval_index_list = []
    for idx in list(gt_ann.keys()):
        frame_id = im_id2imidx[int(idx)]
        if args.timescale[0] <= frame_id < args.timescale[1]:
            eval_index_list.append(int(idx))

    # load rendered lang features
    levels = 3
    compressed_sem_feats = np.zeros((levels, len(eval_index_list), *image_shape, args.feat_dim), dtype=np.float32)

    rendered_features = True
    separated = False
    if rendered_features:
        feat_path = os.path.join(args.checkpoint, f'{args.scene}', 'experiments', f'results_{args.postfix}' if args.postfix is not None else 'results',
                                       f'metrics_qp{args.qp}')
        feat_dir = os.path.join(feat_path, 'images')
        for j, idx in enumerate(eval_index_list):
            frame_id = im_id2imidx[idx]
            # lang_splat = np.load(feat_paths_lvl[frame_id])
            lang_splat = np.load(os.path.join(feat_dir, str(frame_id).zfill(5) + '.npy'))
            assert levels * 3 == lang_splat.shape[2]
            for level in range(0, lang_splat.shape[2], 3):
                i = level // 3
                compressed_sem_feats[i][j] = lang_splat[:, :, level:level+3]
    elif separated:
        for i in range(1, 4):
            scene = args.checkpoint.split('/')[-1]
            # feat_paths_lvl = sorted(glob.glob(os.path.join(args.checkpoint, f'{scene}_{i}', 'test_lang/ours_10000/renders_npy', '*.npy')), key=lambda file_name: int(os.path.basename(file_name).split(".npy")[0]))
            feat_path = os.path.join(args.checkpoint, f'{scene}_{i}', 'test_lang/ours_10000/renders_npy')
            for j, idx in enumerate(eval_index_list):
                frame_id = im_id2imidx[idx]
                compressed_sem_feats[i-1][j] = np.load(os.path.join(feat_path, str(frame_id).zfill(5) + '.npy'))
    else:
        for j, idx in enumerate(eval_index_list):
            frame_id = im_id2imidx[idx]
            feat_dir = os.path.join(args.checkpoint, 'images')
            seg_map = torch.from_numpy(np.load(os.path.join(feat_dir, str(frame_id).zfill(4) + '_s.npy')))
            lf_map = torch.from_numpy(np.load(os.path.join(feat_dir, str(frame_id).zfill(4) + '_f.npy')))

            y, x = torch.meshgrid(torch.arange(0, 1014), torch.arange(0, 1352))
            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)
            seg = seg_map[:, y, x].squeeze(-1).long()
            mask = seg != -1

            point_features, masks = [], []
            for level in range(1, 4):
                point_feature = lf_map[seg[level:level + 1]].squeeze(0)
                point_feature = point_feature.reshape(1014, 1352, -1).permute(0, 1, 2)
                # _mask = mask[level:level + 1].reshape(1, 1352, 1014)
                # point_features.append(point_feature)
                # masks.append(_mask)
                i = level - 1
                compressed_sem_feats[i][j] = point_feature

    # load openclip model
    clip_model = OpenCLIPNetwork('cuda')

    # load language feature autoencoder model
    checkpoint = torch.load(args.ae_ckpt_path, map_location='cuda:0')  # 1
    # print(checkpoint.keys())
    model = Autoencoder(args.encoder_hidden_dims, args.decoder_hidden_dims).to('cuda:0')  # 1
    model.load_state_dict(checkpoint)
    model.eval()

    # eval
    chosen_iou_all, chosen_lvl_list = [], []
    prompt_iou_all_dict = {}
    acc_num = 0
    for j, idx in enumerate(tqdm(eval_index_list)):
        image_name = Path(output_path) / f'{idx:0>5}'
        if args.visualize_results:
            image_name.mkdir(exist_ok=True, parents=True)

        sem_feat = compressed_sem_feats[:, j, ...]
        sem_feat = torch.from_numpy(sem_feat).float().to('cuda:0')  # 1

        rgb_img = cv2.imread(image_paths[idx])[..., ::-1]
        rgb_img = (rgb_img / 255.0).astype(np.float32)
        rgb_img = torch.from_numpy(rgb_img).to('cuda')

        with torch.no_grad():
            lvl, h, w, f = sem_feat.shape
            if f != 512:
                restored_feat = model.decode(sem_feat.flatten(0, 2))
            else:
                restored_feat = sem_feat
            restored_feat = restored_feat.view(lvl, h, w, -1)
        restored_feat = restored_feat.to('cuda:0')

        img_ann = gt_ann[f'{idx}']

        clip_model.set_positives(list(img_ann.keys()))
        # c_iou_list, c_lvl, prompt_iou_lvl_dict, chosen_mask_dict, chosen_mask_for_video_dict = activate_stream(restored_feat, rgb_img, clip_model, image_name,
        #             thresh=mask_thresh, colormap_options=colormap_options, name2id=name2id, scale=args.scale, chose_mask_strategy=args.chose_mask_strategy, imageid=j, visualize_results=args.visualize_results)
        c_iou_list, c_lvl, prompt_iou_lvl_dict, chosen_mask_dict, chosen_mask_for_video_dict, acc_num_img = activate_stream_2(restored_feat, rgb_img, clip_model, image_name,
                    thresh=mask_thresh, colormap_options=colormap_options, name2id=name2id, scale=args.scale, chose_mask_strategy=args.chose_mask_strategy, imageid=j, visualize_results=args.visualize_results)
        acc_num += acc_num_img

        for key, (iou, lvl, lvl_all, tresh_all) in prompt_iou_lvl_dict.items():
            if key not in prompt_iou_all_dict:
                prompt_iou_all_dict[key] = []

            smoothed_video_features_sim = 0
            prompt_iou_all_dict[key].append((idx, iou, lvl, lvl_all, tresh_all, smoothed_video_features_sim))

        chosen_iou_all.extend(c_iou_list)
        chosen_lvl_list.extend(c_lvl)

    #
    result_data = []
    for key in prompt_iou_all_dict.keys():
        format_data =[key]
        mean_iou_key = sum([fm[1].item() for fm in prompt_iou_all_dict[key]])/ len(prompt_iou_all_dict[key])
        format_data.append(mean_iou_key)
        format_data.append([fm[2] for fm in prompt_iou_all_dict[key]])  # Lvls
        format_data.append([fm[3] for fm in prompt_iou_all_dict[key]])  # Similarity
        format_data.append([fm[4] for fm in prompt_iou_all_dict[key]])  # Thresh
        format_data.append([fm[5] for fm in prompt_iou_all_dict[key]])  # Video features Similarity

        for idx in eval_index_list:
            exist_prompt_this_frame = False
            for fm in  prompt_iou_all_dict[key]:
                if fm[0] == idx:
                    format_data.append(fm[1])
                    exist_prompt_this_frame = True
            if not exist_prompt_this_frame:
                format_data.append("NA")

        result_data.append(format_data)
        print(f"key:{key}, mean_iou:{mean_iou_key}")
    print(f"Mean IoU: {sum([fm[1] for fm in result_data]) / len(result_data)}")

    # localization acc
    total_bboxes = 0
    for img_ann in gt_ann.values():
        total_bboxes += len(list(img_ann.keys()))
    acc = acc_num / total_bboxes
    print("Localization accuracy: " + f'{acc:.4f}', f"total_bboxes: {total_bboxes}, acc_num: {acc_num}")

    with open(os.path.join(output_path, 'time-agnostic_results.csv'), mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        header_list = ['Prompt', 'Mean IoU', 'Lvls', 'Similarity', 'Tresh', "Video feature Similarity"]
        for i in eval_index_list:
            header_list.append(f'frame_{i + 1}_iou')
        writer.writerow(header_list)
        for data in result_data:
            writer.writerow(data)