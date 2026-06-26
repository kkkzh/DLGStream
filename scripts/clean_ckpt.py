import os
import shutil
from argparse import ArgumentParser

def delete_non_best_dirs_simple(base_dir=None):
    """简化版本：删除没有指向best软链接的目录"""
    if not os.path.exists(base_dir):
        print(f"目录 '{base_dir}' 不存在")
        return

    best_path = os.path.join(base_dir, 'best')

    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)

        if os.path.isdir(item_path):
            # 检查是否是best的实际目录
            is_best_link = (os.path.realpath(item_path) == os.path.realpath(best_path))

            # 如果是普通目录或者不是指向best的软链接，则删除
            if not is_best_link:
                try:
                    shutil.rmtree(item_path)
                    print(f"已删除: {item_path}")
                except Exception as e:
                    print(f"删除 {item_path} 失败: {e}")


from pathlib import Path

def fix_absolute_symlinks(old_prefix: str, new_root: str, dry_run: bool = False) -> int:
    """
    已知旧机器路径前缀 old_prefix（如 /home/user/proj），
    将 new_root 目录树里所有指向 old_prefix 下的绝对 symlink 改为指向 new_root 下对应位置。
    """
    new_root_p = Path(new_root).resolve()
    old_prefix_p = Path(old_prefix)

    fixed = 0
    for dirpath, dirnames, filenames in os.walk(new_root_p, followlinks=False):
        base = Path(dirpath)
        for name in list(dirnames) + list(filenames):
            link_path = base / name
            if not link_path.is_symlink():
                continue

            raw_target = os.readlink(link_path)
            t = Path(raw_target)
            if not t.is_absolute():
                continue

            # 字符串前缀判断（不依赖旧机器是否存在该路径）
            try:
                rel = t.relative_to(old_prefix_p)
            except ValueError:
                continue

            new_target = new_root_p / rel
            if dry_run:
                print(f"[DRY] {link_path} : {raw_target} -> {new_target}")
            else:
                os.unlink(link_path)
                os.symlink(str(new_target), str(link_path))
                print(f"[FIX] {link_path} : {raw_target} -> {new_target}")
            fixed += 1

    return fixed



if __name__ == '__main__':
    parser = ArgumentParser(description="Extract images from dynerf videos")
    parser.add_argument("--olddir", default='/home/patrickdd/kzh/runtime/4DLangSplat/Ours/coffee_martini/gop1_pt_lang', type=str)
    parser.add_argument("--datadir", default='/home/kzh/3DGS/Work3-results/Ours', type=str)
    parser.add_argument("--scenes", nargs="+", type=str, default=['coffee_martini', 'cook_spinach', 'cut_roasted_beef', 'flame_salmon', 'flame_steak', 'sear_steak'])
    parser.add_argument("--postfix", type=str, default=None)
    parser.add_argument("--gop", type=int, default=60)
    parser.add_argument("--gopids", nargs="+", type=int, default=[])
    args = parser.parse_args()

    # fix_absolute_symlinks(args.olddir, args.datadir, dry_run=False)

    if len(args.gopids) == 0:
        gop_nums = 300 // args.gop
        start = 0
        gop_list = range(start, gop_nums, 1)
    else:
        gop_list = args.gopids
    print(f"Processing gop", gop_list)
    for scene in args.scenes:
        for i in gop_list:
            base_dir = os.path.join(args.datadir, scene, f'gop{i}_{args.postfix}', 'compression')
            delete_non_best_dirs_simple(base_dir)
