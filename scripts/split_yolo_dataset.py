import argparse, os, random, shutil
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='dataset root with images/ and labels/')
    ap.add_argument('--val', type=float, default=0.2)
    args = ap.parse_args()

    root = Path(args.root)
    imgs = list((root/'images').glob('*.*'))
    random.shuffle(imgs)
    n_val = int(len(imgs)*args.val)
    val_imgs = set(imgs[:n_val])

    (root/'images_val').mkdir(exist_ok=True)
    (root/'labels_val').mkdir(exist_ok=True)

    for img in imgs:
        txt = (root/'labels'/ (img.stem + '.txt'))
        if img in val_imgs:
            shutil.copy2(img, root/'images_val'/img.name)
            if txt.exists():
                shutil.copy2(txt, root/'labels_val'/txt.name)
        else:
            # keep train in place
            pass
    print(f'Validation split created: {len(val_imgs)} images.')

if __name__ == '__main__':
    main()
