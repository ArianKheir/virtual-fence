import argparse
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--opset', type=int, default=12)
    ap.add_argument('--imgsz', type=int, default=640)
    args = ap.parse_args()

    model = YOLO(args.weights)
    model.export(format='onnx', opset=args.opset, imgsz=args.imgsz)
    print('Export complete. Check the same folder for .onnx file.')

if __name__ == '__main__':
    main()
