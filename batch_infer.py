import os
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from utils.common import merge_config, get_model
from PIL import Image


def pred2coords(pred, row_anchor, col_anchor, local_width=1, original_image_width=1640, original_image_height=590):
    batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
    batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape

    max_indices_row = pred['loc_row'].argmax(1).cpu()
    valid_row = pred['exist_row'].argmax(1).cpu()
    max_indices_col = pred['loc_col'].argmax(1).cpu()
    valid_col = pred['exist_col'].argmax(1).cpu()

    pred['loc_row'] = pred['loc_row'].cpu()
    pred['loc_col'] = pred['loc_col'].cpu()

    coords = []
    row_lane_idx = [1, 2]
    col_lane_idx = [0, 3]

    for i in row_lane_idx:
        tmp = []
        if valid_row[0, :, i].sum() > num_cls_row / 2:
            for k in range(valid_row.shape[1]):
                if valid_row[0, k, i]:
                    all_ind = torch.tensor(list(range(max(0, max_indices_row[0, k, i] - local_width),
                                                      min(num_grid_row - 1,
                                                          max_indices_row[0, k, i] + local_width) + 1)))
                    out_tmp = (pred['loc_row'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_row - 1) * original_image_width
                    tmp.append((int(out_tmp), int(row_anchor[k] * original_image_height)))
            coords.append(tmp)

    for i in col_lane_idx:
        tmp = []
        if valid_col[0, :, i].sum() > num_cls_col / 4:
            for k in range(valid_col.shape[1]):
                if valid_col[0, k, i]:
                    all_ind = torch.tensor(list(range(max(0, max_indices_col[0, k, i] - local_width),
                                                      min(num_grid_col - 1,
                                                          max_indices_col[0, k, i] + local_width) + 1)))
                    out_tmp = (pred['loc_col'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_col - 1) * original_image_height
                    tmp.append((int(col_anchor[k] * original_image_width), int(out_tmp)))
            coords.append(tmp)
    return coords


if __name__ == "__main__":
    args, cfg = merge_config()
    torch.backends.cudnn.benchmark = True

    net = get_model(cfg)
    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v
    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transforms = transforms.Compose([
        transforms.Resize((int(cfg.train_height / cfg.crop_ratio), cfg.train_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # 输入输出路径
    input_dir = args.input_path
    output_dir = cfg.test_work_dir
    os.makedirs(output_dir, exist_ok=True)

    img_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    print(f"Found {len(img_files)} images to process")

    for idx, img_name in enumerate(img_files):
        img_path = os.path.join(input_dir, img_name)
        print(f"[{idx + 1}/{len(img_files)}] Processing {img_name}...")

        img = cv2.imread(img_path)
        if img is None:
            continue

        img_h, img_w = img.shape[:2]
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_tensor = img_transforms(img_pil).unsqueeze(0).cuda()
        img_tensor = img_tensor[:, :, -cfg.train_height:, :]

        with torch.no_grad():
            pred = net(img_tensor)

        coords = pred2coords(pred, cfg.row_anchor, cfg.col_anchor, original_image_width=img_w,
                             original_image_height=img_h)
        for lane in coords:
            if len(lane) > 1:
                x_coords = [p[0] for p in lane]
                y_coords = [p[1] for p in lane]
                min_y, max_y = min(y_coords), max(y_coords)
                coeffs = np.polyfit(y_coords, x_coords, 1)
                pt_start = (int(np.polyval(coeffs, max_y)), max_y)
                pt_end = (int(np.polyval(coeffs, min_y)), min_y)
                cv2.line(img, pt_start, pt_end, (0, 255, 0), 3)

        out_path = os.path.join(output_dir, f'result_{img_name}')
        cv2.imwrite(out_path, img)

    print(f"All done! Results saved to {output_dir}")
