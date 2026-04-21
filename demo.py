#以下这段代码是原来的测试代码（已经弃用）
# import torch, os, cv2
# from utils.dist_utils import dist_print
# import torch, os
# from utils.common import merge_config, get_model
# import tqdm
# import torchvision.transforms as transforms
# from data.dataset import LaneTestDataset
#
# def pred2coords(pred, row_anchor, col_anchor, local_width = 1, original_image_width = 1640, original_image_height = 590):
#     batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
#     batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape
#
#     max_indices_row = pred['loc_row'].argmax(1).cpu()
#     # n , num_cls, num_lanes
#     valid_row = pred['exist_row'].argmax(1).cpu()
#     # n, num_cls, num_lanes
#
#     max_indices_col = pred['loc_col'].argmax(1).cpu()
#     # n , num_cls, num_lanes
#     valid_col = pred['exist_col'].argmax(1).cpu()
#     # n, num_cls, num_lanes
#
#     pred['loc_row'] = pred['loc_row'].cpu()
#     pred['loc_col'] = pred['loc_col'].cpu()
#
#     coords = []
#
#     row_lane_idx = [1,2]
#     col_lane_idx = [0,3]
#
#     for i in row_lane_idx:
#         tmp = []
#         if valid_row[0,:,i].sum() > num_cls_row / 2:
#             for k in range(valid_row.shape[1]):
#                 if valid_row[0,k,i]:
#                     all_ind = torch.tensor(list(range(max(0,max_indices_row[0,k,i] - local_width), min(num_grid_row-1, max_indices_row[0,k,i] + local_width) + 1)))
#
#                     out_tmp = (pred['loc_row'][0,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5
#                     out_tmp = out_tmp / (num_grid_row-1) * original_image_width
#                     tmp.append((int(out_tmp), int(row_anchor[k] * original_image_height)))
#             coords.append(tmp)
#
#     for i in col_lane_idx:
#         tmp = []
#         if valid_col[0,:,i].sum() > num_cls_col / 4:
#             for k in range(valid_col.shape[1]):
#                 if valid_col[0,k,i]:
#                     all_ind = torch.tensor(list(range(max(0,max_indices_col[0,k,i] - local_width), min(num_grid_col-1, max_indices_col[0,k,i] + local_width) + 1)))
#
#                     out_tmp = (pred['loc_col'][0,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5
#
#                     out_tmp = out_tmp / (num_grid_col-1) * original_image_height
#                     tmp.append((int(col_anchor[k] * original_image_width), int(out_tmp)))
#             coords.append(tmp)
#
#     return coords
# if __name__ == "__main__":
#     torch.backends.cudnn.benchmark = True
#
#     args, cfg = merge_config()
#     cfg.batch_size = 1
#     print('setting batch_size to 1 for demo generation')
#
#     dist_print('start testing...')
#     assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']
#
#     if cfg.dataset == 'CULane':
#         cls_num_per_lane = 18
#     elif cfg.dataset == 'Tusimple':
#         cls_num_per_lane = 56
#     else:
#         raise NotImplementedError
#
#     net = get_model(cfg)
#
#     state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
#     compatible_state_dict = {}
#     for k, v in state_dict.items():
#         if 'module.' in k:
#             compatible_state_dict[k[7:]] = v
#         else:
#             compatible_state_dict[k] = v
#
#     net.load_state_dict(compatible_state_dict, strict=False)
#     net.eval()
#
#     img_transforms = transforms.Compose([
#         transforms.Resize((int(cfg.train_height / cfg.crop_ratio), cfg.train_width)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#     ])
#     if cfg.dataset == 'CULane':
#         splits = ['test0_normal.txt', 'test1_crowd.txt', 'test2_hlight.txt', 'test3_shadow.txt', 'test4_noline.txt', 'test5_arrow.txt', 'test6_curve.txt', 'test7_cross.txt', 'test8_night.txt']
#         datasets = [LaneTestDataset(cfg.data_root,os.path.join(cfg.data_root, 'list/test_split/'+split),img_transform = img_transforms, crop_size = cfg.train_height) for split in splits]
#         img_w, img_h = 1640, 590
#     elif cfg.dataset == 'Tusimple':
#         splits = ['test.txt']
#         datasets = [LaneTestDataset(cfg.data_root,os.path.join(cfg.data_root, split),img_transform = img_transforms, crop_size = cfg.train_height) for split in splits]
#         img_w, img_h = 1280, 720
#     else:
#         raise NotImplementedError
#     for split, dataset in zip(splits, datasets):
#         loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle = False, num_workers=1)
#         fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#         print(split[:-3]+'avi')
#         vout = cv2.VideoWriter(split[:-3]+'avi', fourcc , 30.0, (img_w, img_h))
#         for i, data in enumerate(tqdm.tqdm(loader)):
#             imgs, names = data
#             imgs = imgs.cuda()
#             with torch.no_grad():
#                 pred = net(imgs)
#
#             vis = cv2.imread(os.path.join(cfg.data_root,names[0]))
#             coords = pred2coords(pred, cfg.row_anchor, cfg.col_anchor, original_image_width = img_w, original_image_height = img_h)
#             for lane in coords:
#                 for coord in lane:
#                     cv2.circle(vis,coord,5,(0,255,0),-1)
#             vout.write(vis)
#
#         vout.release()

#以下这段代码是我重写的代码
import torch, os, cv2
import numpy as np
from utils.dist_utils import dist_print
import torch, os
from utils.common import merge_config, get_model
import tqdm
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset
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
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()
    cfg.batch_size = 1
    print('setting batch_size to 1 for demo generation')

    dist_print('start testing...')
    assert cfg.backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

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


    if args.input_path:
        # 判断输入是视频还是图片
        if os.path.splitext(args.input_path)[1].lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            cap = cv2.VideoCapture(args.input_path)
            if not cap.isOpened():
                print(f"Cannot open video: {args.input_path}")
                exit()

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            output_video_path = os.path.join('images', 'result_' + os.path.basename(args.input_path))
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            print(f"Processing video: {args.input_path}")
            print(f"Saving result to: {output_video_path}")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img_tensor = img_transforms(img_pil).unsqueeze(0).cuda()
                # 【关键修复】：切掉上面部分，只保留底部路面区域
                img_tensor = img_tensor[:, :, -cfg.train_height:, :]

                with torch.no_grad():
                    pred = net(img_tensor)


                coords = pred2coords(pred, cfg.row_anchor, cfg.col_anchor, original_image_width=width,
                                     original_image_height=height)

                for lane in coords:
                    if len(lane) > 1:
                        x_coords = [p[0] for p in lane]
                        y_coords = [p[1] for p in lane]

                        # 找出车道线实际的起始（底部）和结束（顶部）位置
                        min_y, max_y = min(y_coords), max(y_coords)

                        # 拟合直线 x = ay + b
                        coeffs = np.polyfit(y_coords, x_coords, 1)

                        # 直线的起点和终点限定在实际检测范围内
                        pt_start = (int(np.polyval(coeffs, max_y)), max_y)
                        pt_end = (int(np.polyval(coeffs, min_y)), min_y)

                        cv2.line(frame, pt_start, pt_end, (0, 255, 0), 3)

                out.write(frame)


                cv2.imshow('Lane Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            out.release()
            cv2.destroyAllWindows()
            print(f"Video processing complete. Result saved to {output_video_path}")
            exit()

        # 处理图片
        img = cv2.imread(args.input_path)
        if img is None:
            print(f"Cannot read image: {args.input_path}")
            exit()

        img_h, img_w = img.shape[:2]
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_tensor = img_transforms(img_pil).unsqueeze(0).cuda()
        # 【关键修复】：切掉上面部分，只保留底部路面区域
        img_tensor = img_tensor[:, :, -cfg.train_height:, :]

        with torch.no_grad():
            pred = net(img_tensor)


        coords = pred2coords(pred, cfg.row_anchor, cfg.col_anchor, original_image_width=img_w,
                             original_image_height=img_h)

        for lane in coords:
            if len(lane) > 1:
                x_coords = [p[0] for p in lane]
                y_coords = [p[1] for p in lane]

                # 找出车道线实际的起始（底部）和结束（顶部）位置
                min_y, max_y = min(y_coords), max(y_coords)

                # 拟合直线 x = ay + b
                coeffs = np.polyfit(y_coords, x_coords, 1)

                # 直线的起点和终点限定在实际检测范围内
                pt_start = (int(np.polyval(coeffs, max_y)), max_y)
                pt_end = (int(np.polyval(coeffs, min_y)), min_y)

                cv2.line(img, pt_start, pt_end, (0, 255, 0), 3)

        cv2.imshow('Lane Detection', img)
        ext = os.path.splitext(args.input_path)[1]
        base_name = os.path.splitext(os.path.basename(args.input_path))[0]
        output_path = f'images/result_{base_name}{ext}'
        cv2.imwrite(output_path, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit()


        # 处理图片
        img = cv2.imread(args.input_path)
        if img is None:
            print(f"Cannot read image: {args.input_path}")
            exit()

        img_h, img_w = img.shape[:2]
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_tensor = img_transforms(img_pil).unsqueeze(0).cuda()

        with torch.no_grad():
            pred = net(img_tensor)

        coords = pred2coords(pred, cfg.row_anchor, cfg.col_anchor, original_image_width=img_w,
                             original_image_height=img_h)
        for lane in coords:
            for coord in lane:
                cv2.circle(img, coord, 5, (0, 255, 0), -1)

        cv2.imshow('Lane Detection', img)
        ext = os.path.splitext(args.input_path)[1]
        base_name = os.path.splitext(os.path.basename(args.input_path))[0]
        output_path = f'images/result_{base_name}{ext}'
        cv2.imwrite(output_path, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit()

    if cfg.dataset == 'CULane':
        splits = ['test0_normal.txt', 'test1_crowd.txt', 'test2_hlight.txt', 'test3_shadow.txt', 'test4_noline.txt',
                  'test5_arrow.txt', 'test6_curve.txt', 'test7_cross.txt', 'test8_night.txt']
        datasets = [LaneTestDataset(cfg.data_root, os.path.join(cfg.data_root, 'list/test_split/' + split),
                                    img_transform=img_transforms, crop_size=cfg.train_height) for split in splits]
        img_w, img_h = 1640, 590
    elif cfg.dataset == 'Tusimple':
        splits = ['test.txt']
        datasets = [LaneTestDataset(cfg.data_root, os.path.join(cfg.data_root, split), img_transform=img_transforms,
                                    crop_size=cfg.train_height) for split in splits]
        img_w, img_h = 1280, 720
    else:
        raise NotImplementedError
    for split, dataset in zip(splits, datasets):
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        print(split[:-3] + 'avi')
        vout = cv2.VideoWriter(split[:-3] + 'avi', fourcc, 30.0, (img_w, img_h))
        for i, data in enumerate(tqdm.tqdm(loader)):
            imgs, names = data
            imgs = imgs.cuda()
            with torch.no_grad():
                pred = net(imgs)

            vis = cv2.imread(os.path.join(cfg.data_root, names[0]))
            coords = pred2coords(pred, cfg.row_anchor, cfg.col_anchor, original_image_width=img_w,
                                 original_image_height=img_h)
            for lane in coords:
                for coord in lane:
                    cv2.circle(vis, coord, 5, (0, 255, 0), -1)
            vout.write(vis)

        vout.release()
