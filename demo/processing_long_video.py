import os
import cv2
import json
import argparse
import os.path as osp
import torch
import numpy as np
from tqdm import tqdm
from mmengine.config import Config, DictAction
from mmengine.runner.amp import autocast
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg
from torchvision.ops import nms

import supervision as sv
from norfair import Detection as NorfairDetection, Tracker


# Annotators
BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=1)
MASK_ANNOTATOR = sv.MaskAnnotator()

class LabelAnnotator(sv.LabelAnnotator):
    @staticmethod
    def resolve_text_background_xyxy(center_coordinates, text_wh, position):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh
        return center_x, center_y, center_x + text_w, center_y + text_h

LABEL_ANNOTATOR = LabelAnnotator(text_padding=4, text_scale=0.5, text_thickness=1)

# Tracker setup
tracker = Tracker(
    distance_function="euclidean",
    distance_threshold=40,
    initialization_delay=0,
    hit_counter_max=15
)

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-World Video to JSON with Tracking')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint path')
    parser.add_argument('video', help='Input video path')
    parser.add_argument('text', help='Text prompts (comma separated or txt file)')
    parser.add_argument('--topk', type=int, default=100)
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', default='/content/drive/MyDrive/yolo_world_outputs')  # ✅ Drive-safe default
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


def extract_frames(video_path, frame_dir):
    os.makedirs(frame_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_path = os.path.join(frame_dir, f'frame_{frame_idx:05d}.jpg')
        cv2.imwrite(frame_path, frame)
        frame_paths.append((frame_path, frame_idx))
        frame_idx += 1

    cap.release()
    return frame_paths, fps


def run_inference(model, image_path, texts, test_pipeline, topk=100, score_thr=0.1, use_amp=False):
    data_info = dict(img_id=0, img_path=image_path, texts=texts)
    data_info = test_pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])

    with autocast(enabled=use_amp), torch.no_grad():
        output = model.test_step(data_batch)[0]
        pred_instances = output.pred_instances
        pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

    if len(pred_instances.scores) > topk:
        indices = pred_instances.scores.float().topk(topk)[1]
        pred_instances = pred_instances[indices]

    boxes = torch.tensor(pred_instances.bboxes)
    scores = torch.tensor(pred_instances.scores)
    keep_idxs = nms(boxes, scores, iou_threshold=0.5)
    pred_instances = pred_instances[keep_idxs]

    return pred_instances.cpu().numpy()


def yolo_to_norfair_detections(pred):
    detections = []
    for bbox in pred['bboxes']:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        points = np.array([[cx, cy]])
        detections.append(NorfairDetection(points=points))
    return detections


def assign_ids_to_detections_from_tracker(pred, tracked_objects):
    results = []
    pred_centers = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in pred['bboxes']]
    used_tracker_ids = set()

    for i, ((x1, y1, x2, y2), label, score, center) in enumerate(
        zip(pred['bboxes'], pred['labels'], pred['scores'], pred_centers)
    ):
        assigned_id = None
        best_dist = float('inf')

        for tracked in tracked_objects:
            tid = tracked.id
            if tid in used_tracker_ids:
                continue
            tracked_center = tracked.estimate[0]
            dist = np.linalg.norm(np.array(center) - tracked_center)
            if dist < best_dist:
                best_dist = dist
                assigned_id = tid

        if assigned_id is not None:
            used_tracker_ids.add(assigned_id)

        results.append({
            "id": assigned_id,
            "label": label,
            "score": score.item(),
            "bbox": [round(x1), round(y1), round(x2), round(y2)],
            "center": [round(center[0]), round(center[1])]
        })

    return results


def annotate_image_with_ids(image_path, detection_data, texts):
    image = cv2.imread(image_path)
    for det in detection_data:
        x1, y1, x2, y2 = det["bbox"]
        obj_id = det["id"]
        label = texts[det["label"]][0]
        score = det["score"]
        center_x, center_y = det["center"]
        text = f"{label} {score:.2f} ID:{obj_id}"
        cv2.putText(image, text, (x1, y1 - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 0, 255),
                    thickness=2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return image


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.load_from = args.checkpoint

    model = init_detector(cfg, checkpoint=args.checkpoint, device=args.device)
    test_pipeline = Compose(get_test_pipeline_cfg(cfg=cfg))

    if args.text.endswith('.txt'):
        with open(args.text) as f:
            texts = [[line.strip()] for line in f.readlines()] + [[' ']]
    else:
        texts = [[t.strip()] for t in args.text.split(',')] + [[' ']]
    model.reparameterize(texts)

    os.makedirs(args.output_dir, exist_ok=True)
    frame_dir = osp.join(args.output_dir, 'frames')
    json_dir = osp.join(args.output_dir, 'frame_outputs')
    annotated_dir = osp.join(args.output_dir, 'annotated_frames')
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(annotated_dir, exist_ok=True)

    frame_paths, fps = extract_frames(args.video, frame_dir)

    for frame_path, idx in tqdm(frame_paths, desc="Processing frames"):
        json_path = os.path.join(json_dir, f"frame_{idx:05d}.json")
        if os.path.exists(json_path):
            continue  # ✅ Skip already-processed frame

        pred = run_inference(
            model, frame_path, texts, test_pipeline,
            topk=args.topk, score_thr=args.threshold, use_amp=args.amp
        )

        norfair_dets = yolo_to_norfair_detections(pred)
        tracked_objects = tracker.update(detections=norfair_dets)
        detections_with_ids = assign_ids_to_detections_from_tracker(pred, tracked_objects)

        out = {
            "frame": idx,
            "time_sec": round(idx / fps, 2),
            "detections": [
                {
                    "id": det["id"],
                    "label": texts[det["label"]][0],
                    "score": round(det["score"], 3),
                    "bbox": det["bbox"]
                }
                for det in detections_with_ids
            ]
        }

        with open(json_path, "w") as f:
            json.dump(out, f, indent=2)

        annotated = annotate_image_with_ids(frame_path, detections_with_ids, texts)
        cv2.imwrite(os.path.join(annotated_dir, f"frame_{idx:05d}.jpg"), annotated)


if __name__ == '__main__':
    main()
