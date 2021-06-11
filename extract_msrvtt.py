import os
import json
import cv2
import pdb
import pandas as pd
import lmdb
import pickle
from tqdm import tqdm
from collections import defaultdict
from encoders_extract_features import FRCNNExtractor

DATA_INFO_MAP = {
    "msrvtt": {
        "in_dir": "/mnt/data1/vincent/proj/soco/trec/data/msrvtt/raw/Video-frames-hdbscan",
        "out_dir": "/mnt/data1/vincent/proj/soco/trec/data/msrvtt/features/{}_frcnn.lmdb",
        "ids": {
            "train9k": "/mnt/data1/vincent/proj/soco/trec/data/msrvtt/raw/MSRVTT_train.9k.csv",
            "train7k": "/mnt/data1/vincent/proj/soco/trec/data/msrvtt/raw/MSRVTT_train.7k.csv",
            "test": "/mnt/data1/vincent/proj/soco/trec/data/msrvtt/raw/MSRVTT_JSFUSION_test.csv"
            },
        "captions": {
            "train9k": "/mnt/data1/vincent/proj/soco/trec/data/msrvtt/raw/MSRVTT_data.json",
            "train7k": "/mnt/data1/vincent/proj/soco/trec/data/msrvtt/raw/MSRVTT_data.json",
            "test": "/mnt/data1/vincent/proj/soco/trec/data/msrvtt/raw/MSRVTT_JSFUSION_test.csv"
        }
    }
}


def get_name_from_path(path):
    def strip_path(path):
        return path.rsplit('/', 1)[-1]

    def remove_ext(name):
        return name.rsplit('.', 1)[0]

    return remove_ext(strip_path(path))


def get_vid_ids_from_file(data_name, path):
    if data_name == "msrvtt":
        # read from csv
        data = pd.read_csv(path)
        return data["video_id"].values
    else:
        raise NotImplementedError


def get_vid_id_caption_map_from_file(data_name, path):
    img_id_caption_map = defaultdict(list)
    if data_name == "msrvtt":
        if path.endswith("json"):
            data = json.load(open(path))["sentences"]
            for d in data:
                img_id_caption_map[d["video_id"]].append(d["caption"])
        elif path.endswith("csv"):
            data = pd.read_csv(path)
            for vid, sent in zip(data["video_id"].values, data["sentence"].values):
                img_id_caption_map[vid].append(sent)

    return img_id_caption_map


if __name__ == "__main__":
    # set the data needed to extract
    data_splits = {"msrvtt": ["test"]}

    extractor = FRCNNExtractor("resources/frcnn-bua-caffe-r101-with-attrs")


    for data_name, splits in data_splits.items():
        for split in splits:
            vid_dir = DATA_INFO_MAP[data_name]["in_dir"].format(split) \
                if "{}" in DATA_INFO_MAP[data_name]["in_dir"]\
                else DATA_INFO_MAP[data_name]["in_dir"]

            lmdb_save_dir = DATA_INFO_MAP[data_name]["out_dir"].format(split)
            img_path_list = os.listdir(vid_dir)

            # filter keys and captions by split
            vid_ids_needed = get_vid_ids_from_file(data_name, DATA_INFO_MAP[data_name]["ids"][split])
            captions = get_vid_id_caption_map_from_file(data_name, DATA_INFO_MAP[data_name]["captions"][split])
            vid_path_needed = [_ for _ in img_path_list if _ in vid_ids_needed]
            captions_needed = {_k: _v for _k, _v in captions.items() if _k in vid_ids_needed}
            print("get {}/{} needed video paths".format(vid_path_needed, img_path_list))

            env = lmdb.open(lmdb_save_dir, map_size=1024**4)
            txn = env.begin(write=True)
            # first write meta info ("keys", "captions")
            meta_map = {"keys": vid_path_needed, "captions": captions_needed}
            for i, (k, v) in enumerate(meta_map.items()):
                txn.put(k.encode(), pickle.dumps(v))

            # extract for each video
            vid_feat_map = {}
            for i, sub_dir in enumerate(tqdm(vid_path_needed)):
                one_vid_imgs = []
                p_dir = os.path.join(vid_dir, sub_dir, "final_images")
                # sort by timestamp
                imgs = sorted([_ for _ in os.listdir(p_dir) if _.endswith("jpeg")], key=lambda x: float(get_name_from_path(x).rsplit("_", 1)[-1]))
                for img in imgs:
                    one_vid_imgs.append(cv2.imread(os.path.join(p_dir, img)))
                frcnn_results, metas = extractor.batch_extract_feat(one_vid_imgs, batch_size=1)
                assert len(frcnn_results) == len(imgs)
                for ii, r in enumerate(frcnn_results):
                    r["img_id"] = imgs[ii]
                # vid_feat_map[sub_dir] = frcnn_results
                # key: video_id value: [{"img_id":,"objects":,"img_feat":},...]
                txn.put(sub_dir.encode(), pickle.dumps(frcnn_results))

                if i % 100 == 0:
                    txn.commit()
                    txn = env.begin(write=True)

            txn.commit()
            env.close()
