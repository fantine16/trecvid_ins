import json
import math
import argparse


def main(params):

    out = []
    local = params['local']
    if local:
        prefix = '/media/liuchang/tv16exp01/tv16ins/data/tv15.INS/Frame_1fps/'
        out_json = 'data/shot_frame_local.json'
    else:
        prefix = '/datacenter/1/tv16ins/data/tv15.INS/Frame_1fps/'
        out_json = 'data/shot_frame_201.json'
    shot_file = open("data/eastenders.masterShotReferenceTable", "r")
    while True:
        line = shot_file.readline()
        if line:
            term = {}
            line = line.strip()
            if len(line) == 0:
                break
            term["video_index"] = line.split(' ')[0]

            line = line[len(line.split(' ')[0]):len(line)].strip()
            term["name"] = line.split(' ')[0]
            line = line[len(line.split(' ')[0]):len(line)].strip()
            term["shot_index"] = term["name"].split("_")[1]
            # print(term["shot_index"])
            term["start_time"] = line.split(' ')[0][1:12].split(":")
            line = line[len(line.split(' ')[0]):len(line)].strip()
            term["end_time"] = line.split(' ')[0][1:12].split(":")
            term["start_second"] = int(term["start_time"][0]) * 3600 + int(term["start_time"][1]) * 60 + int(
                term["start_time"][2]) + int(
                term["start_time"][3]) / 60.0
            term["end_second"] = int(term["end_time"][0]) * 3600 + int(term["end_time"][1]) * 60 + int(
                term["end_time"][2]) + int(term["end_time"][3]) / 60.0
            if math.ceil(term["start_second"]) > math.floor(term["end_second"]):
                term["images_ix"] = []
            else:
                term["images_ix"] = range(int(math.ceil(term["start_second"])), int(math.floor(term["end_second"]) + 1))
            term["images_num"] = len(term["images_ix"])
            term["img_path"] = []
            for i, t in enumerate(term["images_ix"]):
                img_path = prefix + term["video_index"] + "/" + str(t + 1) + ".1fps.png"
                term["img_path"].append(img_path)
            out.append(term)
            if len(out) % 1000 == 0:
                print(term["name"])
        else:
            break
    json.dump(out, open(out_json, 'w'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--local', default=False)
    args = parser.parse_args()
    params = vars(args)
    main(params)