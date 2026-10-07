#!/usr/bin/env python3
"""DeepStream (pyds) version of the detection stage - UNTESTED (no NVIDIA GPU in the dev sandbox).

GPU pipeline:  filesrc -> decode (NVDEC) -> nvstreammux -> nvinfer(person, gie 2) -> nvinfer(buoy, gie 1)
               -> nvtracker (NvDCF) -> nvdsosd -> encode -> mp4
A pad probe collects every frame's skier/buoy boxes and writes <out>/<tag>.frames.json + <tag>.json in
the same format as cvdemo.pipeline, so the course logic runs unchanged on top:

    python3 ds_pipeline.py VIDEO --out ../../runs/deepstream
    python3 ../../tools/rescore.py --run ../../runs/deepstream --render    # pan + pass logic + video

Run inside the DeepStream container (see README.md), with DeepStream-Yolo's parser built.
"""
import argparse
import json
import sys
from pathlib import Path

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

import pyds  # noqa: E402

HERE = Path(__file__).resolve().parent
PERSON_GIE, BUOY_GIE = 2, 1
TRACKER_CFG = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml"

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--out", default=str(HERE / "../../runs/deepstream"))
ap.add_argument("--width", type=int, default=1920)
ap.add_argument("--height", type=int, default=1080)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
tag = Path(a.video).stem.replace(" ", "_")
frames = []


def make(kind, name, **props):
    e = Gst.ElementFactory.make(kind, name)
    if e is None:
        sys.exit(f"cannot create {kind}")
    for k, v in props.items():
        e.set_property(k.replace("_", "-"), v)
    return e


def probe(pad, info, _):
    batch = pyds.gst_buffer_get_nvds_batch_meta(hash(info.get_buffer()))
    l_frame = batch.frame_meta_list
    while l_frame is not None:
        fm = pyds.NvDsFrameMeta.cast(l_frame.data)
        persons, buoys = [], []
        l_obj = fm.obj_meta_list
        while l_obj is not None:
            o = pyds.NvDsObjectMeta.cast(l_obj.data)
            r = o.rect_params
            box = [r.left, r.top, r.left + r.width, r.top + r.height]
            if o.unique_component_id == BUOY_GIE:
                buoys.append(box + [o.confidence])
            elif o.unique_component_id == PERSON_GIE and o.class_id == 0:
                persons.append((box, o.confidence, o.object_id))
            l_obj = l_obj.next
        rec = dict(pan=0.0, skier=None, kpts=None, buoys=buoys, skier_X=None, foot_y=None, boats=None)
        if persons:  # skier = largest confident person (course logic re-selects by continuity anyway)
            b, c, tid = max(persons, key=lambda p: p[1] * (p[0][2] - p[0][0]) * (p[0][3] - p[0][1]))
            rec.update(skier=b + [c, int(tid) & 0x7FFFFFFF], skier_X=(b[0] + b[2]) / 2, foot_y=b[3])
        frames.append(rec)
        l_frame = l_frame.next
    return Gst.PadProbeReturn.OK


Gst.init(None)
pipe = Gst.Pipeline()
src = make("filesrc", "src", location=str(Path(a.video).resolve()))
demux = make("qtdemux", "demux")
parse = make("h264parse", "parse")
dec = make("nvv4l2decoder", "dec")
mux = make("nvstreammux", "mux", width=a.width, height=a.height, batch_size=1, batched_push_timeout=40000)
pgie_person = make("nvinfer", "person", config_file_path=str(HERE / "config_infer_person.txt"))
pgie_buoy = make("nvinfer", "buoy", config_file_path=str(HERE / "config_infer_buoy.txt"))
tracker = make("nvtracker", "tracker", ll_lib_file="/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
               ll_config_file=TRACKER_CFG, tracker_width=960, tracker_height=544)
conv = make("nvvideoconvert", "conv")
osd = make("nvdsosd", "osd")
conv2 = make("nvvideoconvert", "conv2")
enc = make("nvv4l2h264enc", "enc", bitrate=8000000)
parse2 = make("h264parse", "parse2")
mp4 = make("qtmux", "mp4")
sink = make("filesink", "sink", location=str(out / f"{tag}_deepstream.mp4"), sync=False)
for e in (src, demux, parse, dec, mux, pgie_person, pgie_buoy, tracker, conv, osd, conv2, enc, parse2, mp4, sink):
    pipe.add(e)
src.link(demux)
demux.connect("pad-added", lambda d, p: p.link(parse.get_static_pad("sink")) if "video" in p.get_name() else None)
parse.link(dec)
dec.get_static_pad("src").link(mux.request_pad_simple("sink_0"))
for x, y in [(mux, pgie_person), (pgie_person, pgie_buoy), (pgie_buoy, tracker), (tracker, conv), (conv, osd),
             (osd, conv2), (conv2, enc), (enc, parse2), (parse2, mp4), (mp4, sink)]:
    x.link(y)
osd.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, probe, None)

loop = GLib.MainLoop()
bus = pipe.get_bus()
bus.add_signal_watch()
bus.connect("message::eos", lambda *_: loop.quit())
bus.connect("message::error", lambda _, m: (print(m.parse_error()), loop.quit()))
pipe.set_state(Gst.State.PLAYING)
loop.run()
pipe.set_state(Gst.State.NULL)

import cv2  # noqa: E402

cap = cv2.VideoCapture(a.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
sx, sy = W / a.width, H / a.height  # boxes are in streammux resolution -> back to video pixels
for r in frames:
    r["buoys"] = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy, b[4]] for b in r["buoys"]]
    if r["skier"]:
        s = r["skier"]
        r["skier"] = [s[0] * sx, s[1] * sy, s[2] * sx, s[3] * sy, s[4], s[5]]
        r["skier_X"], r["foot_y"] = (r["skier"][0] + r["skier"][2]) / 2, r["skier"][3]
meta = dict(video=str(Path(a.video).resolve()), fps=fps, frames=len(frames), width=W, height=H, shots=[0])
(out / f"{tag}.frames.json").write_text(json.dumps(frames))
(out / f"{tag}.json").write_text(json.dumps(dict(meta=meta, models=dict(skier="deepstream person", pose="-",
                                                                         buoy="deepstream buoy"), score={})))
print("wrote", out / f"{tag}.frames.json")
