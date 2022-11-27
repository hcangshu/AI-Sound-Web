import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
import uuid


import matplotlib.pyplot as plt

import os
import json
import math

import scipy
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
#from transforms import Translate

from scipy.io.wavfile import write
import argparse
parser = argparse.ArgumentParser(description='查看传参')
parser.add_argument("--text",type=str,default="你好。")
parser.add_argument("--character",type=int,default=0)
args = parser.parse_args()
speakers = ['派蒙', '凯亚', '安柏', '丽莎', '琴', '香菱', '枫原万叶', '迪卢克', '温迪', '可莉', '早柚', '托马', '芭芭拉', '优菈', '云堇', '钟离',
           '魈', '凝光', '雷电将军', '北斗', '甘雨', '七七', '刻晴', '神里绫华', '戴因斯雷布', '雷泽', '神里绫人', '罗莎莉亚', '阿贝多', '八重神子',
           '宵宫', '荒泷一斗', '九条裟罗', '夜兰', '珊瑚宫心海', '五郎', '散兵', '女士', '达达利亚', '莫娜', '班尼特', '申鹤', '行秋', '烟绯',
           '久岐忍', '辛焱', '砂糖', '胡桃', '重云', '菲谢尔', '诺艾尔', '迪奥娜', '鹿野院平藏']


class vits:
    def __init__(self):
        self.load_moudle()

    def get_text(self, text):
        text_norm = text_to_sequence(text, self.hps.data.text_cleaners)
        if self.hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        cc = torch.LongTensor(text_norm)
        return cc

    def load_moudle(self):
        self.hps = utils.get_hparams_from_file("./configs/ys.json")

        self.net_g = SynthesizerTrn(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,  #
            **self.hps.model)
        _ = self.net_g.eval()

        _ = utils.load_checkpoint("./ys/ys.pth", self.net_g, None)  # G_389000.pth

    def get_re(self, text, speaker):
        character = speakers.index(speaker)
        stn_tst = self.get_text(text)
        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
            sid = torch.LongTensor([character])  # 指定第几个人说话
            audio = \
                self.net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, sid=sid, noise_scale_w=0.8, length_scale=1.2)[
                    0][
                    0, 0].data.cpu().float().numpy()
            uui = uuid.uuid1()
            scipy.io.wavfile.write("./record/%s.wav" % uui, self.hps.data.sampling_rate, audio)
            return './record/%s.wav' % uui


vits = vits()
app = FastAPI()


@app.get('/run')
def run_moudle(speaker, text):
    print(text, speaker)
    ava = vits.get_re(text, speaker)
    return FileResponse(ava)


uvicorn.run(app, host='127.0.0.1', port=8088)
