# -*- coding: utf-8 -*-
import sys
import os
import torch
import utils
# from config import Config
from infer import infer, latest_version, get_net_g


def test_text_fn(text, language = "ZH"):
    # from infer import get_text
    # bert, ja_bert, en_bert, phones, tones, lang_ids = get_text(
    #     text, language, hps, device
    # )

    from text.cleaner import clean_text
    norm_text, phone, tone, word2ph = clean_text(text, language)
    print(norm_text)
    print(phone)
    print(tone)
    print(word2ph)

    # from text import cleaned_text_to_sequence, get_bert
    # phone, tone, language = cleaned_text_to_sequence(phone, tone, language)


if __name__ == "__main__":
    root_path = os.path.abspath(os.path.join('.', '.'))
    config_path = os.path.join(root_path, 'Data/configs/config.json')
    hps = utils.get_hparams_from_file(config_path)
    language = "ZH"
    device = "cuda"
    

    '''
    日前,全国广电融媒数字主持人工作室在北京正式成立。该工作室由北京广电实战、北京灵境赛博和中国科学技术大学“合成现实”联合实验室三家单位共同发起。工作室将在广播电视主持人领域大力探索人工智能技术的应用与实践。
    工作室计划重点围绕新闻及文艺类数字主持人系统开展研发。通过数字孪生、语音合成、人脸渲染等前沿技术,实现主持人的智能化升级,包括外形、语言、神态等方面的数字化仿真，力求在播报水平，语音情感，表情神态上实现广电级的数字主持人的打造。未来在短视频平台，乃至大荧幕中将可能出现更多由数字主持人主持的内容。
    业内专家指出,数字主持人的最大优势在于内容迭代能力强,还可以实现7×24小时不间断高效工作。这不仅将大幅提升广电融媒的内容制作的效率,也将显著降低制作成本。
    专家认为,数字主持人技术的应用将加速我国广电融媒实现内容制作智能化、节目形式多样化, 对构建新时代中国特色主流媒体起到促进作用。目前我国广电融媒正处于融合发展的关键时期,数字主持人的应用具有重大意义。
    展望未来,全国广电融媒数字主持人工作室将持续推动主持人领域的智能化升级,为我国广电融媒的发展提供强大的技术支撑。相信数字主持人必将成为广电舞台的一道亮丽风景线。
    '''
    '''
    "日前，"
    "全国广电融媒数字主持人工作室在北京正式成立。"
    "该工作室由北京广电实战、北京灵境赛博和中国科学技术大学合成现实联合实验室三家单位共同发起。"
    "工作室将在广播电视主持人领域大力探索人工智能技术的应用与实践。"
    '''
    text = (
        "北京灵境赛博和中国科学技术大学合成现实联合实验室三家单位共同发起。"
    )

    test_text_fn(text)

    import jieba
    jieba.add_word('北京广电实战')
    jieba.add_word('北京灵境赛博')
    jieba.add_word('中国科学技术大学')
    jieba.add_word('合成现实联合实验室')
    test_text_fn(text)






