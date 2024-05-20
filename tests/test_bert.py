# -*- coding: utf-8 -*-
import sys
sys.path.append('.') # win vscode debug 
sys.path.append('..')
sys.path.append('../..')
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import pprint
import torch
from transformers import AutoTokenizer, MegatronBertModel
# import utils
# from config import config
# from infer import infer, latest_version, get_net_g


def test_text_normalize(language = "ZH"):
    # from text.cleaner import clean_text
    def clean_text(text):
        from text import chinese    
        norm_text = chinese.text_normalize(text)
        phones, tones, word2ph = chinese.g2p(norm_text)
        return norm_text, phones, tones, word2ph
    s = (
        "日前,全国广电融媒数字主持人工作室在北京正式成立。该工作室由北京广电实战、北京灵境赛博和中国科学技术大学“合成现实”联合实验室三家单位共同发起。工作室将在广播电视主持人领域大力探索人工智能技术的应用与实践。"
        # "工作室计划重点围绕新闻及文艺类数字主持人系统开展研发。通过数字孪生、语音合成、人脸渲染等前沿技术,实现主持人的智能化升级,包括外形、语言、神态等方面的数字化仿真，力求在播报水平，语音情感，表情神态上实现广电级的数字主持人的打造。"
        # "未来在短视频平台，乃至大荧幕中将可能出现更多由数字主持人主持的内容。业内专家指出,数字主持人的最大优势在于内容迭代能力强,还可以实现7×24小时不间断高效工作。这不仅将大幅提升广电融媒的内容制作的效率,也将显著降低制作成本。"
        # "专家认为,数字主持人技术的应用将加速我国广电融媒实现内容制作智能化、节目形式多样化, 对构建新时代中国特色主流媒体起到促进作用。目前我国广电融媒正处于融合发展的关键时期,数字主持人的应用具有重大意义。"
        # "展望未来,全国广电融媒数字主持人工作室将持续推动主持人领域的智能化升级,为我国广电融媒的发展提供强大的技术支撑。相信数字主持人必将成为广电舞台的一道亮丽风景线。"
        '\n'
    )
    print(s)
    norm_text, phone, tone, word2ph = clean_text(s)
    print(norm_text, phone, tone, word2ph)

    import jieba
    for i in [
        '北京广电实战',
        '北京灵境赛博',
        '中国科学技术大学',
        '合成现实联合实验室',
    ]:
        jieba.add_word(i)
    norm_text, phone, tone, word2ph = clean_text(s)
    print(norm_text, phone, tone, word2ph)
    return norm_text, phone, tone, word2ph


def test_g2p_mix(text):
    '''
    "Erlangshen-MegatronBert-1.3B": {
        "repo_id": "Fengshenbang/Erlangshen-MegatronBert-1.3B",
        "files": ["pytorch_model.bin"]
    }
    '''
    from g2p_mix import G2pMix
    g2per = G2pMix()
    fpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../bert/IDEA-CCNL/Erlangshen-MegatronBert-1.3B")) # repo_id
    tokenizer = AutoTokenizer.from_pretrained(fpath)
    device = "cuda"
    this_model = MegatronBertModel.from_pretrained(fpath).to(device) # torch_dtype=torch.float16
    '''
    注释断言
    bert中英文emb建模单元不同；英文bpe；
    先按单词合并，再按word2ph、音素分配；
    求和再平均
    '''
    phones = [ i for i in g2per.g2p(text, sandhi=True) ]
    # C-TODO word2ph
    word2ph = [ len(i.phones) for i in phones ]
    word2ph = [1] + word2ph + [1]
    words = [ i.word.lower() for i in phones ]
    words = ["[CLS]"] + words + ["[SEP]"]

    def align_bpes_to_words(bpes, words):
        wds_pt, bpes_pt, lst = 0, 0, []
        while wds_pt < len(words) and bpes_pt < len(bpes):
            if words[wds_pt] == bpes[bpes_pt] or bpes[bpes_pt] == "[UNK]":
                lst.append((words[wds_pt], [ bpes[bpes_pt] ]))
                # wds_pt, bpes_pt = wds_pt + 1, bpes_pt + 1
            else:
                bpes_start_pt, bpes_pt = bpes_pt, bpes_pt + 1
                while bpes_pt < len(bpes) and words[wds_pt] != ''.join(bpes[bpes_start_pt: bpes_pt + 1]):
                    bpes_pt += 1
                lst.append((words[wds_pt], bpes[bpes_start_pt: bpes_pt + 1]))
            wds_pt, bpes_pt = wds_pt + 1, bpes_pt + 1
        return lst

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = this_model(**inputs, output_hidden_states=True)
        res = (
            torch.nn.functional.normalize(
                torch.cat(res["hidden_states"][-3:-2], -1)[0], dim=0
            )
            .float()
            .cpu()
        )
        # print(res.shape)
    tokens = [ token.replace("##", "") for token in inputs.tokens() ]
    lst = align_bpes_to_words(tokens, words)
    slices = list(map(lambda x: len(x[1]), lst))
    # print(slices)
    embs = torch.split(res, slices, dim=0)
    embs = torch.cat([torch.sum(emb, dim=0, keepdim=True) for emb in embs], dim=0)
    # print(embs.shape)
    embs = [(embs[idx] / n).repeat(n, 1) for idx, n in enumerate(word2ph)]
    embs = torch.cat(embs, dim=0)
    # print(embs.shape)
    return embs.T


if __name__ == '__main__':
    # root_path = os.path.abspath(os.path.join('.', '.'))
    # config_path = os.path.join(root_path, 'Data/configs/config.json')
    # hps = utils.get_hparams_from_file(config_path)
    # language = "ZH"
    # device = "cuda"
    
    # test_text_normalize()

    # s = "日前,全国广电融媒数字主持人工作室在北京正式成立。该工作室由北京广电实战、北京灵境赛博和中国科学技术大学“合成现实”联合实验室三家单位共同发起。工作室将在广播电视主持人领域大力探索人工智能技术的应用与实践。"
    # s = "你这个idea, 不太make sense。"
    s = "突然想起西城男孩的一句歌词：“this love is unbreakable”；我感慨万千！"
    test_g2p_mix(s)

    pass




