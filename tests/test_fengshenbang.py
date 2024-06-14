import os
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


fpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../bert/IDEA-CCNL/Erlangshen-UniMC-MegatronBERT-1.3B-Chinese")) # repo_id


def test_for_fill_mask():
    pipeline_ins = pipeline(
        'fill-mask',
        model=fpath,
        model_revision='v1.0.0',
    )
    print(pipeline_ins('段誉轻[MASK]折扇，摇了摇[MASK]。'))


'''
https://blog.csdn.net/Yonggie/article/details/130455903 # IDEA封神榜大语言模型二郎神系列Erlangshen-Ubert-110M-Chinese使用
'''
def test_for_text_classification():
    import argparse
    from fengshen.pipelines.multiplechoice import UniMCPipelines
    total_parser = argparse.ArgumentParser("TASK NAME")
    total_parser = UniMCPipelines.pipelines_args(total_parser)
    args = total_parser.parse_args()
    pretrained_model_path = fpath
    args.learning_rate=2e-5
    args.max_length=512
    args.max_epochs=3
    args.batchsize=8
    args.default_root_dir='./'
    model = UniMCPipelines(args, pretrained_model_path)

    train_data = []
    dev_data = []
    test_data = [
            {"texta": "放弃了途观L和荣威RX5，果断入手这部车，外观霸气又好开", 
            "textb": "", 
            "question": "下面新闻属于哪一个类别？", 
            "choice": [
                "房产", 
                "汽车", 
                "教育", 
                "科技"
                ], 
            # "answer": "汽车", 
            "label": 1, 
            "id": 7759}
        ]

    if args.train:
        model.train(train_data, dev_data)
    result = model.predict(test_data)
    for line in result[:20]:
        print(line)


def test_for_ner():
    import argparse
    from fengshen import UbertPipelines
    total_parser = argparse.ArgumentParser("TASK NAME")
    total_parser = UbertPipelines.pipelines_args(total_parser)
    args = total_parser.parse_args()
    args.pretrained_model_path = "../bert/IDEA-CCNL/Erlangshen-Ubert-330M-Chinese"

    test_data=[
        {
            "task_type": "抽取任务", 
            "subtask_type": "实体识别", 
            "text": "这也让很多业主据此认为，雅清苑是政府公务员挤对了国家的经适房政策。", 
            "choices": [ 
                {"entity_type": "小区名字"}, 
                {"entity_type": "岗位职责"}
                ],
            "id": 0}
    ]

    model = UbertPipelines(args)
    result = model.predict(test_data)
    for line in result:
        print(line)
    
    
def test_for_context():
    import argparse
    from fengshen.pipelines.multiplechoice import UniMCPipelines
    total_parser = argparse.ArgumentParser("TASK NAME")
    total_parser = UniMCPipelines.pipelines_args(total_parser)
    args = total_parser.parse_args()
    pretrained_model_path = fpath
    args.learning_rate=2e-5
    args.max_length=512
    args.max_epochs=3
    args.batchsize=8
    args.default_root_dir='./'
    model = UniMCPipelines(args, pretrained_model_path)

    train_data = []
    dev_data = []
    test_data = [
            {"texta": "北京有什么好吃的呢？周杰伦新歌怎么样？", 
            "textb": "", 
            "question": "基于文本", # C-TODO ...
            "choice": [
                "可以推出：两句话是上下文关系", 
                "不能推出：两句话是上下文关系", 
                "很难推出：两句话是上下文关系"
                ], 
            # "answer": "不能推出：外来货物入境不需要经过海关", 
            "label": 1, 
            "id": 23}
        ]

    if args.train:
        model.train(train_data, dev_data)
    result = model.predict(test_data)
    for line in result[:20]:
        print(line)


if __name__ == '__main__':
    # test_for_ner()
    
    # test_for_fill_mask()
    
    # test_for_text_classification()
    
    test_for_context()


