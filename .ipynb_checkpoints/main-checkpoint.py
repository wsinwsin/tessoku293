import argparse
from transformers import AutoTokenizer, BertForSequenceClassification
import pandas as pd
import numpy as np

def search_print(text, kk = 1):# キーワードに沿った鉄則を出力する
    #text:キーワード:str
    #kk: top kk まで出力可能
    t = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    result = my_model(**t)
    pt_predictions = nn.functional.softmax(result.logits, dim=-1)

    sorted, idx = torch.sort(pt_predictions, dim = -1, descending = True)
    for i in idx[0][:kk]+1:
        print(f"鉄則{i.item()},{te_list[i.item()]}")

def main(args):

    text = args.keyword #キーワード

    kk = args.k#Top10などの表示する数
    te_path = args.tessoku293_path#鉄則のnpyファイルのパス#tessoku293_list.npy
    
    te_list = np.load(te_path, allow_pickle='TRUE')
    #te_list = {1: 'プロジェクトの行く手を照らせ', 2: '常に目的を意識せよ', 3: 'テストは幅広い顧客へのサービス業だと心得よ',...}
    
    #文字をトークン化するモデルのロード
    
    model_name = "cl-tohoku/bert-base-japanese-whole-word-masking" #事前学習したモデルの名前
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    #モデルのロード
    model_path = args.model_path#FineTuning後のチェックポイントのパスを入れる。checkpoint-3500
    my_model = BertForSequenceClassification.from_pretrained(model_path)
    
    search_print(text, te_list, my_model, tokenizer, kk)
    
if __name__ == "__main__":
    #コマンドライン引数を変数に代入
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword",default=None,type=str,required=True)
    parser.add_argument("--model_path",type=str,required=True)
    parser.add_argument("--k",type=int, default=10)
    parser.add_argument("--tessoku293_path",type=str,required=True)
    args = parser.parse_args()
    main(args)