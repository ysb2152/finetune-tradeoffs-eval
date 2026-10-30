"""약신호 결합 분류기"""
import json
import argparse
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression

def load_conf_rows(regime):

    with open("runs/eval78/"+regime+"_search_conf.json","r",encoding="utf-8") as f:

        conf=json.load(f)

    return conf["rows"]

def split_rows(rows,split_path):

    with open(split_path, "r", encoding="utf-8") as f:
        splits=json.load(f)
        tune=set(splits["tune"]["papers"])
        test=set(splits["test"]["papers"])

    tune_list=[]
    test_list=[]

    for row in rows:
        if row["paper_id"] in tune:
            tune_list.append(row)
        elif row["paper_id"] in test:
            test_list.append(row)

    return (tune_list,test_list)

def features(rows):

    features_list=[]
    for row in rows:
        features_list.append([row["top_score"],row["gen_confidence"]])

    features_list=np.array(features_list)

    return features_list

def labels(rows):

    label_list=[]

    for row in rows:

        if row["matched_type"]=="unanswerable":
            label_list.append(1)
        else:
            label_list.append(0)

    label_list=np.array(label_list)

    return label_list

def fit_gate(tune_rows):

    X= features(tune_rows)
    Y=labels(tune_rows)
    mean=X.mean(axis=0)
    std=X.std(axis=0)
    Xs=(X-mean)/std
    model=LogisticRegression().fit(Xs,Y)

    return (model,mean,std)

def gate_scores(model,mean,std,rows):

    X=features(rows)
    Xs=(X-mean)/std
    rows=model.predict_proba(Xs)

    return rows[:,1]

def pick_threshold(scores,y,budget):

    ans=scores[y==0]
    threshold=np.quantile(ans,1-budget)

    return threshold

def evaluate(rows,scores,threshold):

    f1_before=[]
    f1_after=[]
    return_dict=dict()
    answerable=0
    False_answerable=0
    catch_unanswerable=0
    n_gated=0
    all_unanswerable=0
    all_answerable=0

    for row,score in zip(rows,scores):
        f1_before.append(row["f1"])
        if score>=threshold:
            n_gated+=1
            if row["matched_type"]=="unanswerable":
                f1_after.append(1)
                catch_unanswerable+=1
            else:
                f1_after.append(0)
                False_answerable+=1
        else:
            f1_after.append(row["f1"])

        if row["matched_type"]=="unanswerable":
            all_unanswerable+=1
        else:
            all_answerable+=1

    if f1_before:
        return_dict["f1_before"]=sum(f1_before)/len(f1_before)
    else:
        return_dict["f1_before"]=0

    if f1_after:
        return_dict["f1_after"]=sum(f1_after)/len(f1_after)
    else:
        return_dict["f1_after"]=0

    return_dict["over_abstention"]= False_answerable/all_answerable
    return_dict["catch"]= catch_unanswerable/all_unanswerable
    return_dict["n_gated"]= n_gated

    return return_dict

def run(regime,split_path,budget):

    conf_rows=load_conf_rows(regime)
    tune_list,test_list= split_rows(conf_rows,split_path)
    model,mean,std=fit_gate(tune_list)


    score=gate_scores(model,mean,std,tune_list)
    test_score=gate_scores(model,mean,std,test_list)
    label_list=labels(tune_list)
    threshold=pick_threshold(score,label_list,budget)

    return_dict=evaluate(test_list,test_score,threshold)
    return_dict["regime"]=regime
    return_dict["threshold"]=threshold

    return return_dict

def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", type=str, choices=["base","full","lora","qlora"],required=True)
    parser.add_argument("split_path", type=Path)
    parser.add_argument("--budget",type=float,default=0.069)
    parser.add_argument("--out", type=Path, help="결과 JSON 저장 경로",required=True)

    args = parser.parse_args()

    return_dict = run(args.regime,args.split_path, args.budget)
    text = json.dumps(return_dict, ensure_ascii=False, indent=3)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    (args.out).write_text(text, encoding="utf-8")

    print(text)


if __name__ == "__main__":
    main()
