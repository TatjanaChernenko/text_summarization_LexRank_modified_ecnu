# -*- coding: utf-8 -*-
from pyrouge import Rouge155
from pprint import pprint
import json

ref_texts = []
def get_sum(sum_file):
    summary_text = []
    with open(sum_file, "r") as f:
        data = f.readlines()
        print(data)
        a = data[0].split(".")
        for el in a:
            if el != " ":
                x = el.strip()+"."
                summary_text.append(x)
        print("\nSum: ", summary_text, "--Sum\n")
    return summary_text
        
def get_ref_sum(ref_files):
    ref_text = {}
    for el in ref_files:
        print("el:", el)
        first = el.split("/")[-1].split(".")[3]
        second = el.split("/")[-1].split(".")[4]
        print(first,second) 
        with open(el, "r") as f:
            data = f.readlines()
        su = []
        for e in data:
            if e != " ":
                x = e.strip()
                print("x: ", x)
                su.append(x)
        print("\nSU: ", su, "---SU\n")
        ref_text[first+second] = su
        print("\nREF text; ", ref_text)
    return ref_text 

def get_rouge(summary_text, ref_texts):
    rouge1 = Rouge155(n_words=100,average="sentence")
    rouge2 = Rouge155(average="sentence") #!!! DAS MACHEN für 4 Texte als 2.Variante
    #rouge = Rouge155(n_words=100,average="token")
    #rouge = Rouge155(average="token")
    score1=rouge1.score_summary(summary_text, ref_texts)
    score2=rouge2.score_summary(summary_text, ref_texts)
    pprint(score1)
    return score1, score2
    
def write_eval(score1,score2,path,outputfile,sum_text):
    with open(path+"/"+outputfile+"_100words", "w") as f:
        #f.write("Summarization with LexRank + EN7 with parameters n_words=100,average=\"sentence\".\nROUGE scores for the text "+ sum_text+ ":\n")    
        f.write("Summarization with LexRank with parameters n_words=100,average=\"sentence\".\nROUGE scores for the text "+ sum_text+ ":\n")    
        a = json.dumps(score1,indent=4, separators=(',', ': '))
        #print(a)
        f.write(a+"\n")
    with open(path+"/"+outputfile, "w") as f:
        f.write("Summarization with LexRank + EN7 average=\"sentence\".\nROUGE scores for the text "+ sum_text+ ":\n")    
        #f.write("Summarization with LexRank with parameters n_words=100,average=\"sentence\".\nROUGE scores for the text "+ sum_text+ ":\n")    
        a = json.dumps(score2,indent=4, separators=(',', ': '))
        #print(a)
        f.write(a+"\n")
    print("DONE")

if __name__ == "__main__":    

    # FOR 30020 with standard LexRANK
    output_path = r".\eval"
    ref_1 =r".\..\lib\DUC2003\peer7.2\D30020.M.100.T.E.html"
    ref_2 =r".\..\lib\DUC2003\peer7.2\D30020.M.100.T.F.html"
    ref_3 =r".\..\lib\DUC2003\peer7.2\D30020.M.100.T.G.html"
    ref_4 =r".\..\lib\DUC2003\peer7.2\D30020.M.100.T.H.html"
    ref_files = []
    ref_files.append(ref_1)
    ref_files.append(ref_2)
    ref_files.append(ref_3)
    ref_files.append(ref_4)
    #sum_text = "/home/tatiana/Desktop/4_Semester/A_Textzusam/Probe/summaries_1/sum_30020_all.txt"
    sum_text = r".\..\output\aummaries_standard\sum_30020_all.txt"
    sum_t = get_sum(sum_text)
    ref = get_ref_sum(ref_files)
    score1,score2 = get_rouge(sum_t, ref)
    write_eval(score1,score2, output_path, "eval_sum_30020_standard",sum_text)

    #output_path = "/home/tatiana/Desktop/4_Semester/A_Textzusam/Probe/summaries_3/"
    # FOR 30020:
    ref_1 =r".\..\lib\DUC2003\peer7.2\D30020.M.100.T.E.html"
    ref_2 =r".\..\lib\DUC2003\peer7.2\D30020.M.100.T.F.html"
    ref_3 =r".\..\lib\DUC2003\peer7.2\D30020.M.100.T.G.html"
    ref_4 =r".\..\lib\DUC2003\peer7.2\D30020.M.100.T.H.html"
    ref_files = []
    ref_files.append(ref_1)
    ref_files.append(ref_2)
    ref_files.append(ref_3)
    ref_files.append(ref_4)
    sum_text = r".\..\output\aummaries_improved\sum_30020_all.txt"
    sum_t = get_sum(sum_text)
    ref = get_ref_sum(ref_files)
    score1,score2 = get_rouge(sum_t, ref)
    write_eval(score1, score2, output_path, "eval_sum_30020",sum_text)

    #output_path = "/home/tatiana/Desktop/4_Semester/A_Textzusam/Probe/summaries_3/"
    # FOR 30034:
    ref_1 =r".\..\lib\DUC2003\peer7.2\D30034.M.100.T.A.html"
    ref_2 =r".\..\lib\DUC2003\peer7.2\D30034.M.100.T.B.html"
    ref_3 =r".\..\lib\DUC2003\peer7.2\D30034.M.100.T.I.html"
    ref_4 =r".\..\lib\DUC2003\peer7.2\D30034.M.100.T.J.html"
    ref_files = []
    ref_files.append(ref_1)
    ref_files.append(ref_2)
    ref_files.append(ref_3)
    ref_files.append(ref_4)
    sum_text = r".\..\output\aummaries_improved\sum_30034_all.txt"
    sum_t = get_sum(sum_text)
    ref = get_ref_sum(ref_files)
    score1,score2 = get_rouge(sum_t, ref)
    write_eval(score1, score2, output_path, "eval_sum_30034",sum_text)

    #FOR 31010
    ref_1 =r".\..\lib\DUC2003\peer7.2\D31010.M.100.T.C.html"
    ref_2 =r".\..\lib\DUC2003\peer7.2\D31010.M.100.T.D.html"
    ref_3 =r".\..\lib\DUC2003\peer7.2\D31010.M.100.T.E.html"
    ref_4 =r".\..\lib\DUC2003\peer7.2\D31010.M.100.T.F.html"
    ref_files = []
    ref_files.append(ref_1)
    ref_files.append(ref_2)
    ref_files.append(ref_3)
    ref_files.append(ref_4)
    sum_text = r".\..\output\aummaries_improved\sum_31010_all.txt"
    sum_t = get_sum(sum_text)
    ref = get_ref_sum(ref_files)
    score1,score2 = get_rouge(sum_t, ref)
    write_eval(score1,score2, output_path, "eval_sum_31010",sum_text)

    # FOR 30034 with standard LexRANK
    #output_path = "/home/tatiana/Desktop/4_Semester/A_Textzusam/Probe/summaries_1/"
    ref_1 =r".\..\lib\DUC2003\peer7.2\D30034.M.100.T.A.html"
    ref_2 =r".\..\lib\DUC2003\peer7.2\D30034.M.100.T.B.html"
    ref_3 =r".\..\lib\DUC2003\peer7.2\D30034.M.100.T.I.html"
    ref_4 =r".\..\lib\DUC2003\peer7.2\D30034.M.100.T.J.html"
    ref_files = []
    ref_files.append(ref_1)
    ref_files.append(ref_2)
    ref_files.append(ref_3)
    ref_files.append(ref_4)
    sum_text = "r.\..\output\summaries_standard\sum_30034_all.txt"
    sum_t = get_sum(sum_text)
    ref = get_ref_sum(ref_files)
    score1,score2 = get_rouge(sum_t, ref)
    write_eval(score1,score2, output_path, "eval_sum_30034_standard",sum_text)

    # FOR 31010 with standard LexRANK
    #utput_path = "/home/tatiana/Desktop/4_Semester/A_Textzusam/Probe/summaries_1/"
    ref_1 =r".\..\lib\DUC2003\peer7.2\D31010.M.100.T.C.html"
    ref_2 =r".\..\lib\DUC2003\peer7.2\D31010.M.100.T.D.html"
    ref_3 =r".\..\lib\DUC2003\peer7.2\D31010.M.100.T.E.html"
    ref_4 =r".\..\lib\DUC2003\peer7.2\D31010.M.100.T.F.html"
    ref_files = []
    ref_files.append(ref_1)
    ref_files.append(ref_2)
    ref_files.append(ref_3)
    ref_files.append(ref_4)
    sum_text = r".\..\output\aummaries_standard\sum_31010_all.txt"
    sum_t = get_sum(sum_text)
    ref = get_ref_sum(ref_files)
    score1,score2 = get_rouge(sum_t, ref)
    write_eval(score1,score2, output_path, "eval_sum_31010_standard",sum_text)


