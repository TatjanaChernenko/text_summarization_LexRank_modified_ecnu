# -*- coding: utf-8 -*-
import os
from bs4 import BeautifulSoup as bs

def get_topic(topics_file):
    all_topics = []
    with open(topics_file, "r") as f:
        topics = f.readlines()
    for i in range(len(topics)):
        if topics[i].startswith("---"):
            if topics[i+1] != "\n":
                topic = topics[i+1]
                all_topics.append(topic)
            else:
                topic = topics[i+2]
                all_topics.append(topic)
    for r in range(len(all_topics)):
        all_topics[r] = all_topics[r].split(".")        
        all_topics[r] = all_topics[r][0]
    return all_topics

def get_files_for_topic(path_to_testdata):
    files = os.listdir(path_to_testdata)
    return files

def get_texts(path_to_testdata,files, topic,output_path):
    all_sentences_for_topic = []
    for el in files:
        if el[1:-1] == topic:
            path_to_test_files_for_topic = el
    path = path_to_testdata+"/"+path_to_test_files_for_topic
    docs = os.listdir(path)
    for el in docs:
        with open(path+"/"+el, "r") as f:
            data = f.read()
            b = bs(data)
            text = b.find_all("text")
            text = bs(str(text))
            text = text.text
            all_sentences_for_topic.append(text)
           
    with open(output_path+"/"+topic+"_all.txt", "w") as f:
        for el in all_sentences_for_topic:
            el = el.replace("[","") 
            el = el.replace("]","")
            for e in el:
                f.write(e) 
    return path_to_test_files_for_topic
        

if __name__ == "__main__":
    #output_path = "/home/tatiana/Desktop/4_Semester/A_Textzusam/Probe/output_1"
    output_path = r"../lib/test_data"
    #path_to_testdata = "/home/tatiana/Desktop/4_Semester/A_Textzusam/Probe/DUC2003/duc2003_testdata/task2/docs"
    path_to_testdata = "../lib/DUC2003/task2/docs"
    #topics_file = "/home/tatiana/Desktop/4_Semester/A_Textzusam/Probe/DUC2003/duc2003_testdata/task2/topics"
    topics_file = r"../lib/DUC2003/task2/topics"
    all_topics = get_topic(topics_file)
    files = get_files_for_topic(path_to_testdata)
    for topic in all_topics:
        path_to_test_files_for_topic = get_texts(path_to_testdata,files, topic,output_path)
    
