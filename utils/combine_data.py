import pandas as pd
import os
import re
import unicodedata
import nltk
from nltk.corpus import stopwords

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def clean_stopwords_shortwords(w):
    stopwords_list=stopwords.words('english')
    words = w.split() 
    clean_words = [word for word in words if (word not in stopwords_list) and len(word) > 2]
    return " ".join(clean_words) 

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿'$0-9]+", " ", w)
    w=re.sub(r'@\w+', '',w)
    return w

data_dir = "./data/dailydialog/"
act_file = os.path.join(data_dir + "dialogues_act.txt")
emotion_file = os.path.join(data_dir + "dialogues_emotion.txt")
text_file = os.path.join(data_dir + "dialogues_text.txt")
topic_file = os.path.join(data_dir + "dialogues_topic.txt")

txt_content = open(text_file, 'r', encoding="utf-8")
emotion_content = open(emotion_file, 'r', encoding="utf-8")
act_content = open(act_file, 'r', encoding="utf-8")
topic_content = open(topic_file, 'r', encoding="utf-8")
data = []
count = 1
for line1, line2, line3, line4 in zip(txt_content, emotion_content, act_content, topic_content):
    text = line1.strip().split("__eou__")[:-1]
    emotion = line2.strip().split(" ")
    act = line3.strip().split(" ")
    topic = line4.strip()
    for i in range(len(text)):
        record = {
            "conversation_id" : count,
            "topic": topic,
            "message" : preprocess_sentence(text[i]),
            "emotion" : emotion[i],
            "act" : act[i],
        }
        data.append(record)
    count = count + 1
df = pd.DataFrame(data)
df.to_csv(data_dir + "all_data.csv", index=False)



