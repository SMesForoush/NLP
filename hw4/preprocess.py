import json
import random
import re
import hazm
import pandas as pd
from transformers import AutoTokenizer

NORMAL = 'بیان عادی'
INTERESTING = 'بیان جذاب'
model_name_or_path = "HooshvareLab/bert-fa-zwnj-base"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


def read_data(path, state=None):
    data = json.load(open(path))
    if state is not None:
        data = data.get(state)
    return data


def label_prob(labels):
    normal_counter = 0
    for l in labels:
        if l == NORMAL:
            normal_counter += 1
    return normal_counter / len(labels)


def balance_train(df):
    df_int = df[df['normal_prob'] <= 0.5]
    df_norm = df[df['normal_prob'] > 0.5]
    mid_num = (len(df_norm) - len(df_int)) // 2
    df_norm_new = df_norm.sample(n=len(df_norm) - mid_num).reset_index(drop=True)
    df_int_add = df_int.sample(n=min(mid_num, len(df_int))).reset_index(drop=True)
    new_df = pd.concat([df_norm_new, df_int, df_int_add])
    new_df = new_df.sample(frac=1).reset_index(drop=True)
    return new_df


# in case of need to vote

def label_vote(labels):
    if all(x == labels[0] for x in labels):
        return labels[0]
    normal_counter = 0
    interesting_counter = 0
    for l in labels:
        if l == NORMAL:
            normal_counter += 1
            if normal_counter > 1:
                return NORMAL
        elif l == INTERESTING:
            interesting_counter += 1
            if interesting_counter > 1:
                return INTERESTING
    return random.choice([NORMAL, INTERESTING])


def cleaning(text):
    text = text.strip()
    normalizer = hazm.Normalizer()
    text = normalizer.normalize(text)

    # removing additional characters
    text = re.sub("\d", "", text)
    text = text.lower()
    text = re.sub("[a-z]", "", text)
    text = re.sub('/|\+|\||\-|؛|،', ' ', text)

    # may be needed: ((  (,),",",«,»,:,!  ))

    # removing wierd patterns
    wierd_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u'\U00010000-\U0010ffff'
                               u"\u200d"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\u3030"
                               u"\ufe0f"
                               u"\u2069"
                               u"\u2066"
                               # u"\u200c"
                               u"\u2068"
                               u"\u2067"
                               "]+", flags=re.UNICODE)

    text = wierd_pattern.sub(r'', text)

    # removing extra spaces, hashtags
    text = re.sub("\s+", " ", text)
    return text


def token_title(text):
    return tokenizer.tokenize(text)

def preprocess_df(f, balance=True):
        # read from json
    p = f'./{f}.json'
    data = read_data(p)
    data = data[f]
    df = pd.DataFrame(data)
    # remove empty news
    df = df[df['annotations'].map(lambda d: len(d)) > 0]
    df = df[df['text'] != ''].reset_index(drop=True)
    # get probs of title type
    df['normal_prob'] = df['annotations'].apply(label_prob)
    df['interesting_prob'] = 1 - df['normal_prob']
    # balancing data for training
    if balance:
        df = balance_train(df)
    # cleaning
    df['clean'] = df['text'].apply(cleaning)
    # tokenizing
    df['tokens'] = df['clean'].apply(token_title)
    return df


def preprocess():
    files = ['train', 'eval', 'test']
    for f in files:
        df = preprocess_df(f, f=="train")
        # save csv
        df.to_csv(f'{f}_clean.csv', encoding='utf-8-sig')
        print(f'{f} is done')

     
if __name__ == "__main__":
    preprocess()
