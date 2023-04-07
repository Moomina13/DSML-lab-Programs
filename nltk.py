#!/usr/bin/env python
# coding: utf-8

# In[6]:


import nltk 
from nltk.corpus import state_union 
from nltk.tokenize import PunktSentenceTokenizer
nltk.download('state_union')


# In[7]:


train_text = state_union.raw("2005-GWBush.txt") 
sample_text = state_union.raw("2006-GWBush.txt")


# In[8]:


custom_sent_tokenizer = PunktSentenceTokenizer(train_text)


# In[9]:


tokenized = custom_sent_tokenizer.tokenize(sample_text)


# In[10]:


def process_content():
    try:
        for i in tokenized[:5]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)

    except Exception as e:
        print(str(e))


process_content()


# In[ ]:




