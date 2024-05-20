import pandas as  pd
import tensorflow as tf
import re
import streamlit as st
from sklearn.model_selection import train_test_split
import time
from transformers import BertTokenizer, TFBertForSequenceClassification


path = 'path-to-save'
# Load tokenizer
bert_tokenizer = BertTokenizer.from_pretrained(path +'/Tokenizer_Movie')
# Load model
bert_model = TFBertForSequenceClassification.from_pretrained(path +'/Model_Movie')


@st.cache_resource(show_spinner="Loading Model",max_entries=30)
def Get_sentiment(Review,Tokenizer=bert_tokenizer,Model=bert_model):
    # Convert Review to a list if it's not already a list
    # path = 'path-to-save'
    # # Load tokenizer
    # Tokenizer = BertTokenizer.from_pretrained(path +'/Tokenizer_Movie')
    # # Load model
    # Model = TFBertForSequenceClassification.from_pretrained(path +'/Model_Movie')

    if not isinstance(Review, list):
        Review = [Review]

    Input_ids, Token_type_ids, Attention_mask = Tokenizer.batch_encode_plus(Review,
                                                                             padding=True,
                                                                             truncation=True,
                                                                             max_length=128,
                                                                             return_tensors='tf').values()
    prediction = Model.predict([Input_ids, Token_type_ids, Attention_mask])

    label = {
    1: 'positive',
    0: 'Negative'
            }
    # Use argmax along the appropriate axis to get the predicted labels
    pred_labels = tf.argmax(prediction.logits, axis=1)

    # Convert the TensorFlow tensor to a NumPy array and then to a list to get the predicted sentiment labels
    pred_labels = [label[i] for i in pred_labels.numpy().tolist()]
    return pred_labels

Review =st.text_input("Input Your Sentence to know It's Sentiment:",value="movie is good but some scenes are bad")

if Review:

    seconds=time.time()
    st.write(time.ctime(seconds))

    st.write(Get_sentiment(Review))

    print(Get_sentiment(Review))

    seconds1=time.time()
    st.write(time.ctime(seconds1))
    

        
    

    