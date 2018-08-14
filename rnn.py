from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN
import os
import re
# 1.delete the tag of 'html'
def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('',text)
#2. achieve the diretory of the file
def read_files(filetype):
    path='/Users/writ/Downloads/aclImdb/'
    file_list=[]
    positive_path = path + filetype +'/pos/'
    for f in os.listdir(positive_path):
        file_list+=[positive_path+f]

    negative_path = path + filetype +'/neg/'
    for f in os.listdir(negative_path):
        file_list+=[negative_path+f]

    # print('read',filetype,'files:',len(file_list))

    all_labels=([1]*12500 + [0] *12500)

    all_texts = []
    for fi in file_list:
        with open(fi,encoding='utf8') as file_input:
            all_texts += [rm_tags(' '.join(file_input.readlines()))]
    return all_labels,all_texts

#3. acheieve the dir of data
y_train,train_text=read_files('train')
y_test,test_text=read_files('test')

#4. establish token
token=Tokenizer(num_words=3800)
token.fit_on_texts(train_text)
# print(token.word_index)
# print(token.document_count)

#5. transfer to the number list
x_train_seq = token.texts_to_sequences(train_text)
x_test_seq = token.texts_to_sequences(test_text)
# print(train_text[0])
# print(x_train_seq[0])

#6. alignment length
x_train=sequence.pad_sequences(x_train_seq,maxlen=380)
x_test=sequence.pad_sequences(x_test_seq,maxlen=380)
# print('before pad_sequences length',len(x_train_seq[0]))
# print(x_train_seq[0])

# 7.set up the model
model = Sequential()

# 8. add the embedding layer to the model
model.add(Embedding(output_dim=32,input_dim=3800,input_length=380))
model.add(Dropout(0.35))   #avoid over-fitting

#9. add the flatten layer to the model
model.add(SimpleRNN(units=16))

#hidden layer
model.add(Dense(units=256,activation='relu'))
model.add(Dropout(0.35))

#10.add the output layer to the model
model.add(Dense(units=1,activation='sigmoid'))

model.summary()

# 11.define the method of the training
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#12.start training
train_history=model.fit(x_train,y_train,batch_size=100,epochs=10,verbose=2,validation_split=0.2)

# 13.evaluate the accuracy of the model
scores=model.evaluate(x_test,y_test,verbose=1)
print(scores[1])

#14. predict
predict=model.predict_classes(x_test)
predict[:20]
predict_classes=predict.reshape(-1)
predict_classes[:20]

# 15. show the result of model
SentimentDict={1:'正面的',0:'负面的'}
def display_test_Sentiment(i):
    print(test_text[i])
    print('label真实值:',SentimentDict[y_test[i]],'预测结果：',SentimentDict[predict_classes[i]])
display_test_Sentiment(2)
