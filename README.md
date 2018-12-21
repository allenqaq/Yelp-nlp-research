# Yelp-nlp-research

Sentiment Analysis on Yelp’s Reviews


1. Background

    Sentiment analysis (sometimes known as opinion mining or emotion AI) refers to the use of natural language processing, text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify, and study affective states and subjective information. Sentiment analysis is widely applied to voice of the customer materials such as reviews and survey responses, online and social media, and healthcare materials for applications that range from marketing to customer service to clinical medicine.
    Sentiment analysis aims to determine the attitude of a speaker, writer, or other subject with respect to some topic or the overall contextual polarity or emotional reaction to a document, interaction, or event. The attitude may be a judgment or evaluation (see appraisal theory), affective state (the emotional state of the author or speaker), or the intended emotional communication (emotional effect intended by the author or interlocutor).
    Yelp is an American multinational corporation headquartered in San Francisco, California. It develops, hosts and markets Yelp.com and the Yelp mobile app, which publish crowd-sourced reviews about local businesses, as well as the online reservation service Yelp Reservations. The company also trains small businesses in how to respond to reviews, hosts social events for reviewers, and provides data about businesses, including health inspection scores. Our project is the sentiment analysis on the reviews of Yelp. 

2. Dataset

    The dataset is the Yelp’s reviews from the restaurant in DC. We collected 150 popular restaurants in DC and for each restaurant we record its latest 100 review. Thus, we have 150,000 data. 

Figure. 1
The technology of collecting the data is simply crawling from the Yelp website. 

Figure. 2
And the data format is JSON format or you can call it Python List + dictionary.
Each row of data has 3 attributes: reviews, name of the restaurant and rating Value.

Figure. 3

3. Sentiment Analysis with deep learning

    Natural Language Processing is a system to teach machine how to process or read human language. Before the era of deep learning, NLP is also a flourishing field. At that time, we need to do a lot of complex characteristic engineering according to the knowledge of linguistics such as phonemes, morphemes, and so on.
    In the past few years, the development of deep learning has made remarkable progress, to a certain extent, we can eliminate the dependence on linguistics. As the barriers to entry have been reduced, the application of the NLP task has also become one of the major areas of deep learning research. 

3.1 RNN Algorithms

    Here we apply RNN to NLP sentiment analysis. One of the unique features of NLP data is that it is time series data. In a sentence, the appearance of each word depends on its former word and the latter word. Because of this dependency, we use recurrent neural network to process this time series data. 
    The neural network has rings, which can persist information. Like Figure 1, it shows a simple recurrent neural network consists of an input layer, a hidden layer, and an output layer. 

Figure.4 The structre of RNN

Figure. 5 The loop of hidden layer

    The hidden layer loop allows information to be passed from one step of the network to the next showed in Figure 2. In RNN, each word in a sentence is considered a time steps. In fact, the number of time steps will be equal to the maximum sequence length (like here  …… ). The intermediate state associated with each time step is also used as a new component, known as the hidden state vector . From an abstract point of view, this vector is used to encapsulate and summarize all the information seen in the preceding time steps. Just as  represents a vector, it encapsulates all the information of a particular word. The hidden state is the function of the current word vector and the hidden state vector of the previous step. And these two sums need to be activated by the activation function as following.

    In the above formula, the U and W represents the weight matrix. If you study these two matrices carefully, you will find that one of the matrices is multiplied with our input . The other is a hidden loading vector that multiplies the output of the hidden layer in the previous time step. U is the same in all the time steps, but the matrix W is different in each input. we use the Backpropagation through time (BPTT) algorithm to update the weight. At the last moment, the hidden layer's state vector is sent to a Softmax classifier to carry out a two classification, that is to say whether the text is positive or negative. Although the RNN can theoretically establish the dependence between states with long intervals, but due to the problem of explosion or disappearance of gradients, in fact, it can only learn short-term dependence. This is the problem of long-term dependence.

3.2 LSTM Networks

    In order to solve the long-term dependence problem mentioned in the previous section, a very good solution is to introduce the gate control to control the cumulative speed of information, including selectively adding new information and selectively forgetting the accumulated information the network got before (Hochreiter and Schmidhuber 1997).
    The long and short-term memory network is a variant of recurrent neural network, which can effectively solve the problem of the gradient explosion or disappearance of recurrent neural network (Gers et al., 2000, Hochreiter and Schmidhuber, 1997). From an abstract point of view, LSTM preserves long-term dependency information in the sentence. As we have seen earlier,  is very simple in the traditional RNN network, and this simple structure does not effectively link historical information together.

Figure. 6 Architecture of LSTM

    The LSTM unit outputs  based on the input data  and the hidden layer. In these units, the expression of  is much more complex than the classic RNN network. Like Figure 3 shows that the LSTM’s complex components are divided into four parts: the input gate i, the output gate o, the forgetting gate f, and a memory controller c. The core of the LSTM model is a memory cell c encoding memory at every time step of what inputs have been observed up to this step. The behavior of the cell is controlled by three gates.
Each gate uses  and  as input and uses these inputs to calculate some of the intermediate states. Each of the intermediate states will be sent to different pipes, and the information will eventually be brought together to . For the sake of simplicity, we do not care about the specific derivation of each door. These doors can be considered different modules and have different functions. The input gate determines how much emphasis is placed on each input. The forgotten gate determines what information we will discard, and the output gate determines the final  according to the intermediate state.

3.3 Sentiments Analysis with RNN and LSTM Networks

The task of sentiment analysis is to analyze the mood of a word or sentence that is positive and negative. In our task to analysis the yelp’s restaurant review from DC, we can divide this particular task into 4 different steps.
3.3.1 Exploration the data we used
Before starting, make a preliminary exploration of the data used. In particular, we need to know how many different words in the data and How many words make up every word.

Figure. 7 The exploration of data
3.3.2 Word Embedding
    Depending on the number of different words, we can set the size of the vocabulary to a fixed value, and for words that are not in the vocabulary, replace them with the pseudo-word UNK. According to the maximum length of the sentence, we can unify the length of the sentence, the short sentence filled with 0.

Figure. 8 Word Embedding
    As mentioned earlier, we set vocabulary size to 30002. Contains the first 30000 words in the training data sorted by descending order of words, plus a dummy word UNK and padding word 0. Next, create two lookup tables, word2index and index2word, for word and number conversion. The following is based on the lookup table to convert the sentence into a sequence of numbers. We need to truncate and pad the input sequences so that they are all the same length for modeling, not enough fill 0, more cut off. The model will learn the zero values carry no information so indeed the sequences are not the same length in terms of content, but same length vectors is required to perform the computation in RNN, the maximum sentence length set to 500.
3.3.3 Define and fit our RNN with LSTM model
    Like Figure 6, We define our RNN model with Keras. The first layer is the Embedded layer that uses 128 length vectors to represent each word. The next layer is the LSTM layer with 64 memory units. For we want get the positive and negative emotion finally, so we use a Dense output layer with a single neuron and a sigmoid activation function to make 0 or 1 predictions for the two classes. Because it is a binary classification problem, log loss is used as the loss function (binary_crossentropy in Keras). The efficient Adam optimization algorithm is used. Recurrent Neural networks like LSTM generally have the problem of overfitting. Dropout can be applied between layers using the Dropout Keras layer. We can do this easily by adding new Dropout layers set dropout rate 0.2 between the Embedding and LSTM layers and the LSTM and Dense output layers.

Figure. 9 The RNN with LSTM unit Model
3.3.4 Training and Test Model
    After Training and Testing we run this model and get the output in Figure 7.

Figure. 10 The output of RNN with LSTM Model

3.4 CNN Algorithms

Convolutional Neural Network (CNN) is a kind of deep and feed-forward neural networks. It attracts many attentions nowadays. Jiuxiang Gu et al. had reported that (as shown in Figure 11) CNN was used in many field, such as image classification, object detection, pose estimation, visual saliency detection, and action recognition. For these applications, CNN is most used to extract and identify information from the pictures. For convolution, it is a mathematical operation that produces a third function with two other functions. In this research, convolution operation is a matrix and acts as image filter that extract specified information from a picture. In addition, there are two important concepts about CNN, translation invariance and weight sharing. For example, in a picture there is a cat. No matter the cat is in the left or right of the picture, it could also be identified by CNN. This is translation invariance. To achieve this, the weighted value and bias should be same in the same layer, which is weight sharing.

Figure. 11 Hierarchically structured taxonomy

Figure 12 shows how the CNN algorithm works. For an image, the filter, which is also called kernels, is the same in one
feature maps according to weight sharing. There could be several different filters, and each one will extract and identify different information from the input image. After convolutions, there would be an operation called subsampling or pooling, which combine the outputs of neuron clusters into one single neuron. For example, max pooling would use the maximum value from a 2*2 clusters into one neuron in the next layer. After repeating above two steps, full connections neural network would be used to get the final results.

Figure. 12 Processing of CNN algorithms

3.5 Sentiments Analysis with CNN

Besides the applications in image processing, CNN could also be used in natural language processing. Figure 13 is the processing of how a sentence is analyzed by CNN algorithms. For a sentence, the dimension is one. However, the dimension of an image is two. To apply the CNN algorithms, a “two dimensions” sentence is needed. To solve this problem, embedding is introduced. It will help to create the second dimension of a sentence by fixed number of columns by algorithms, such as word2vec models. With this method, CNN algorithms could be used to do sentences analysis.


Figure. 13 Processing of CNN application in NLP
For this research, we will focus on the sentiment analysis using CNN algorithms. To achieve goal, reviews from Yelp will be preprocessed. Then, the CNN model will be built. At last, data from reviews will be split into training set and testing set to train the CNN model and test the accuracy of the model.
First, nltk toolkit was used to preprocess the original reviews from Yelp. After word tokenization, some simple statistics could be done. For example, the ten most frequent words in positive than negative are: delight, gem, busboys, die, divine, wonderful, falafel, omg, whenever, and fantastic. The ten most frequent words in negative than positive are: worst, poor, mediocre, terrible, horrible, rude, charged, disappointing, dirty, and attitude. In addition, the average length of the reviews is 166.4, which could help us to choose the number of columns in embedding. Besides, words in sentences should also be replaced be index numbers of the words in a dictionary. With this, it could be calculated in the CNN algorithms.
After the preprocessing, the CNN model was built. In this research, Keras, which is a high-level neural networks API, was used to build the model. Keras runs on the top of TensorFlow, and is easy to build the convolutional neural networks. The parameters could be found in the relative coding document. Below, results will be shown with some changes in parameters.

Figure. 14 summary of default model
Table. 1 sentiments analysis by CNN results
Changed Parameters
Time (s)
train loss
train acc
validation loss
validation acc
default
42
0.1666
0.9382
0.3140
0.8623
epochs = 2 -> 6
45
0.0013
0.9999
0.5414
0.8670
batch_size = 32 -> 64
36
0.2481
0.9023
0.3222
0.8620
embedding_dims = 64 -> 32
32
0.2472
0.9004
0.3724
0.8307
filters = 64 -> 32
27
0.1957
0.9258
0.3437
0.8607
kernel_size = 3 -> 8
60
0.1259
0.9569
0.3407
0.8697


As the data shown above, we could find that increasing epochs and kernel size will increase the accuracy. However, it need more research to verify how parameters affect the accuracy. 


3.6 RNN Vs. CNN

Comparing the results from RNN and CNN, we could find that the accuracy of validation in RNN is a little higher than the accuracy in CNN model. However, the time cost in CNN algorithms is about 1/10 of the time cost in RNN algorithms (run the RNN algorithms in the same laptop of CNN algorithms, the time cost is about 240s), which means that CNN is much faster than RNN if we could tolerate a little lost in accuracy. Wenpeng Yin et al. has reported that CNN is better for tasks like sentiment classification and RNN is better for sequence modeling. However, some other researches support different ideas. The paper focused on the comparison of CNN and RNN. And the below image is the results from the paper.

Figure. 15 Best results of CNN, GRU, LSTM in NLP tasks
The results showed that RNN performed better in a range of task. And, hidden size and batch size is crucial to good performance for CNNs and RNNs.

4. Future Work

    For future work, this work could be improved in three ways. First, models used to analyze sentiment can be improved. RNN and CNN could be combined to create a better model. Second, parameter settings could be improved. From the discussion in 3.6, we could find that hidden size and batch size are very important. Besides, we could use our model to do some applications. For example, the output of model could be more detailed and we can use the results to improve the scores of restaurants in Yelp. It will help customers to find the best restaurants. 


References:

1. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.
2. Gers, F. A., Schmidhuber, J., & Cummins, F. (1999). Learning to forget: Continual prediction with LSTM.
3. Jiuxiang Gu et. al, Recent Advances in Convolutional Neural Networks, arXiv:1512.07108
4. Yoon Kim, Convolutional Neural Networks for Sentence Clasification, arXiv:1408.5882v2
5. Y. LeCun, et. al., Gradient-based learning applied to document recognition, IEEE, 1998, 2278-2324
6. Ye Zhang, Byron C. Wallace, A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification, arXiv:1510.03820v4 
7. Siwei Lai etc., Recurrent Convolutional Neural Network for Text Classification
8. Daojian Zeng, Kang Liu, Yubo Chen and Jun Zhao, Distant Supervision for Relation Extraction via Piecewise Convolutional Neural Networks
9. Xipeng qiu, Deep Learning for Natural Language Processing, CCF ADL, 2016
10. J. L. Elman. Finding structure in time, cognitive science, 1990
11. Wenpeng Yin, Katharina Kann, Mo Yu, Hinrich Schutze, Comparative Study of CNN and RNN for Natural Language Processing, arXiv:1702.01923v1 





