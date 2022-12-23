# Date-Translation
In this project, we build a LSTM-based seq2seq model with attention mechanism to translate human-readable date to machine-readable date.

The training, validation and test data are generated and preprocessed before being fed to the model. Data preprocessing including data cleaning, tokenization, constructing vocabulary and generating sequences from texts, etc., is performed using a customized preprocessor class written in the 'preprocessing.py' file.

The model is composed of an encoder to take the input sequence and encode it into a representation vector (hidden state) and a decoder to generate the output sequence based on the encoded information from the input sequence. The attention layer further allows the decoder to utilize the information of all the hidden states of the encoder at every step of generating a new element during the decoding stage. In that way, the model learns to pay attention to different parts of the input sequence when generating different parts of the output sequence, which results in better performance. Specifically, one layer or multiple layers of bidirectional LSTM cells are used for the encoder and the same number of layers of unidirectional LSTM cells are used for the decoder.

The human-readable dates are in miscellaneous formats containing numbers, English letters and punctuations, while the machine-readable dates are all in the format of 'YYYY-MM-DD'. We define a predicted sequence as correct only if it exactly match the ground truth element-by-element.

The loss and accuracy of the model on the training and validation datasets as a function of training epoch are shown in the following figures:

<img src="https://github.com/JiayuX/Date-Translation/blob/main/1.png" width="1000"/>

The trained model achieved 99.60% accuracy on the test dataset. Due the limited computational resources on my laptop, I simply manually tuned the hyper-parameters, trained the model for several times and selected the best one. A better model could be gotten with more training data and more sysmatic hyper-parameter tuning given enough computational resources and time.

Here is a glimpse of the translation by the model on some dates that I made up:

<img src="https://raw.githubusercontent.com/JiayuX/Date-Translation/main/2.png" width="400"/>
