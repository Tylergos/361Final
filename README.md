# Author Identification from Audio Data

This repository contains the code and resources for a project that aims to identify authors based on sampled audio data. The project explores the effectiveness of various models, including an LSTM-based model, a lexicon-type model, and a similarity-based model, to predict the author given a text or audio sample.

The related paper can be found in [NLP_Final_Project.pdf](https://github.com/Tylergos/361Final/blob/063dd94636c6c82aa3c3e11149f12ca746b13110/NLP_Final_Project.pdf)

## Motivation

Author identification is an important problem in the era of online misinformation and AI-generated voices. Identifying the true author of a statement can help prevent the spread of fake information and address potential security issues.

## Dataset

The dataset used in this project is taken from [Jain et al., 2019](https://www.kaggle.com/datasets/vjcalling/speaker-recognition-audio-dataset). It contains audio data of 50 different speakers, each with approximately one hour of audio. The audio samples come from public domain audiobook recordings and educational videos. The speakers have a variety of accents, but all speak English. Due to data corruption issues, the final dataset used consists of 2511 samples.

## Preprocessing

1. Audio data was split into 80% training and 20% validation sets.
2. Audio data was compressed from 44kHz to 11kHz.
3. Text data was extracted from audio using Google's Speech Recognition API.
4. Text data was tokenized using the NLTK tokenizer.

## Models

### Lexicon Model

This model is based on the unique tokens associated with each author. The prediction is made based on the author with the highest number of tokens present in the test document.

### Similarity Model

This model uses a binarized author-term matrix and cosine similarity to make predictions. The predicted author is the one with the highest cosine similarity value to the test document vector.

### LSTM Model

This model uses an LSTM-based architecture trained on the audio data inputs. It consists of 5 layers: input, 1D-CNN, max pooling, LSTM, and dense output layers.

## Results

The performance of each model was evaluated using recall, precision, F1-score, and overall accuracy. Generally, the LSTM model outperformed the other models, especially when dealing with shorter data samples.

| Model                | Recall | Precision | F1-Score | Overall Accuracy |
|----------------------|--------|-----------|----------|------------------|
| Lexicon ~6s          | 0.366  | 0.495     | 0.387    | 0.423            |
| Lexicon ~30s         | 0.717  | 0.781     | 0.729    | 0.768            |
| Lexicon ~60s         | 0.825  | 0.862     | 0.829    | 0.864            |
| Similarity ~6s       | 0.342  | 0.417     | 0.351    | 0.379            |
| Similarity ~30s      | 0.739  | 0.780     | 0.743    | 0.778            |
| Similarity ~60s      | 0.861  | 0.880     | 0.864    | 0.895            |
| LSTM ~6s             | 0.905  | 0.909     | 0.905    | 0.905            |
| LSTM ~30s            | 0.957  | 0.957     | 0.955    | 0.958            |
| LSTM ~60s            | 0.973  | 0.977     | 0.974    | 0.967            |

## Conclusion

The LSTM model performed the best among the three models, showcasing its ability to handle varying lengths of audio data while maintaining high accuracy. The lexicon and similarity models also demonstrated better performance as the data length increased, but their performance was significantly lower than the LSTM model, particularly for shorter data samples.

In conclusion, the LSTM model's superior performance can be attributed to its ability to process a large number of data points even in short audio samples and its overall greater predictive power compared to the text-based models. The results of this project suggest that LSTM-based models are a promising approach for author identification from audio data.
