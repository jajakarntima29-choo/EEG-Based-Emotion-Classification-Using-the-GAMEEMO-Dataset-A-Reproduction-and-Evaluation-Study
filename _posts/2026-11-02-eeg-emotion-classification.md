---
layout: post
title: ""
date: 2026-11-02
permalink: /2026/11/02/eeg-emotion-classification/
---

<style>
:root{--soft-blue:#DDEFFD;--sky-blue:#B8DDF5;--soft-pink:#FADCEA;--navy:#1F3A56;--teal:#1F7A9A;--muted:#5A6B82;--bg:#F6FAFE;}
body{background:var(--bg);color:var(--navy);line-height:1.75;}
h1,h2,h3,h4{color:var(--teal);line-height:1.3;}
h2{border-bottom:3px solid var(--soft-pink);padding-bottom:.25em;margin-top:2.2em;}
h3{border-left:5px solid var(--sky-blue);padding-left:.6em;margin-top:1.8em;}
a{color:var(--teal);}
a:hover{background:var(--soft-pink);}
strong{background:linear-gradient(transparent 62%,var(--soft-pink) 62%);}
img{max-width:100%;height:auto;border-radius:12px;display:block;margin:1.2em auto;box-shadow:0 2px 12px rgba(31,58,86,.10);}
p>em:only-child{display:block;text-align:center;color:var(--muted);font-size:.92em;margin-top:-.4em;}
table{border-collapse:collapse;width:100%;margin:1.2em 0;border-radius:10px;overflow:hidden;}
th{background:var(--soft-blue);color:var(--navy);}
th,td{border:1px solid var(--sky-blue);padding:8px 12px;}
tr:nth-child(even) td{background:var(--bg);}
blockquote{border-left:4px solid var(--soft-pink);background:var(--soft-blue);margin:1.2em 0;padding:.6em 1em;border-radius:0 10px 10px 0;}
code{background:var(--soft-blue);padding:2px 6px;border-radius:6px;}
pre{background:var(--soft-blue);border:1px solid var(--sky-blue);border-radius:10px;padding:12px;overflow-x:auto;}
hr{border:none;border-top:2px solid var(--soft-pink);}
</style>


<img src="{{ '/assets/images/cover-eeg.png.png' | relative_url }}"
     alt="EEG-Based Emotion Classification Project Cover"
     style="width:100%; height:auto; border-radius:12px;">

<p align="center"><span style="color: #E88BB5;"><strong>Project by JJa Karntima Choojangwhang</strong></span></p>

## 1. Introduction

Have you ever wondered whether we can recognize someone's emotions without actually asking them how they feel?

Emotions are a big part of how we communicate with the world. We express them through our voices, facial expressions, and body language. But what happens when someone has difficulty expressing or communicating their feelings?

This question made me interested in exploring brain signals as another possible way to understand emotional states.

For this project, I worked with Electroencephalography (EEG), a technique that records electrical activity produced by the brain. The idea was to investigate whether these signals contain patterns that machine learning models can learn and use to classify different emotions.

I used the GAMEEMO dataset to build an EEG-based emotion classification pipeline, starting from exploring the raw signals, processing the data, extracting features, and finally training machine learning models.

Throughout this project, I did not just want to achieve a high classification score. I also wanted to understand how the models behave, why their performance changes under different evaluation strategies, and whether they can recognize emotional patterns from people they have never seen before.

And, as I would later discover, getting a good score is only one part of the challenge!

## 2. Problem Statement

At first, emotion classification might sound like a straightforward task: collect data, train a model, and let it predict an emotion.

But EEG signals are not that simple.

Brain signals are complex time-series data, and their patterns can vary across time, channels, and individuals. Two people experiencing the same emotional condition may still produce different EEG patterns.

This creates an important challenge: a machine learning model might learn to recognize patterns from the participants included in its training data, but struggle when it encounters someone new.

For example, imagine training a model using EEG recordings from one group of people. If we then test it using different recordings from those same people, the results might look promising. But would the model still perform well if we gave it EEG signals from people it had never encountered?

That became one of the main questions I wanted to investigate.

Therefore, this project focuses not only on classifying emotions but also on comparing two evaluation approaches: window-wise evaluation and subject-wise evaluation. The goal is to understand whether the model is learning emotional patterns or simply becoming familiar with the characteristics of the participants in its training data.

## 3. Objectives

With this challenge in mind, I established several objectives for the project.

First, I wanted to explore EEG signals from the GAMEEMO dataset and understand what the data looked like before building any models.

Next, I aimed to develop a complete processing pipeline, from signal segmentation to feature extraction. I was particularly interested in combining different types of features to represent EEG signals more comprehensively.

After that, I wanted to compare three machine learning models: Support Vector Machine (SVM), k-Nearest Neighbors (kNN), and Multilayer Perceptron Neural Network (MLPNN).

Finally, I wanted to investigate how different evaluation strategies affect classification performance, especially when the model is tested on participants who were not included in the training process.

In short, the project was about more than finding the model with the highest score. It was about understanding how well that model could handle data from different people.

## 4. Dataset Overview: GAMEEMO

### 4.1 Getting to Know the Dataset

Before building a model, I needed EEG data to work with. For this project, I chose the GAMEEMO dataset, which was designed for emotion recognition using EEG signals.

The dataset contains recordings collected while participants played different computer games intended to evoke different emotional states.

There are four emotional categories:

| Game | Emotional category |
|---|---|
| G1 | Boring |
| G2 | Calm |
| G3 | Horror |
| G4 | Funny |

The original dataset contains 28 participants, with EEG recordings collected from 14 channels at a sampling frequency of 128 Hz.

What I found interesting about this dataset was that it allowed me to work with several emotional conditions while also exploring how EEG signals differ between participants.

### 4.2 Choosing the Participants

Although the original dataset contains 28 participants, I decided to use 18 participants for my experiments:

- S06–S14
- S20–S28

But why these participants?

Before making the selection, I performed preliminary Exploratory Data Analysis (EDA) to inspect the EEG recordings. While examining the signals, I noticed that some participants showed unusual peaks or patterns compared with other recordings.

This made me take a closer look at the data rather than immediately using every participant.

It is important to mention that an unusual peak does not automatically mean that the signal is incorrect or corrupted. It was simply an observation that encouraged me to investigate the recordings further and consider signal characteristics when selecting the participants.

This was one of my first reminders that working with real-world physiological data requires more than just loading a dataset and running a model. Understanding the data comes first.

## 5. Exploratory Data Analysis (EDA)

Once I had the dataset, the next step was to explore it.

I used EDA to visualize EEG signals from different channels and participants. I wanted to understand how the signals behaved, how much they varied, and whether there were any patterns that deserved further investigation.

During this process, I noticed that EEG signals were not identical across participants. Some recordings showed more noticeable peaks in certain frequency ranges, while others had different signal patterns.

These observations helped me understand the variability of EEG data and provided useful information for the participant selection process.

One thing I learned from this stage is that looking at the data visually can reveal things that are not immediately obvious from a table of numbers.

It also helped me become more familiar with the dataset before moving on to the more technical parts of the project.

![Unusual EEG spectral peaks]({{ '/assets/images/psd-abnormal.png' | relative_url }})

*Figure 1. EEG Power Spectral Density showing unusual spectral peaks.*

![Relatively normal EEG spectral pattern]({{ '/assets/images/psd-normal.png' | relative_url }})

*Figure 2. EEG Power Spectral Density showing relatively consistent spectral patterns.*

## 6. Data Preprocessing

After exploring the dataset, it was time to prepare the signals for feature extraction.

The original EEG recordings were stored as CSV files. I first selected the 14 EEG channels needed for the experiment and removed rows containing missing values using `dropna()`.

Then came an important step: dividing the continuous signals into smaller segments.

### 6.1 Signal Segmentation

Instead of processing an entire recording as one large signal, I divided it into smaller windows, each containing 4 seconds of EEG data.

The sampling frequency was 128 Hz, so each window contained:

\[
128 \times 4 = 512 \text{ samples}
\]

In this implementation, I used non-overlapping windows. This means that once one 4-second segment was processed, the next segment started immediately after it ended.

I chose this approach to avoid reusing the same samples across consecutive windows and to reduce the amount of data that needed to be processed.

After segmentation, each window was ready for the next step: feature extraction.

## 7. Feature Extraction

This was one of the most interesting parts of the project because raw EEG signals are not directly what traditional machine learning models work with.

I needed to transform each signal window into numerical features that could describe its characteristics.

Following the original GAMEEMO study, I extracted **10 features** from each EEG window. Rather than relying on only one type of measurement, these features capture different aspects of the signal, and they fall into two groups:

- **Time-domain and complexity features**, computed directly from the raw signal of each channel: Hjorth Parameters, Zero-crossings, Detrended Fluctuation Analysis (DFA), Sample Entropy (SampEn), and Multiscale Entropy (MSE).
- **Wavelet-based features**, computed from the Discrete Wavelet Transform (DWT) coefficients: Standard Deviation, Variance, Average Wavelet Energy, Shannon Entropy, and Logarithmic Energy Entropy.

Each of the 14 channels was processed separately, and the extracted features were later combined into one feature vector.

### 7.1 Time-Domain and Complexity Features

These five features are calculated directly from the raw signal of each channel in a 4-second window.

**1) Hjorth Parameters**

Hjorth parameters are commonly used to describe EEG signals. There are three parameters:

- **Activity:** Describes the variance or overall power of the signal.
- **Mobility:** Represents the rate of change of the signal and is related to its frequency characteristics.
- **Complexity:** Describes how the frequency characteristics of the signal change compared with a simpler signal.

Together, these three parameters provide a compact way to describe the behavior of an EEG signal.

**2) Zero-Crossings**

Zero-crossings counts how many times a signal crosses the zero-amplitude level. Although it is a simple measurement, it provides additional information about how frequently the signal changes around zero.

**3) Detrended Fluctuation Analysis (DFA)**

DFA is used to investigate long-range correlations and scaling behavior in time-series data. In other words, it helps describe patterns and relationships in a signal that may not be captured by basic statistical features.

**4) Sample Entropy (SampEn)**

Sample Entropy measures the irregularity or complexity of a time series. It provides information about how predictable or repetitive the signal patterns are.

I included this feature because EEG signals can have complex patterns that are not always well represented by simple measurements such as mean or variance.

**5) Multiscale Entropy (MSE)**

Unlike Sample Entropy, which evaluates complexity at a particular scale, MSE examines signal complexity across different time scales.

In this project, MSE was calculated up to scale 3, giving three values for each channel.

### 7.2 Wavelet-Based Features (DWT db2)

The other five features are based on the Discrete Wavelet Transform (DWT).

DWT is a signal processing technique that decomposes a signal into different frequency components at multiple levels. This is particularly useful for EEG because brain signals are non-stationary, meaning their characteristics can change over time.

I applied DWT with the Daubechies 2 wavelet (db2) at a decomposition level of 4. This produced five sets of coefficients, covering every sub-band from D1 to D4 and A4:

- Approximation coefficients at level 4 (A4)
- Detail coefficients at level 4 (D4)
- Detail coefficients at level 3 (D3)
- Detail coefficients at level 2 (D2)
- Detail coefficients at level 1 (D1)

For each of these five sub-bands, I calculated the following five features:

| Wavelet feature | What it describes |
|---|---|
| Standard Deviation | How widely coefficient values are spread. |
| Variance | How much coefficient values vary. |
| Average Wavelet Energy | The average squared magnitude of the coefficients. |
| Shannon Entropy | The uncertainty or distribution of coefficient values. |
| Logarithmic Energy Entropy | Information related to the energy distribution of the coefficients. |

This gives 5 features × 5 sub-bands = 25 wavelet-based values per channel.

### 7.3 Putting All Features Together

After extracting the features from each channel, I combined them into one feature vector for each EEG window.

The 10 features and the number of values each one contributes per channel are summarized below:

| # | Feature | Group | Values per channel |
|---|---|---|---:|
| 1 | Hjorth Parameters (Activity, Mobility, Complexity) | Time-domain and complexity | 3 |
| 2 | Zero-crossings | Time-domain and complexity | 1 |
| 3 | Detrended Fluctuation Analysis (DFA) | Time-domain and complexity | 1 |
| 4 | Sample Entropy (SampEn) | Time-domain and complexity | 1 |
| 5 | Multiscale Entropy (MSE, scales 1–3) | Time-domain and complexity | 3 |
| 6 | Standard Deviation | Wavelet-based (D1–D4, A4) | 5 |
| 7 | Variance | Wavelet-based (D1–D4, A4) | 5 |
| 8 | Average Wavelet Energy | Wavelet-based (D1–D4, A4) | 5 |
| 9 | Shannon Entropy | Wavelet-based (D1–D4, A4) | 5 |
| 10 | Logarithmic Energy Entropy | Wavelet-based (D1–D4, A4) | 5 |
| | **Total** | | **34** |

Since the dataset used 14 EEG channels:

\[
34 \times 14 = 476
\]

This means that every 4-second EEG window was represented by **476 numerical features**.

The complete process can be summarized as:

**EEG Signal → 4-Second Window → Feature Extraction per Channel → Feature Concatenation → 476 Features**

At this point, the raw EEG signals had been transformed into structured numerical data, ready to be used for machine learning.

## 8. Machine Learning Models

With the feature extraction process completed, I moved on to the machine learning stage.

I selected three models for comparison: SVM, kNN, and MLPNN. Each model approaches classification differently, so I wanted to see how they would perform using the same extracted features.

### 8.1 Support Vector Machine (SVM)

Support Vector Machine is a supervised learning algorithm that tries to find a decision boundary separating different classes.

I included SVM because it is a well-known classification method that can work with structured numerical features.

### 8.2 k-Nearest Neighbors (kNN)

k-Nearest Neighbors works by looking at the closest samples to a new data point and using their labels to make a prediction.

The basic idea is simple: samples that are close to one another in the feature space may belong to the same class.

### 8.3 Multilayer Perceptron Neural Network (MLPNN)

The third model was Multilayer Perceptron Neural Network, or MLPNN.

It is a feedforward neural network consisting of interconnected neurons arranged in layers. During training, it adjusts its weights to learn patterns from the input features.

I wanted to compare this model with the other two to see whether a neural network could capture useful relationships among the extracted EEG features.

### 8.4 Feature Scaling

Before training the models, I applied StandardScaler to the feature vectors.

This step transforms the features so that they have approximately zero mean and unit variance.

Importantly, the scaler was fitted using only the training data in each fold. The same fitted scaler was then applied to the corresponding test data.

This prevents information from the test set from being used when calculating the scaling parameters.

## 9. Experimental Design and Evaluation

After preparing the data and selecting the models, I needed a way to evaluate their performance.

I designed the experiments around two classification tasks and two evaluation protocols. (The results reported in the original GAMEEMO study are shown in Section 10.1 for reference.)

### 9.1 Classification Tasks

I started with the original four emotional categories from GAMEEMO.

**Four-Class Classification**

The model was trained to classify each EEG window into one of four categories:

- Boring
- Calm
- Horror
- Funny

I also wanted to investigate whether combining the categories into broader groups would make the classification task easier.

**Binary Classification**

For the second experiment, I grouped the four categories into two classes:

| Original category | Binary label |
|---|---|
| Boring | Negative |
| Horror | Negative |
| Calm | Positive |
| Funny | Positive |

This allowed me to compare the results of a four-class task with a simpler binary classification task.

### 9.2 Why Did I Choose 3-Fold Instead of 10-Fold?

One question that may come up is: if the original paper used 10-fold cross-validation, why did I choose 3-fold?

The reason was mainly related to the number of participants available in my experiment and the evaluation design I wanted to investigate.

I selected 18 participants in total. Since I also wanted to perform subject-wise evaluation, I needed to divide the participants into separate groups while keeping all data from the same participant together.

With 3-fold cross-validation, the 18 participants could be divided evenly into three groups of six.

For each fold:

- 12 participants were used for training.
- 6 different participants were used for testing.

This created a straightforward participant-level split and made it easier to explain and interpret the subject-wise evaluation.

On the other hand, using 10-fold cross-validation with only 18 participants would result in much smaller and uneven test groups, with approximately one or two participants in each fold.

There was also a practical consideration: every fold required the models to be trained and evaluated again. Using three folds reduced the number of repeated training runs and made the experiments more manageable.

Therefore, I chose 3-fold cross-validation as a practical approach for this project, considering the number of participants, the subject-wise evaluation design, and computational resources.

This does not mean that 3-fold is always better than 10-fold. They are simply different evaluation settings, and the results should be interpreted within the context of the experiment.

### 9.3 Window-Wise Evaluation (Protocol A)

For the first evaluation protocol, I used window-wise cross-validation with `KFold`.

The feature windows were shuffled and divided into three folds. In each round, two folds were used for training, while the remaining fold was used for testing.

The process was repeated three times until every fold had been used as the test set.

This approach evaluates the model using windows drawn from the same overall participant pool.

However, there is an important limitation: windows from the same participant can appear in both the training and testing sets.

This means the model may have already encountered participant-specific characteristics during training, which could make the task easier.

### 9.4 Subject-Wise Evaluation (Protocol B)

For the second protocol, I wanted to make the evaluation more challenging.

Instead of randomly splitting individual windows, I used `GroupKFold` and assigned participant IDs as groups.

With 18 participants and three folds, each test fold contained six participants, while the other 12 participants were used for training.

The key difference is that all windows belonging to the same participant stayed in the same fold.

Therefore, when the model was tested, it encountered participants whose data had never been used during training.

I considered this an important experiment because it gave me a better understanding of how the models might perform when facing new individuals.

### 9.5 Evaluation Metrics

To compare the models, I used two evaluation metrics: Accuracy and Macro F1-score.

**Accuracy** measures the proportion of predictions that the model classified correctly.

\[
Accuracy = \frac{\text{Correct Predictions}}{\text{Total Predictions}}
\]

**Macro F1-score** calculates the F1-score for each class and then takes the average, giving every class equal importance.

\[
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
\]

Using both metrics allowed me to look at overall classification performance as well as the balance between Precision and Recall across classes.

## 10. Results and Discussion

After completing the experiments, I compared the performance of SVM, kNN, and MLPNN under both evaluation protocols.

The results were quite interesting because the performance changed depending on how the data was divided.


### 10.1 Reference: Results Reported in the Original Study

Before looking at my own numbers, it helps to see what the original GAMEEMO study reported. I used it as a reference point for my reproduction.

![Original GAMEEMO paper results]({{ '/assets/images/paper-results.png' | relative_url }})

*Figure 3. Classification accuracy reported in the original GAMEEMO study.*

Keep in mind that the experimental settings are not identical. The original study used 10-fold cross-validation, while my experiments used 18 selected participants with 3-fold cross-validation. The comparison is therefore meant to give context rather than a one-to-one match.

### 10.2 Four-Class Classification

![Our 3-Fold Cross-Validation Results]({{ '/assets/images/model-results-3fold.png' | relative_url }})

*Figure 4. Model evaluation results using 3-fold cross-validation.*

| Model | Window-wise Accuracy | Window-wise F1 | Subject-wise Accuracy | Subject-wise F1 |
|---|---:|---:|---:|---:|
| kNN | 0.69 | 0.69 | 0.35 | 0.34 |
| SVM | 0.75 | 0.74 | 0.36 | 0.34 |
| MLPNN | 0.82 | 0.82 | 0.35 | 0.32 |

For window-wise evaluation, MLPNN achieved the highest performance, with an Accuracy and F1-score of approximately 0.82.

At first glance, these results looked promising. However, when I examined the subject-wise results, the scores dropped considerably.

All three models achieved Accuracy values of around 0.35–0.36, showing that classifying EEG signals from unseen participants was much more difficult.

This was an important finding because it showed that good performance on windows from the same participant pool does not necessarily mean that a model can generalize to new people.

### 10.3 Binary Classification

| Model | Window-wise Accuracy | Window-wise F1 | Subject-wise Accuracy | Subject-wise F1 |
|---|---:|---:|---:|---:|
| kNN | 0.79 | 0.79 | 0.56 | 0.55 |
| SVM | 0.79 | 0.79 | 0.55 | 0.50 |
| MLPNN | 0.85 | 0.85 | 0.55 | 0.52 |

For binary classification, the performance was generally higher than in the four-class experiment.

MLPNN achieved the highest window-wise Accuracy and F1-score, both approximately 0.85.

However, the subject-wise scores were still noticeably lower, with Accuracy values around 0.55–0.56.

This suggests that grouping the four emotional categories into two broader classes made the classification task easier, but it did not completely solve the problem of differences between participants.

### 10.4 How Do My Results Compare With the Original Study?

The window-wise protocol is the closest counterpart to a standard k-fold evaluation on pooled data, so it is the fairest point of comparison with the original study shown in Figure 3. Because windows from the same participant can appear in both training and testing sets, this protocol tends to give higher scores.

The subject-wise protocol is a stricter test that goes beyond what a standard k-fold setup measures. The large gap between the two protocols is the main additional insight from this reproduction.

### 10.5 What Did I Learn From the Results?

Looking at the results, I found three main things worth discussing.

**First, the evaluation protocol really matters.**

The difference between window-wise and subject-wise performance was substantial. This suggests that models can perform well when training and testing data come from the same participant pool, but struggle when tested on completely unseen participants.

**Second, individual differences are a major challenge.**

EEG signals vary from person to person. A model may learn patterns that work well for the participants it has already seen, but those patterns may not transfer easily to someone else.

**Finally, binary classification performed better than four-class classification.**

This was not particularly surprising because distinguishing between two broader groups is generally a simpler task than distinguishing among four separate categories.

Among the three models, MLPNN performed best in the window-wise experiments. However, its performance also decreased under subject-wise evaluation.

For me, the most important takeaway was that a high Accuracy score alone does not tell the whole story. We also need to understand how the model behaves when the testing conditions become more challenging.

## 11. Limitations and Future Work

Of course, this project still has several limitations, and there are many things I would like to improve in the future.

### 11.1 Limitations

The first limitation is the number of participants. Although the original GAMEEMO dataset contains 28 participants, I used only 18 for the experiments. This means that the results may not fully represent the variability present in the complete dataset.

Another limitation is subject variability. As seen in the subject-wise results, differences between individuals made classification much more difficult.

The dataset itself also comes from specific gaming conditions. Therefore, the emotional categories in GAMEEMO may not represent every emotional experience in everyday life.

Finally, the current classification performance, especially under subject-wise evaluation, still needs improvement before considering any practical application.

### 11.2 Future Work

If I had more time to continue this project, there are several directions I would like to explore.

**1. Improve data quality assessment**

I would like to investigate the EEG recordings more systematically and establish clearer criteria for assessing signal quality and selecting participants.

**2. Explore additional datasets**

Using other EEG emotion datasets could help determine whether the current feature extraction pipeline works across different datasets and recording conditions.

**3. Improve cross-subject generalization**

Since subject-wise performance was one of the biggest challenges, I would like to explore other preprocessing methods, feature selection techniques, and machine learning approaches that may reduce the effect of individual differences.

**4. Explore different evaluation strategies**

I would also like to investigate additional subject-independent evaluation methods to better understand how the models perform on people who were not included in the training data.

There is still a lot to explore, but these experiments gave me a clearer idea of where the main challenges are.

## 12. Conclusion

Looking back, this project was a valuable opportunity to explore EEG signal processing and machine learning from beginning to end.

I started with raw EEG recordings from the GAMEEMO dataset, explored the signals through EDA, prepared the data using 4-second segmentation, and extracted 476 features from each window using the 10 time-domain, complexity-based, and wavelet-based features from the original study.

I then compared three machine learning models: SVM, kNN, and MLPNN, using both four-class and binary classification tasks.

The experiments showed that MLPNN achieved the highest performance in the window-wise setting. However, the performance dropped considerably when the models were evaluated on unseen participants using subject-wise cross-validation.

This taught me an important lesson: building a machine learning model is not just about achieving a high score. It is also about understanding what the model has learned, how the data is structured, and whether the results can be trusted under different testing conditions.

Although the current results are not yet sufficient for real-world or clinical applications, this project has given me a foundation to continue learning and improving my understanding of EEG-based emotion recognition.

And honestly, beyond the technical knowledge, one of the most valuable parts of this experience was learning how to approach a problem, deal with unexpected results, and keep going even when things became confusing.

## A Little Appreciation for Our TAs

No project journey is complete without the people who helped make it possible.

I would like to give a special thank you to our TAs, **Pung and Ong**, for guiding and supporting me throughout this project.

First of all, let's be honest: having handsome and kind TAs definitely makes learning a little more enjoyable! But jokes aside, what I really appreciate is how approachable and friendly they are.

There were many moments when I had absolutely no background knowledge about a topic. Sometimes, I did not even know what questions I should ask. But instead of making me feel bad about not knowing something, they always tried their best to explain things in a way that I could understand.

Even when the topic was difficult or I needed things to be explained more than once, they were patient and willing to help me work through it.

Of course, they do have another side too. They can be a little strict sometimes... but just a little! (I promise, only a little.)

I think that balance is what makes them great TAs. They are friendly enough to make it comfortable to ask questions, but they also know when to push us to think harder and take our work seriously.

Thank you, Pung and Ong, for your patience, encouragement, and all the time you spent helping us. I genuinely appreciate everything you have done. This project would have been a lot harder without your guidance.

**Thank you for being not only great TAs, but also people who made this learning experience much more enjoyable!**

## References

1. GAMEEMO dataset paper and original dataset documentation. **[https://www.sciencedirect.com/science/article/abs/pii/S1746809420301075]**
