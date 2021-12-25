# GNN Based Robot Information Dissemination Prediction

This is a research focusing on the analysis of social robot information dissemination based on dynamic network. The architecture of the model is Graph Recurrent Neural Network.

This is also my graduation project of B.Eng degree in Xi'an Jiaotong University. 

## Description

Social media robots mean the robots that operating on social media. Being a part of social network, social media robots could imitate real human users and affect the public opinion by bruiting all kinds of information to users. Fake news reported by those robots might lead to unrest and panic. Therefore, in order to prevent social media robots from causing negative influence, researchers have been focusing on predicting the area of effects of the information they disseminate in social networks. The research of analysing if or when the information diffusion events would happen between robot users and human users could be abstracted as Link Prediction problem in social network. Nevertheless, analysing the information diffusion process is time-consuming due to social networks’ massive size and complexity. Besides, traditional machine learning algorhithms is not suitable for that research because social networks are changing unstoppably. For instance, a user will make new friends, or deliver new messages to others, giving high dynamics to social network and much more difficulty to Link Prediction problem. Hence, finding an effective and accurate link prediction method to capture the dynamic and structural features in social networks, and infering the information transmission tendency between robots and human users are burning issues in social media robots research. 

The key parts of this research are predicting the information diffusion process of social media robots by utilizing dynamic link prediction method. To begin with, a dynamic link prediction model based on graph neural network would be built and achieve seizing dynamic and structural features of social networks. This model was implemented on a small scale dataset Social Network. The correctness and effectiveness of this model could be proved by analysing the prediction results. Besides, this research processes a huger datasets named TwiBot-20 which contains social network robots information, and also implement the model which used previously onto this datasets and make link prediction. Apart from evaluating the performance of prediction, this research visualizes the experiment results to present the process of propagation and diffusion of information among robot users and genuine human users. 

The following conclusions were obtained from the analysis of the experiment results. Firstly, the built model is well adapted to dynamic social network, making relative accurate link prediction and learning highly expressive and discriminative node representation vector, and performing 0.99 accuracy rate in the dataset. Secondly, by forecasting when a information dissemination events will happen in social network, this model may be suitable for explore the scope of influence of social media robots in social network. This research is beneficial to more indepth study and analysis of social network and social media robots. 

## Dependencies
* OS version: Ubuntu 16.04.6 LTS
* CPU: Intel(R) Core(TM) i9-7900X 3.30GHz
* GPU: Nvidia Titan Xp 12GB
* Memory: 125.5GB

## Data sets
* TwiBot-20: https://github.com/BunsenFeng/TwiBot-20
* Social Evolution: http://realitycommons.media.mit.edu/socialevolution.html


## Authors
Mingjun Wang 
wangmnjn@gmail.com

