# Reddit Post Classifier
This is a machine learning project that can classify Reddit posts into their respective subreddits. The project is designed to take in a post's title as input and predict which subreddit the post belongs to. This project also aims to visualize the relationships between post embeddings using unsupervised Machine Learning techniques.

## Installation
1. Clone the repository: git clone https://github.com/gearymich/reddit-post-classifier.git
2. Install the required packages: pip install -r requirements.txt

## Usage
1. Train the model by running python train.py.
2. Classify a Reddit post by running python predict.py  --model "insert model here" --title "insert title here".
3. Our visualization of choice, T-SNE, can be built using the model_scripts/visuals.py script

## Methodology
The project uses various statistical methods for text classification, including Naive Bayes, XXX, XXX, and XXX. The method used for classification can be selected during training by editing the config.json file.

## Dataset
The project uses a dataset of Reddit posts that has been pre-processed and cleaned for use in this project. The dataset contains posts from various subreddits, and the project is designed to classify posts into one of these subreddits.

## Results
The project achieves an accuracy of X% on the test set, demonstrating its effectiveness in classifying Reddit posts into their respective subreddits.

## Future Work
Possible future improvements to the project include:
* Improving the pre-processing and cleaning of the dataset
* Experimenting with other statistical methods for text classification
* Implementing more advanced neural network architectures
* Developing a web application or browser extension for easy use of the model.

## Credits
This project was developed in equal parts by Samar Aldakheel, Megan Aloise, Michael Geary, and Bao Huynh as apart of WPI's CS 547 - Information Retrieval Graduate Course. The dataset used in the project was obtained from the publicly available Reddit API.
