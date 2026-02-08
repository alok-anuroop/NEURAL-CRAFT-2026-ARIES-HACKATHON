I would like to explain my approach here:
I used to use a Transformer because it performs greatly in understanding the context as it reads the words in a sentence in parallel.The one I used was Bert.
I created explore_data.py,config.py and then created and ran Baseline.py for vectorisation using TF-IDF vectoriser which enables Bert to understand data better.
I ran 5 epoches initially but then increased to 7 epoches to make the model train better.
Increasing the number of epoches further wouldlead to overfitting of data,so i stopped at this point
I had also divided the training data into training and validation sets.
Validation set helped me decide which is the best model and whether to save the changes after each epoch as sometimes epoches make it worse due to overfitting.

