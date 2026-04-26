# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

You may complete this model card for whichever version you used, or compare both if you explored them.

## 1. Model Overview

**Model type:**  
Describe whether you used the rule based model, the ML model, or both.  
Example: “I used the rule based model only” or “I compared both models.”

I used the rule based model only at first.

**Intended purpose:**  
What is this model trying to do?  
Example: classify short text messages as moods like positive, negative, neutral, or mixed.

This model is trying to classify text messages into moods based on set of rules based on ground truth.

**How it works (brief):**  
For the rule based version, describe the scoring rules you created.  
For the ML version, describe how training works at a high level (no math needed).

Initially, the model looks at the ground truth of all sample posts. If anything in the new test corresponds, it follows the rules and scales posts accordingly.


## 2. Data

**Dataset description:**  
Summarize how many posts are in `SAMPLE_POSTS` and how you added new ones.

I added new posts using my own opinions with current slang, and negation and with words that change in meaning.

**Labeling process:**  
Explain how you chose labels for your new examples.  
Mention any posts that were hard to label or could have multiple valid labels.

"I just ate but I'm still hungry" I labeled as neutral, but it could be negative

**Important characteristics of your dataset:**  
Examples you might include:  

- Contains slang or emojis  
- Includes sarcasm  
- Some posts express mixed feelings  
- Contains short or ambiguous messages

-contains emojis
-contains current slang
-has difficult to label text

**Possible issues with the dataset:**  
Think about imbalance, ambiguity, or missing kinds of language.

-imbalance with very charged positive text

## 3. How the Rule Based Model Works (if used)

**Your scoring rules:**  
Describe the modeling choices you made.  
Examples:  

- How positive and negative words affect score  
- Negation rules you added  
- Weighted words  
- Emoji handling  
- Threshold decisions for labels

-single word tokens intially have a big impact
-negation words can flip the way a token affects the score

**Strengths of this approach:**  
Where does it behave predictably or reasonably well?
With obvious sentences, it gets it right consistently.

**Weaknesses of this approach:**  
Where does it fail?  
Examples: sarcasm, subtlety, mixed moods, unfamiliar slang.
Where subtle tokens or sarcasm, it doesn't score the text properly

## 4. How the ML Model Works (if used)

**Features used:**  
Describe the representation.  
Example: “Bag of words using CountVectorizer.”

Vectorizes words using labels and CountVectorizer, then characterizes vectorized input in how its similar to the sample ones, and changes label based on that.
**Training data:**  
State that the model trained on `SAMPLE_POSTS` and `TRUE_LABELS`.

**Training behavior:**  
Did you observe changes in accuracy when you added more examples or changed labels?

Adding more examples improved accuracy, especially when they were more balanced.

**Strengths and weaknesses:**  
Strengths might include learning patterns automatically.  
Weaknesses might include overfitting to the training data or picking up spurious cues.

Strengths: It learns directly from how humans talk, rather than rules
Weaknesses: it can overfit or underfit based on training data

## 5. Evaluation

**How you evaluated the model:**  
Both versions can be evaluated on the labeled posts in `dataset.py`.  
Describe what accuracy you observed.

It's more accurate with emojis and somewhat more subtle texts, but its often still wrong with sarcasm. After adding some more examples, it's able to identify better with some sarcastic phrases.

**Examples of correct predictions:**  
Provide 2 or 3 examples and explain why they were correct.

"I can't wait for this party!" - positive. It's correct because it identifies excitement in can't wait
"I can't wait to waste my time with some homework." - negative. It's correct because it identifies that "can't wait" is sarcastic because of the use of a negative word "waste"

**Examples of incorrect predictions:**  
Provide 2 or 3 examples and explain why the model made a mistake.  
If you used both models, show how their failures differed.

"I hate you lol" - negative. this could be positive because of the playful "lol" added after, but the ML model senses "hate" and says negative


## 6. Limitations

Describe the most important limitations.  
Examples:  

- The dataset is small  
- The model does not generalize to longer posts  
- It cannot detect sarcasm reliably  
- It depends heavily on the words you chose or labeled

The dataset is small and limited to one human's knowledge of the language. Another implementation of the English language will be considered differently.

## 7. Ethical Considerations

Discuss any potential impacts of using mood detection in real applications.  
Examples: 

- Misclassifying a message expressing distress  
- Misinterpreting mood for certain language communities  
- Privacy considerations if analyzing personal messages

If you are using this for say, classifying emotion. It may misclassify someone's emotions, which in medical or psychological implementations, may lead to misjudgments on treatments or bad diagnoses of conditions.

## 8. Ideas for Improvement

List ways to improve either model.  
Possible directions:  

- Add more labeled data  
- Use TF IDF instead of CountVectorizer  
- Add better preprocessing for emojis or slang  
- Use a small neural network or transformer model  
- Improve the rule based scoring method  
- Add a real test set instead of training accuracy only

We would improve the ML model by going through more diverse texts, like going through a dataset of X tweets and adding those. This way, the ML would understand the labeling of different cadences, tones, and slang that a large amount of people use.
