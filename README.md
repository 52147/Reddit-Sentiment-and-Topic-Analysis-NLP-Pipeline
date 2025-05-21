# Reddit Sentiment and Topic Analysis – NLP Pipeline

This project explores the relationship between Reddit post content, sentiment, and community feedback (upvote scores). Using the `praw` API, Hugging Face Transformers, `spaCy`, and scikit-learn, we conducted a series of analyses on posts from r/MachineLearning to understand what factors correlate with post popularity.

## Overview

We investigated four core questions:

1. Which words are common in high-scoring posts?
2. Do positive posts exhibit different linguistic styles?
3. Are topic clusters associated with different sentiment and score patterns?
4. Are there posts that are emotionally positive but receive low scores? (Sentiment bias)

---

## 1. High vs Low Scoring Vocabulary (TF-IDF)

**Method**  
We split posts into:
- `high` score group: > 15 upvotes
- `low` score group: ≤ 15 upvotes

TF-IDF was used to extract and compare the top keywords from each group.

**Findings**

| Word            | High TF-IDF | Low TF-IDF | Diff     |
|-----------------|-------------|------------|----------|
| promotion       | 0.088       | 0.000      | +0.088   |
| using           | 0.076       | 0.000      | +0.076   |
| ai              | 0.081       | 0.020      | +0.060   |
| implementation  | 0.051       | 0.000      | +0.051   |

**Conclusion**  
High-scoring posts often include concrete terms like `implementation`, `training`, `open-source`.  
Low-scoring posts tend to use vague or question-based wording like “how”, “why”, “problem”.

---

## 2. Part-of-Speech (POS) Analysis by Sentiment

**Method**  
Using `spaCy`, we performed POS tagging to compare linguistic style between positive and negative posts.

**Findings**

| POS     | POSITIVE | NEGATIVE | Diff |
|---------|----------|----------|------|
| VERB    | 1        | 26       | -25  |
| ADJ     | 2        | 10       | -8   |
| NOUN    | 14       | 29       | -15  |

**Conclusion**  
Positive posts tend to be shorter and contain more adjectives/verbs indicating action (`released`, `new`, `efficient`).  
Negative posts have heavier noun usage and more auxiliary/modal verbs.

---

## 3. Topic Modeling + Sentiment/Score Correlation

**Method**  
Using Latent Dirichlet Allocation (LDA), we grouped Reddit posts into 3 topics based on their content.

**Topics and Examples**

| Topic | Top Words                                 | Avg Score | Sentiment (NEG/POS) |
|-------|--------------------------------------------|-----------|----------------------|
| 0     | feedback, immunity, proposal               | 11.14     | 6 / 1               |
| 1     | reuse, rope, v3, 2025                      | 21.66     | 6 / 3               |
| 2     | rl, training, representation               | 7.25      | 3 / 1               |

**Conclusion**  
Posts about tool reuse, implementation, or benchmark release (Topic 1) receive higher scores and more positive sentiment than theoretical or speculative content.

---

## 4. Sentiment Bias: Positive But Low-Scoring Posts

**Method**  
We built a linear regression model using:
sentiment_feature = sentiment_polarity × confidence

Then we calculated residuals between predicted and actual scores.

**Outliers Identified**

| Title                                | Score | Sentiment | Predicted | Residual |
|--------------------------------------|-------|-----------|-----------|----------|
| Benchmarking gender bias             | 4     | POSITIVE  | 24.9      | -20.9    |
| YFlow framework post                 | 0     | POSITIVE  | 26.5      | -26.4    |

**Conclusion**  
Some posts that are objectively positive still underperform. Possible reasons include:
- Niche or highly technical content
- Poor timing or lack of visibility
- Misalignment with current community interests

---

## Technologies Used

- Python 3.x
- `praw` – Reddit API
- `transformers` – Hugging Face sentiment analysis
- `spaCy` – POS tagging
- `scikit-learn` – TF-IDF, regression, LDA
- `matplotlib`, `seaborn` – Visualization

---

## Future Work

- Expand to include Reddit comments (not just post titles)
- Add fine-tuned emotion classifiers (beyond binary sentiment)
- Integrate graph-based community interaction analysis (e.g., NetworkX)
