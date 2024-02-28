# iPhone-Twitter-Analysis
From nothing to the top S&P Companies by Market Cap in 2020, Apple has mark its teritory as a global giant. The iPhone, launched in 2007, wasn't just a phone; it set new standards for mobile tech. It influenced culture and commerce, becoming a symbol of luxury. During the height of the pandemic in 2020 and 2021, sentiment towards the iPhone may have fluctuated as consumers' needs and priorities evolved in response to changing circumstances. Factors such as reliability, connectivity, and the availability of digital services likely became more salient considerations for consumers evaluating their smartphone choices. The aim of this project is to analyze iPhone reviews on platforms like Twitter to examine if the sentiment towards the iPhone shifted before, during, and after the COVID pandemic. By examining trends and patterns in consumer opinions, we can gain insights into the evolving role of the iPhone in the lives of users and its resilience in the face of unprecedented challenges.

See the summary of this project in slides here: https://drive.google.com/file/d/1cd6clQaeyu2BPy3Z612K94MuWOI2lWiu/view?usp=sharing and in dashboard here: https://iphone-sentiment-analysis.streamlit.app/

# Technologies used
Python (Pandas, Matplotlib, Seaborn, Pickle, Sklearn, Scipy), Streamlit

## Table of Content:
### Data Collection:
  - Sources: Twitter, obtained by crawling
  - Contains Full_text of twits
### Data Cleaning:
  - Removing stop words and transform text to feature by TF-IDF Vectorizer
  - Predict Sentiment of the twitter text
  - Grouping data into three periods: pre-COVID, during COVID, post-COVID
### Hypothesis Testing:
  - Hypothesis testing used: ANOVA, Paired t-test, Mann Whitney U-Test, Wilcoxon Signed-Rank Test, Kruskal-Wallis Test
  - The result is fail to reject null hypothesis (p-value is more than the significance level, which is 0.05)
  - Conclusion: there is not enough evidence to conclude that the sentiment score is difference before, during, and after COVID.
### Webpage by Streamlit:
  - The twitter analysis is displayed using Streamlit
  - The dashboard can be seen here: https://iphone-sentiment-analysis.streamlit.app/

# Ilustration of Web App:
![image](https://github.com/cindysuyitno/iPhone-Twitter-Analysis/assets/105575967/1dc47477-d70f-4902-9512-051b077cef18)
![image](https://github.com/cindysuyitno/iPhone-Twitter-Analysis/assets/105575967/7d73432e-8967-4554-b3d5-8fbed81284e5)
![image](https://github.com/cindysuyitno/iPhone-Twitter-Analysis/assets/105575967/ae6e3eb9-3204-4878-b494-cdc9630b2744)

# Comments and Suggestion from Author
Based on our hypothesis testing results, there is insufficient statistical evidence to conclude that there is a difference in Twitter user sentiment towards the iPhone across the pre-COVID, during COVID, and post-COVID periods. This suggests that one possible factor contributing to the iPhone's sustained popularity and dominance in the market is its consistent performance and service offerings to users.

However, there are areas for potential improvement in this project:
- Model Development: The Sentiment Analysis model used in this project was trained on Amazon customer reviews, which may not fully capture the language and nuances present in Twitter data. Further refinement of the model using Twitter-specific data could enhance its accuracy in predicting sentiment from tweets.
- Data Collection: Due to restrictions imposed by Twitter's policies, data collection was limited, resulting in a bias towards sentiment expressed during certain periods of the year, particularly in June-July and November-December. A more comprehensive approach to data collection, covering a wider range of time periods, could provide a more representative sample of user sentiment throughout the year.

Addressing these aspects could lead to a more robust analysis and a deeper understanding of user sentiment towards the iPhone on Twitter.
