import pandas as pd
import pickle
import smtplib
import zipfile

import streamlit as st
from streamlit_player import st_player
from email.mime.text import MIMEText


# Extracting models
zip_file_path = "sentiment_analysis_model.zip"
pkl_file_name = "sentiment_analysis_model.pkl"

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    with zip_ref.open(pkl_file_name, 'r') as pkl_file:
        data = pickle.load(pkl_file)

model = data['model']
tfid_vectorizer = data['tfid_vectorizer']

# Function to predict sentiment
def predict_sentiment(text):
    prediction = model.predict(tfid_vectorizer.transform([text]))[0]
    if prediction == 0:
        return 'Negative'
    elif prediction == 1:
        return 'Neutral'
    else:
        return 'Positive'

# Texts used in Dashboards
background_text = """**The Rise of Apple**

In the world of tech, few companies have left as big a mark as Apple. From its start in a garage in 1976,
Apple grew into a global giant, known for its game-changing products. The Macintosh in 1984, iPod, an
iPhone all reshaped how we live, work, and connect.

The iPhone, launched in 2007, wasn't just a phone; it set new standards for mobile tech. Its user-friendly
design and cutting-edge features made it a global sensation. Beyond tech, it influenced culture and commerce,
becoming a symbol of luxury.

**The Impact of COVID-19**

The outbreak of COVID-19 in 2019 had profound effects on global society, including the technology sector.
As lockdowns and social distancing measures were implemented worldwide, people increasingly relied on
technology to stay connected, work remotely, and access essential services.

For Apple and its flagship product, the iPhone, the pandemic brought both challenges and opportunities.
On one hand, disruptions in the global supply chain and retail closures impacted iPhone sales and production.
On the other hand, the shift towards remote work and digital communication boosted demand for smartphones,
including the iPhone, as essential tools for staying connected in a socially distant world.

**Sentiment Analysis During COVID-19**

During the height of the pandemic in 2020 and 2021, sentiment towards the iPhone may have fluctuated as consumers'
needs and priorities evolved in response to changing circumstances. Factors such as reliability, connectivity,
and the availability of digital services likely became more salient considerations for consumers evaluating
their smartphone choices.

As we analyze iPhone reviews on platforms like Twitter, we can explore how sentiment towards the iPhone
shifted before, during, and after the COVID-19 pandemic. By examining trends and patterns in consumer opinions,
we can gain insights into the evolving role of the iPhone in the lives of users and its resilience in
the face of unprecedented challenges.

**Sources:**

1. Isaacson, Walter. *Steve Jobs*. Simon & Schuster, 2011.
2. Kahney, Leander. *Inside Steve's Brain*. Portfolio, 2008.
3. Graham, Lawrence. *The Steve Jobs Way: iLeadership for a New Generation*. McGraw-Hill Education, 2010.
4. Vance, Ashlee. *Elon Musk: Tesla, SpaceX, and the Quest for a Fantastic Future*. Ecco, 2015."""

data_processing = """
Below is a sample of the data obtained through Twitter crawling. The 'full_text' column contains the textual
content extracted from Twitter posts, which will serve as input for sentiment analysis models.

The sentiment analysis models have been trained using a comprehensive dataset comprising over 400,000 reviews
sourced from Amazon Customer Reviews on Electronic Products.

These models are designed to categorize the sentiment of each post into one of three distinct categories:

- Negative: Assigned a score of 0
- Neutral: Assigned a score of 1
- Positive: Assigned a score of 2
"""

sentiment_analysis_introduction = """
In this analysis, we conducted sentiment analysis of iPhone reviews on Twitter from three distinct periods:
before COVID (2017-2018), during COVID (2019-2021), and after COVID (2022-2023). We collected
approximately 7,500 data points (2,500 data per period), to perform hypothesis testing.
"""

hypothesis_testing = """
To evaluate the differences in sentiment towards the iPhone across the three periods, we conducted hypothesis
testing. There are many hypothesis testing (see the hypothesis testing cheat sheet below).
"""

kruskal_wallis_testing = """
However, for our purposes, the Kruskal-Wallis test emerged as the most suitable due to several considerations:

- **Non-Parametric Data**: Our sentiment data did not adhere to parametric assumptions, making tests like
ANOVA or t-test inappropriate.
- **Comparison of Multiple Sample Means**: With sentiment scores from three periods (pre-COVID, during COVID,
post-COVID), we needed a test capable of comparing multiple sample means, eliminating options like the Wilcoxon test.
- **Large Sample Size**: Our dataset's substantial size rendered tests like the Mann-Whitney test unsuitable.
"""

hypothesis_assumptions = """
Before conducting hypothesis testing, we made the following assumptions:

- **Data Representation**: Twitter data faithfully represents sentiments among iPhone users.
- **Sample Representativeness**: Our sample is a true reflection of the broader population of iPhone users on Twitter.
- **Sentiment Accuracy**: Sentiment analysis effectively categorizes tweets as positive, negative, or neutral.
- **Independence**: Observations within each group are independent.
- **Uniformity of Distributions**: All sentiment score distributions across periods exhibit similar shapes.
- **Significance Level**: We adopted a standard alpha risk level of 0.05.
"""

null_alternative_hypothesis = """
- **Null Hypothesis (H0)**: There is no significant difference in sentiment towards the iPhone across the three periods
(pre-COVID, during COVID, and post-COVID).
- **Alternative Hypothesis (H1)**: There is a significant difference in sentiment towards the iPhone across the three periods.

If the p-value is below a predetermined significance level (0.05), we reject the null hypothesis and conclude that
there is a significant difference in sentiment towards the iPhone across the three periods.
"""

ksuskal_wallis_code = """
#perform Kruskal-Wallis Test 
stats.kruskal(sentiment_score_pre_covid, sentiment_score_covid, sentiment_score_post_covid)
"""

ksuskal_wallis_results = "KruskalResult(statistic=2.9451819543342244, pvalue=0.22933052460195202)"

hypothesis_results = """
Upon conducting the Kruskal-Wallis test, we obtained the following result:

- **Test Statistic**: 2.945
- **p-value**: 0.229
Given that the p-value exceeds our significance level of 0.05, we failed to reject the null hypothesis. Consequently,
we conclude that there is no statistically significant difference in sentiment towards the iPhone across the three periods.
"""

code_and_notebook = """
For transparency and reproducibility, the source code for our project components is accessible via the following links:

- Github repo project: https://github.com/cindysuyitno/iPhone-Twitter-Analysis
- Crawling twitter: https://github.com/cindysuyitno/iPhone-Twitter-Analysis/blob/main/crawl_twitter.ipynb
- Models training: https://github.com/cindysuyitno/Sentiment-Analysis/blob/main/sentiment%20analysis.ipynb
- Sentiment predicting: https://github.com/cindysuyitno/iPhone-Twitter-Analysis/blob/main/predict_sentiment.ipynb
- Hypothesis testing: https://github.com/cindysuyitno/iPhone-Twitter-Analysis/blob/main/hypothesis_testing.ipynb
- Streamlit: https://github.com/cindysuyitno/iPhone-Twitter-Analysis/edit/main/streamlit.py
"""

conclusion = """
Based on our hypothesis testing results, there is insufficient statistical evidence to conclude that there is a difference in
Twitter user sentiment towards the iPhone across the pre-COVID, during COVID, and post-COVID periods. This suggests that
one possible factor contributing to the iPhone's sustained popularity and dominance in the market is its consistent
performance and service offerings to users.

However, there are areas for potential improvement in this project:

- **Model Development**: The Sentiment Analysis model used in this project was trained on Amazon customer reviews,
which may not fully capture the language and nuances present in Twitter data. Further refinement of the model using
Twitter-specific data could enhance its accuracy in predicting sentiment from tweets.

- **Data Collection**: Due to restrictions imposed by Twitter's policies, data collection was limited, resulting in
a bias towards sentiment expressed during certain periods of the year, particularly in June-July and November-December.
A more comprehensive approach to data collection, covering a wider range of time periods, could provide a more representative
sample of user sentiment throughout the year.

Addressing these aspects could lead to a more robust analysis and a deeper understanding of user sentiment towards the iPhone on Twitter.
"""


# Streamlit app
def main():
    st.set_page_config(
        page_title="Ex-stream-ly Cool App",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Sentiment Analysis Dashboard")
    st.markdown("by: Cindy Suyitno")
    
    # Introduction tab
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Background and Introduction",
        "Data Processing and Analysis",
        "Predict Your Sentiment Text",
        "Conclusion and Review"])

    if page == "Background and Introduction":
        st.header("Background")

        st.subheader("Top 10 S&P 500 Companies by Market Cap (1980-2020):")
        video_url = "https://www.youtube.com/watch?v=kfMFDcuDKYA"
        st_player(video_url)

        st.markdown(background_text)

    elif page == "Data Processing and Analysis":
        st.header("Data Processing")

        df = pd.read_csv("sample_sentiment.csv", usecols=[
            'created_at', 'id_str', 'full_text', 'quote_count', 'reply_count', 'retweet_count',
            'favorite_count', 'lang', 'user_id_str', 'conversation_id_str', 'username', 'tweet_url',
            'Sentiment', 'Sentiment_Clause'
        ])

        st.markdown(data_processing)
        st.write(df.head())

        st.header("Analysis")
        st.subheader("Sentiment Analysis of iPhone Reviews on Twitter")
        st.write(sentiment_analysis_introduction)

        st.subheader("Hypothesis Testing")
        st.write(hypothesis_testing)
        image_source = "https://sixsigmadsi.com/wp-content/uploads/2018/07/Hypothesis-Testing-Roadmap-Continuous-Data.jpg"
        st.image("Hypothesis-Testing-Roadmap-Continuous-Data.jpg",
                 caption=f"Image Source: [Source Name]({image_source})",
                 width=700)
        st.write(kruskal_wallis_testing)

        st.subheader("Hypothesis Assumptions")
        st.write(hypothesis_assumptions)

        st.subheader("Null and Alternative Hypotheses")
        st.write(null_alternative_hypothesis)

        st.subheader("Kruskal-Wallis Test")
        st.code(ksuskal_wallis_code, language='python')
        st.write(ksuskal_wallis_results)

        st.subheader("Hypothesis Testing Results")
        st.write(hypothesis_results)

        st.header("Code and Notebook")
        st.markdown(code_and_notebook)

    elif page == "Predict Your Sentiment Text":
        st.header("Predict Your Sentiment Text!")
        text_input = st.text_area("Enter text to predict sentiment:")
        if st.button("Predict"):
            if text_input:
                sentiment = predict_sentiment(text_input)
                st.write("Predicted Sentiment:", sentiment)
            else:
                st.warning("Please enter some text.")

    elif page == "Conclusion and Review":
        st.header("Conclusion and Review")
        st.subheader("Conclusion")
        st.write(conclusion)
        
        st.subheader("Submit Your Review")
        st.write("If there is anything you want to discuss or comment, please write your review here:")
        email_sender = st.text_input('From')
        subject = st.text_input('Subject')
        body = st.text_area('Body')

        if st.button("Send Email"):
            if body:
                try:
                    msg = MIMEText(body)
                    msg['From'] = email_sender
                    msg['Subject'] = subject

                    server = smtplib.SMTP('smtp.gmail.com', 587)
                    server.starttls()
                    server.login("your_email@gmail.com", "your_email_password")
                    server.sendmail(email_sender, 'cyndisuyitno@gmail.com', msg.as_string())
                    server.quit()

                    st.success('Thank you for your review! It has been submitted. 🚀')
                except Exception as e:
                    st.error(f"Oops! There is an error : {e}")
            else:
                st.warning("Please write a review before submitting.")

if __name__ == "__main__":
    main()
