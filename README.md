# Predicting High-Engagement Social Media Posts Based on Internet Trends

## Motivation
The motivation for our project stems from the intertwining of social media and daily life, alongside the rapid proliferation of online news which significantly impacts public discourse. As social media platforms like Instagram become primary channels for information dissemination, understanding and leveraging current internet trends from reputable sources becomes crucial. Our project aims to bridge the gap between real-world events and online social interaction by recommending Instagram posts based on web-scraped trends from trustworthy news outlets. This endeavor is not only timely but imperative, as it fosters a more informed social media landscape, enriching user engagement with topical, credible content. By incorporating a thorough analysis of news source trustworthiness, comment sentiment, and temporal dynamics, we strive to curate a more nuanced, insightful social media experience. This initiative resonates with a broader effort to mitigate the spread of misinformation while promoting a more enlightening, responsible social media interaction. Through leveraging advanced NLP tools such as sentiment analysis algorithms, topic modeling techniques (e.g., Latent Dirichlet Allocation), and Named Entity Recognition (NER) to identify and categorize current news trends, we aspire to contribute to the ongoing discourse on how data science and NLP can be harnessed to enhance digital literacy and foster a more informed citizenry in the digital age.

## Goal
In our project, we aim to create an intelligent recommendation system to suggest posting content to Instagram posters based on real-time trends harvested from credible news sources. By leveraging advanced Natural Language Processing (NLP) tools and methodologies, we aim to bridge the gap between trending real-world events and social media content, thus fostering a more informed and engaging social media landscape. The motivation for this goal is rooted in the works of Zhou et al. and Vaswani et al., which respectively highlight the potential of personalized recommendation systems and the transformative impact of the Transformer architecture in processing sequential data efficiently. Our endeavor addresses a significant gap in current literature by not only harnessing trending information but also scrutinizing the trustworthiness of the news sources and analyzing the sentiment encapsulated in discussions surrounding these trends.

The primary challenges en route to achieving this goal encompass accurately identifying and extracting trending topics from a plethora of online news sources, ensuring the credibility of these sources, and effectively analyzing the sentiment and temporal dynamics surrounding these trends. Moreover, devising a robust recommendation model that can tailor content suggestions to individual Instagram posters based on these extracted trends, while ensuring relevance and engagement, presents a complex challenge.

Additionally, should time permit, exploring the integration of other social media platforms to provide a more comprehensive content recommendation system is envisaged. Through solving these challenges and potentially achieving our stretch goals, we aspire to significantly contribute to the domain of data-driven social media content creation and dissemination, ultimately enhancing the digital literacy and interactive experience of the online community.

## Methodology
Our approach can be broken down into five distinct meta-steps that are as follows:

1. **Get the current trends for a geographic region of interest (in our case, we will use trends from the USA).** This will be done at a refresh rate set at 24 hours.

2. **Use the trends to extract relevant posts from different media houses (from the past 2 days) and rank them based on relevance to the topic and credibility of the source.** This will help us form the corpus (later converted to embeddings for each trending item).

3. **Obtain historical posts from a content publisher and create embeddings/vector representations of that content.** This will help us get the embeddings that we use for matching/comparison to understand better the type of content we should post from what is trending.

4. **Calculate scores of current trends with respect to the historical posts from the given content publisher.** To calculate scores, we would experiment with simpler approaches like TF-IDF embeddings and more complex approaches that use pre-trained models like BERT.

5. **Rank the content based on the best scores and come up with the "ideal post" depending on the rank.**

<b> In summary </b>, we will look at trending pieces and will be able to tailor them to suit the content of our audience. The idea stems from popular websites like <i>Buzzfeed</i> but without the need for any content writers.
