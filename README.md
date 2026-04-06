🏥 India Wellness Centers Analysis

A data analysis project focused on understanding how healthcare infrastructure is distributed across India, and what patterns emerge when we look beyond surface-level metrics.

📌 Problem Thinking

Most datasets can tell you what is happening.
This project tries to understand why it might be happening.

Instead of just counting wellness centers, the goal was to explore:

Why do some cities have higher concentration of centers?
Does more centers actually mean better doctor availability?
How do categories and cities interact with each other?
📊 Approach

The project was built in layers:

1. Data Cleaning

Real-world data came with inconsistencies:

Missing values
Formatting issues
Mixed data types

Cleaning was necessary before any meaningful analysis.

2. Exploratory Analysis

Rather than jumping to conclusions, multiple perspectives were explored:

Category-wise distribution of centers
City-wise concentration
Doctor availability across categories
Doctor-to-center ratios
Category mix in top cities
3. Pattern Exploration

Instead of treating features independently, relationships were explored:

City density (how crowded a city is in terms of centers)
Category frequency
Interaction between city and category

This helped uncover patterns that are not visible in isolated analysis.

4. Predictive Layer

A regression model was built to test whether these patterns actually carry predictive value.

Key idea:
Better features > more complex models

🔍 Key Observations
High number of centers does not always translate to better doctor availability
Certain categories dominate, which may indicate policy or demand bias
A few cities carry a disproportionate share of healthcare infrastructure
Interaction between features explains more than individual variables
⚙️ Tech Stack

Python
Pandas, NumPy
Matplotlib, Seaborn
Scikit-learn



🚀 Takeaway

This project reinforced a simple idea:

Data analysis is less about tools and more about perspective.
The quality of insights depends on the quality of questions being asked.

📬 Open to Feedback

If you notice something I could have explored differently or improved, I would genuinely like to hear your perspective.
