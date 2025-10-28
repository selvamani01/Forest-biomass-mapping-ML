# Forest-biomass-mapping-ML
Machine Learning model to map and estimate forest biomass in India
🌲 Forest Biomass Mapping using Machine Learning

This project focuses on predicting aboveground carbon stock in forest regions using machine learning techniques. Forests act as major carbon sinks and play a vital role in regulating the global carbon cycle. Accurate estimation of forest biomass helps in understanding carbon sequestration potential, supporting sustainable forest monitoring, policy development, and conservation efforts. By analyzing Earth observation-based datasets, the model estimates how much carbon is stored in specific forested areas.

The application is built using a Random Forest Regression model that takes forest-related variables as inputs such as tree cover percentage, forest extent over the years, aboveground carbon density, and annual tree cover loss from 2001 to 2022. Based on this input, the system predicts the total carbon stock for each region provided. The completed model is integrated into a Flask-based web application where users can upload a CSV file containing the input features. The system will process the data, generate predictions, display the results on the webpage, and allow users to download the output as a CSV file.

This tool provides an efficient method for evaluating forest biomass at scale, enabling faster and more accurate assessments than manual field methods. It supports decision-making in climate action, carbon credit evaluation, and environmental research. The web interface is user-friendly and requires only a single CSV upload to generate results. The project uses technologies such as Python, Flask, Pandas, Scikit-learn, HTML, and GitHub for version control.

In the future, the application can be expanded by integrating geospatial visualizations like forest maps, deploying on cloud platforms, adding live forest loss updates using APIs, and incorporating model interpretability dashboards to better explain the predictions. This enhances the usability of the system for forestry experts, researchers, and environmental organizations.

Developed by Selvamani P, this project demonstrates practical use of machine learning in ecosystem monitoring and contributes to improving sustainable forest management practices. ⭐ Feel free to star the repository if you find the work helpful and inspiring!
