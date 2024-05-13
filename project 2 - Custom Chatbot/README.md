# Project Overview
## Project: Build Your Own Custom Chatbot

In this project, the task is to customize a chatbot by changing the dataset to build a custom chatbot.

### What Will Be Built?

Upon completion of this project, there will be a custom OpenAI chatbot tailored to a scenario of choice. The project involves selecting a data source, justifying its suitability for the task, incorporating it into the custom chatbot code, and formulating questions to showcase the performance of the custom prompt.

### Data Sources

The project recommends two main data sources: Wikipedia API and CSV data.

#### Wikipedia API
The Wikipedia API offers similarity to the examples demonstrated in previous demos and exercises. Any article other than [2022](https://en.wikipedia.org/wiki/2022) or [2023 Syria-Turkey Earthquake](https://en.wikipedia.org/wiki/2023_Syria%E2%80%93Turkey_earthquake) can be utilized provided it meets the project requirements.

#### CSV Data
The project includes a data directory containing CSV files for use:
- `2023_fashion_trends.csv`: This file contains reports and quotes about fashion trends for 2023, including source URL, article title, and text snippet.
- `character_descriptions.csv`: This file contains character descriptions from various media, including theater, television, and film productions, providing the name, description, medium, and setting of each character, all invented by an OpenAI model.
- `nyc_food_scrap_drop_off_sites.csv`: This file contains information about food scrap drop-off sites in New York City, including locations, hours, and other details. The data was retrieved in early 2023.

There's also the option to use custom data sources, such as web scraping or existing documents, provided the dataset comprises at least 20 rows of text data. Number-heavy data, such as budgets or inventory, is not suitable as OpenAI language models are not optimized for numeric or logical reasoning.

### Custom Scenario

In addition to the technical aspect of preparing and incorporating a new dataset, the project requires an explanation of why the chosen dataset is appropriate for the task. Explanations are provided in two sections:

1. **Introduction Paragraph**: At the beginning of the notebook, there will be a brief paragraph describing the dataset choice and setting up the scenario where this customization would be advantageous.
   
2. **Demonstration of Customization**: At the end of the notebook, the project will demonstrate the model's question and answer (Q&A) performance before and after customization to highlight the changes.
