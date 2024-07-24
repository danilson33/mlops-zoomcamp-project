# About the project

This project aims to predict diamond price according to it's parameters.

## Problem statement

Diamond is a solid form of carbon element that present in crystal structure that known as diamond cubic making it unique. Diamond is known with their hardness, good thermal conductivity, high index of refraction, high dispersion, and adamantine luster. The high luster gives diamond the ability to reflect lights that strikes on their surface thus giving them the ‘sparkle’.

Colour and clarity determine the price of diamond to be selected as jewelry gems. Jewelry diamonds have the lowest number of specific gravity with it happens to be very close to 3.52 with minimal impurities and defects. Quality of diamonds that are made into jewelry gems are determined by color, cut, clarity and carat weight. Diamond attributes are as follows:

• Colour: Most quality diamond ranging from colorless to slightly yellow, brown or grey. The highest and most valuable diamonds is the one that are completely colorless.

• Clarity: An ideal diamond is free from fracture and particles of foreign material within the gems as low clarity gems tends to degrade the appearance, reduce the strength of the stone thus lower its value.

• Cut: Quality of designs and craftsmanship determines the appearance of diamonds that later determines the price. Angles of facets cut, proportions of design and quality of polishing determines face-up appearance, brilliance, scintillation, pattern and fire. A perfect diamond stones are perfectly polished, highly reflective, emit maximum amount of fire, faceted faces equal in size and the edges meet perfectly also identical in shape.

• Carat: A unit of weight equal to 1/5 of a gram or 1/142 of an ounce. Small diamonds are usually cost less per carat because of its common presences.

Another category of diamonds that are currently becoming a trend among diamond jewelry lovers are colored diamonds that occur in variety of hues such as red, pink, yellow, orange, purple, blue, green, and brown. The quality of this diamond’s type is determined by intensity, purity, and quality of their colour, which, the most saturated and vivid colour hold a greater price.


## Data source

The dataset used in this project is the [Diamonds dataset](https://www.kaggle.com/datasets/shivam2503/diamonds). 
This classic dataset contains the prices and other attributes of almost 54,000 diamonds. It's a great dataset for beginners learning to work with data analysis and visualization.


## Approach

The project follows the following approach:

1. Model training and evaluation : A machine learning model is trained evaluated
2. Model tracking: The trained model is tracked using MLflow.
3. Workflow automation: The model training process is automated using Prefect.
4. Model deployment: The trained model is deployed using FastAPI.
5. Model monitoring: The deployed model is monitored to ensure that it continues to perform well.


## Requirements

The project requires the following dependencies:

- python 3.12 or higher
- docker
- miniconda


## How to run the project:

1. Clone the repository
```bash
git clone git@github.com:danilson33/mlops-zoomcamp-project.git
```
Download [diamonds dataset](https://www.kaggle.com/datasets/shivam2503/diamonds) .
Make sure the path is 
```
input/diamonds/diamonds.csv
```
2. Setup the environment: 
```bash
make setup
```
3. Run docker-compose: 
```bash
make start
```
This will start the MLflow server, the Prefect server, the Prefect agent and the fastapi server.
4. Update data and create needed buckets
```bash
make update-data
```
5. Train the model
```bash
make train-model
```
6. Deploy the Prefect workflow
```bash
make deploy-prefect
```
7. Prediction example
You can test API on [http://localhost:8000/docs](http://localhost:8000/docs)
or by curl:
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "carat": 0.23,
  "cut": "Ideal",
  "color": "E",
  "clarity": "SI2",
  "depth": 61.5,
  "table": 55,
  "x": 3.95,
  "y": 3.98,
  "z": 2.43
}'
```
Response:
```json
{
  "prediction": 164.3172607421875
}
```
8. [OPTIONAL] Run tests
```bash
make test
```
9. [OPTIONAL] Run quality checks
```bash
make quality-checks
```

## Docker services and ports

| **Service Name** | **Port** | **Description**                           |
|------------------|----------|-------------------------------------------|
| _localstack_     | 9000     | Localstack object storage server          |
| _mlflow_         | 5001     | MLflow server for managing ML experiments |
| _mlflow_db_      | 3307     | MySQL database for MLflow                 |
| _phpmyadmin_     | 8081     | Web-based MySQL database administration   |
| _app_            | 8000     | FastAPI application                       |
| _prefect_server_ | 4200     | Prefect server for workflow management    |
| _agent_          | N/A      | Prefect agent for executing workflows     |
| _prefect_deploy_ | N/A      | Prefect deployment for workflows          |


# Project structure:

- [notebooks](notebooks) - Folder with notebooks
  - [EDA](<notebooks/EDA.ipynb>) - Exploratory data analysis and data preparation
  - [Model selection](<notebooks/model training.ipynb>) - Model creation and selection
- [app](app) - Folder with fastapi web app
- [train](train) - Folder with train scripts
  - [model training](train/train_flow.py) - Script for data preparation and model training
  - [prefect deploy](train/deploy.py) - Script for prefect deploy
- [input](input) - Folder with data
- [tests](tests) - Folder with tests
- [images](images) - Folder with images
- [README.md](README.md) - Project description
- [Makefile](Makefile) - Makefile with project commands