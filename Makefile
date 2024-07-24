test:
	pytest tests/

quality-checks:
	isort .
	black .
	pylint . --recursive=y --fail-under=9

start:
	docker-compose up -d

stop:
	docker-compose down

update-data:
	awslocal s3api create-bucket --bucket mlflow
	awslocal s3api create-bucket --bucket data
	awslocal s3api put-object --bucket data --key diamonds.csv --body input/diamonds/diamonds.csv 

logs:
	docker-compose logs -f

deploy-prefect:
	docker-compose run prefect_deploy python deploy.py

train-model:
	python train/train_flow.py

setup:
	conda create -n diamond-price-prediction python=3.12
	# conda init
	# conda activate diamond-price-prediction
	pip install -r train/requirements.txt
	pip install pytest black isort pylint 