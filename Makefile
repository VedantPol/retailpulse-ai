.PHONY: up test train

up:
	docker compose up --build

test:
	pytest -q

train:
	cd backend && python training/generate_data.py && python training/train_forecast_model.py && python training/train_recommender.py

