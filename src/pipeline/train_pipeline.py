

import sys
from typing import Any, Dict

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


class TrainingPipeline:
	def __init__(self) -> None:
		self.data_ingestion = DataIngestion()
		self.data_transformation = DataTransformation()
		self.model_trainer = ModelTrainer()

	def run(self) -> Dict[str, Any]:
		"""Run the full training pipeline and return artifact metadata."""
		try:
			logging.info("Starting training pipeline")

			train_path, test_path = self.data_ingestion.initiate_data_ingestion()
			train_arr, test_arr, preprocessor_path = (
				self.data_transformation.initiate_data_transformation(
					train_path, test_path
				)
			)
			r2_score = self.model_trainer.initiate_model_trainer(train_arr, test_arr)
			model_path = self.model_trainer.model_trainer_config.trained_model_file_path

			logging.info("Training pipeline finished")
			return {
				"train_data_path": train_path,
				"test_data_path": test_path,
				"preprocessor_path": preprocessor_path,
				"model_path": model_path,
				"r2_score": r2_score,
			}
		except Exception as exc:
			raise CustomException(exc, sys) from exc


def run_training_pipeline() -> Dict[str, Any]:
	
	pipeline = TrainingPipeline()
	return pipeline.run()


if __name__ == "__main__":
	run_training_pipeline()
