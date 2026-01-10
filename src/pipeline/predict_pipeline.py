

import os
import sys
from typing import Any, Dict, Optional

import dill
import numpy as np
import pandas as pd

from src.exception import CustomException


class PredictPipeline:
	def __init__(
		self,
		*,
		model_path: Optional[str] = None,
		preprocessor_path: Optional[str] = None,
	) -> None:
		artifacts_dir = "artifacts"
		self.model_path = model_path or os.path.join(artifacts_dir, "model.pkl")
		self.preprocessor_path = preprocessor_path or os.path.join(
			artifacts_dir, "preprocessor.pkl"
		)

	def predict(self, features: pd.DataFrame) -> np.ndarray:
		"""Return predictions for the provided feature dataframe."""
		try:
			model = self._load_object(self.model_path)
			preprocessor = self._load_object(self.preprocessor_path)
			transformed = preprocessor.transform(features)
			return model.predict(transformed)
		except Exception as exc:  # pragma: no cover - error path wrapper
			raise CustomException(exc, sys) from exc

	def predict_from_dict(self, payload: Dict[str, Any]) -> float:
		"""Helper to accept raw dict payload and return a single prediction."""
		custom_data = CustomData(**payload)
		features = custom_data.to_dataframe()
		result = self.predict(features)
		return float(result[0])

	@staticmethod
	def _load_object(file_path: str) -> Any:
		with open(file_path, "rb") as file_obj:
			return dill.load(file_obj)


class CustomData:
	"""Container to build a dataframe from raw request values."""

	def __init__(
		self,
		gender: str,
		race_ethnicity: str,
		parental_level_of_education: str,
		lunch: str,
		test_preparation_course: str,
		reading_score: float,
		writing_score: float,
	) -> None:
		self.gender = gender
		self.race_ethnicity = race_ethnicity
		self.parental_level_of_education = parental_level_of_education
		self.lunch = lunch
		self.test_preparation_course = test_preparation_course
		self.reading_score = reading_score
		self.writing_score = writing_score

	def to_dataframe(self) -> pd.DataFrame:
		data: Dict[str, list[Any]] = {
			"gender": [self.gender],
			"race_ethnicity": [self.race_ethnicity],
			"parental_level_of_education": [self.parental_level_of_education],
			"lunch": [self.lunch],
			"test_preparation_course": [self.test_preparation_course],
			"reading_score": [float(self.reading_score)],
			"writing_score": [float(self.writing_score)],
		}
		return pd.DataFrame(data)


def get_prediction_from_raw_inputs(payload: Dict[str, Any]) -> np.ndarray:
	"""Utility for FastAPI routes: accepts dict payload and returns prediction."""
	custom_data = CustomData(**payload)
	df = custom_data.to_dataframe()
	pipeline = PredictPipeline()
	return pipeline.predict(df)


if __name__ == "__main__":  
	sample = CustomData(
		gender="female",
		race_ethnicity="group B",
		parental_level_of_education="bachelor",
		lunch="standard",
		test_preparation_course="none",
		reading_score=72,
		writing_score=70,
	)
	print(get_prediction_from_raw_inputs(sample.__dict__))
