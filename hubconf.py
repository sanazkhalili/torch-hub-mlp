# import the necessary packages
import torch
from pyimagesearch import mlp
import os

# define entry point/callable function 
# to initialize and return model
def custom_model():
	""" # This docstring shows up in hub.help()
	Initializes the MLP model instance
	Loads weights from path and
	returns the model
	"""
	# initialize the model
	# load weights from path
	# returns model
	model = mlp.get_training_model()
	model_path = os.path.join(repo_dir, "output", "model_wt.pth")
	if not os.path.exists(model_path):
		raise FileNotFoundError(f"Model file not found at {model_path}")
	model.load_state_dict(torch.load(model_path))
	return model
