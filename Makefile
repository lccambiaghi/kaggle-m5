POETRY ?= PYTHONPATH=src poetry run

notebook:
	JUPYTER_PATH=src poetry run jupyter notebook
