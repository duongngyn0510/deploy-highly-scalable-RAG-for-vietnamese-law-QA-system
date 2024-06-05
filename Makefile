# Any args passed to the make script, use with $(call args, default_value)
args = `arg="$(filter-out $@,$(MAKECMDGOALS))" && echo $${arg:-${1}}`

########################################################################################################################
# Quality checks
########################################################################################################################
black:
	poetry run black . --check

format:
	poetry run black .

mypy:
	poetry run mypy src

check:
	make format
	make mypy

########################################################################################################################
# Run
########################################################################################################################

run:
	poetry run python -m src

dev:
	PYTHONUNBUFFERED=1 PGPT_PROFILES=local poetry run python -m uvicorn src.main:app --reload --port 8001

########################################################################################################################
# Misc
########################################################################################################################

list:
	@echo "Available commands:"
	@echo "  black           : Check code format with black"
	@echo "  format          : Format code with black and ruff"
	@echo "  mypy            : Run mypy for type checking"
	@echo "  check           : Run format and mypy commands"
	@echo "  run             : Run the application"
	@echo "  dev             : Run the application in development mode"
