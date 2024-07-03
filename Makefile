clean:
	find . -path "*/__pycache__/*" -delete
	find . -type d -name "__pycache__" -empty -delete
	rm -rf .pytest_cache
	rm .coverage

lint:
	flake8 ./prot_graph/ ./scripts/

test:
	coverage run --source ./prot_graph/ -m --omit="*/tests/*" pytest ./tests/ && coverage report -m