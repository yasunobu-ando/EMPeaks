install:
	.venv/bin/maturin develop
	sed -i '' '1s|.*|#!$(shell pwd)/.venv/bin/python3|' .venv/bin/empeaks

test:
	nosetests tests
