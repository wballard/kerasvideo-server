
install:
	pip install -r requirements.txt
.PHONY: install

#server will need to be running
test:
	curl -F file=@var/data/sample-0.jpg http://localhost:5000/mnist/classify

debug:
	python server.py