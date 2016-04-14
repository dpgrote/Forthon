FORTHON=Forthon
PYTHON=python

example: example.F example.v
	$(FORTHON) example
	mv build/*/*/*.so .
	$(PYTHON) example.py

clean:
	rm -rf build *.so
