PYTHON=python3
FORTHON=Forthon3

example: example.F example.v example_extra.F
	$(FORTHON) --no2underscores -g example example_extra.F
	mv build/*/*/*.so .
	$(PYTHON) example.py

clean:
	rm -rf build examplepy*.so
