FORTHON=Forthon
PYTHON=python2

example: example.F example.v example_extra.f
	$(FORTHON) --no2underscores -g example example_extra.f
	mv build/*/*/*.so .
	$(PYTHON) example.py

clean:
	rm -rf build examplepy.so
