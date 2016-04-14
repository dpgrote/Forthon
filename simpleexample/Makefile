FORTHON=Forthon

example: example.F example.v
	$(FORTHON) example
	mv build/*/*/*.so .
	python example.py

clean:
	rm -rf build *.so
