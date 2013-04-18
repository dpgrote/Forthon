
example: example.F example.v example_extra.f
	Forthon --no2underscores -g example example_extra.f
	mv build/*/examplepy.so .
	python example.py

clean:
	rm -rf build examplepy.so
