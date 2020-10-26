PYTHON=python
PYTHON_VERSION := $(shell $(PYTHON) -c "import sys;print(sys.version_info.major)")

# Check if python ran successfully, whether PYTHON_VERSION was given a value.
ifeq ($(PYTHON_VERSION), )
    # Running python failed.
    # Try python3
    PYTHON=python3
    # Note that the version is fetched again to make sure that python3 can be found.
    PYTHON_VERSION := $(shell $(PYTHON) -c "import sys;print(sys.version_info.major)")
endif

ifeq ($(PYTHON_VERSION), 2)
    FORTHON=Forthon
else ifeq ($(PYTHON_VERSION), 3)
    FORTHON=Forthon3
else
        @echo $(PYTHON) executable not found
endif

example: example.F example.v example_extra.F
	$(FORTHON) --no2underscores -g example example_extra.F
	mv build/*/*/*.so .
	$(PYTHON) example.py

clean:
	rm -rf build examplepy.so
