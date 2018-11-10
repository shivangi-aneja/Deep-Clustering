# for calling other targets within given one
THIS_FILE := $(lastword $(MAKEFILE_LIST))

test:
	@$(MAKE) -f $(THIS_FILE) clean
	nosetests deep_clustering/ --config .noserc
	@$(MAKE) -f $(THIS_FILE) clean

clean:
	find . -name '*.pyc' -type f -delete

.PHONY: test clean
