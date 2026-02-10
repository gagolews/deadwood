# Copyright (c) 2013-2026, Marek Gagolewski <https://www.gagolewski.com>

FILES_RMD = \
	python.Rmd \
	r.Rmd \
	sklearn_toy_example.Rmd

RMD_MD_OUTPUTS=$(patsubst %.Rmd,%.md,$(FILES_RMD))

%.md: %.Rmd
	./Rmd2md.sh "$<"

all : rmd

rmd : $(RMD_MD_OUTPUTS)

clean:
	rm -f $(RMD_MD_OUTPUTS)
