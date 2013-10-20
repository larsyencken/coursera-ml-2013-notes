#
#  Makefile
#

TARGETS = \
		  pdf/.gitignore \
		  pdf/week1.pdf \

default: $(TARGETS)

pdf/.gitignore:
	git clone -b pdf https://github.com/larsyencken/coursera-2013-ml-notes

pdf/week1.pdf: week1.md
	pandoc -t latex -H header.tex --latex-engine=xelatex $< -o $@

watch:
	make
	ifchanges.py --cmd=make *.md
