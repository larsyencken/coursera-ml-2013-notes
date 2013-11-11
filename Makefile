#
#  Makefile
#

TARGETS = \
		  pdf/.gitignore \
		  pdf/week1.pdf \
		  pdf/week2.pdf \
		  pdf/week3.pdf \

PANDOC = pandoc -t latex -H header.tex --latex-engine=xelatex --variable fontsize=12pt

default: $(TARGETS)

pdf/.gitignore:
	git clone -b pdf https://github.com/larsyencken/coursera-2013-ml-notes

pdf/week1.pdf: week1.md header.tex
	$(PANDOC) $< -o $@

pdf/week2.pdf: week2.md header.tex
	$(PANDOC) $< -o $@

pdf/week3.pdf: week3.md header.tex
	$(PANDOC) $< -o $@


watch:
	make
	ifchanges.py --cmd=make *.md
