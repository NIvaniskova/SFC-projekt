##################################################################################
# Project:		ANFIS in adaptive control
# Course:		Soft Computing
# Year:			2025/2026
# File:			Makefile
# Date:			November 28, 2025
# Author:		Natalia Ivaniskova <xivanin00@stud.fit.vutbr.cz>
#
# Description:	Build script
##################################################################################

CC=python3
MAIN_FILE=src/main.py
ZIP_FILENAME=11-xivanin00.zip

all: clean run

run: $(MAIN_FILE)
	$(CC) $(MAIN_FILE)

pack: clean pack_files

pack_files: src data README.md requirements.txt Makefile SFC_technical_report.pdf install_dependencies.sh
	zip -r $(ZIP_FILENAME) $^

clean:
	rm -f $(ZIP_FILENAME)
	rm -f *.chr
