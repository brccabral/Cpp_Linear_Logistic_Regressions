CC = g++
INCLUDE_DIR := $(PWD)/include
CPPFLAGS := -g

all: objdir obj/ETL.o
	$(CC) $(CPPFLAGS) -o main -I$(INCLUDE_DIR)/ main.cpp obj/ETL.o

objdir:
	mkdir -p obj

obj/ETL.o: $(PWD)/ETL/ETL.cpp
	$(CC) $(CPPFLAGS) -o obj/ETL.o -I$(INCLUDE_DIR)/ -c $(PWD)/ETL/ETL.cpp