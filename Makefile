CC = g++
INCLUDE_DIR := $(PWD)/include
CPPFLAGS := -g -std=c++11

all: objdir obj/ETL.o obj/main.o
	$(CC) $(CPPFLAGS) -o main -I$(INCLUDE_DIR)/ obj/main.o obj/ETL.o

objdir:
	mkdir -p obj

obj/ETL.o: $(PWD)/ETL/ETL.cpp
	$(CC) $(CPPFLAGS) -o obj/ETL.o -I$(INCLUDE_DIR)/ -c $(PWD)/ETL/ETL.cpp

obj/main.o: $(PWD)/main.cpp
	$(CC) $(CPPFLAGS) -o obj/main.o -I$(INCLUDE_DIR)/ -c $(PWD)/main.cpp

clean:
	-rm -r obj