CC = g++
INCLUDE_DIR := $(PWD)/include
CPPFLAGS := -g -std=c++11

all: objdir obj/ETL.o obj/main_LinearRegression.o obj/LinearRegression.o
	$(CC) $(CPPFLAGS) -o linear -I$(INCLUDE_DIR)/ obj/main_LinearRegression.o obj/ETL.o obj/LinearRegression.o

objdir:
	mkdir -p obj

obj/ETL.o: $(PWD)/ETL/ETL.cpp
	$(CC) $(CPPFLAGS) -o obj/ETL.o -I$(INCLUDE_DIR)/ -c $(PWD)/ETL/ETL.cpp

obj/main_LinearRegression.o: $(PWD)/main/LinearRegression.cpp
	$(CC) $(CPPFLAGS) -o obj/main_LinearRegression.o -I$(INCLUDE_DIR)/ -c $(PWD)/main/LinearRegression.cpp

obj/LinearRegression.o: $(PWD)/LinearRegression/LinearRegression.cpp
	$(CC) $(CPPFLAGS) -o obj/LinearRegression.o -I$(INCLUDE_DIR)/ -c $(PWD)/LinearRegression/LinearRegression.cpp

clean:
	-rm -r obj