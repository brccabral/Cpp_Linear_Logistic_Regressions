CC = g++
INCLUDE_DIR := $(PWD)/include
CPPFLAGS := -O2

all: objdir obj/ETL.o obj/main_LinearRegression.o obj/main_LogisticRegression.o obj/LinearRegression.o obj/LogisticRegression.o
	$(CC) $(CPPFLAGS) -o linear -I$(INCLUDE_DIR)/ obj/main_LinearRegression.o obj/ETL.o obj/LinearRegression.o
	$(CC) $(CPPFLAGS) -o logistic -I$(INCLUDE_DIR)/ obj/main_LogisticRegression.o obj/ETL.o obj/LogisticRegression.o

objdir:
	mkdir -p obj

obj/ETL.o: $(PWD)/ETL/ETL.cpp
	$(CC) $(CPPFLAGS) -o obj/ETL.o -I$(INCLUDE_DIR)/ -c $(PWD)/ETL/ETL.cpp

obj/main_LinearRegression.o: $(PWD)/main/LinearRegression.cpp
	$(CC) $(CPPFLAGS) -o obj/main_LinearRegression.o -I$(INCLUDE_DIR)/ -c $(PWD)/main/LinearRegression.cpp

obj/LinearRegression.o: $(PWD)/LinearRegression/LinearRegression.cpp
	$(CC) $(CPPFLAGS) -o obj/LinearRegression.o -I$(INCLUDE_DIR)/ -c $(PWD)/LinearRegression/LinearRegression.cpp

obj/main_LogisticRegression.o: $(PWD)/main/LogisticRegression.cpp
	$(CC) $(CPPFLAGS) -o obj/main_LogisticRegression.o -I$(INCLUDE_DIR)/ -c $(PWD)/main/LogisticRegression.cpp

obj/LogisticRegression.o: $(PWD)/LogisticRegression/LogisticRegression.cpp
	$(CC) $(CPPFLAGS) -o obj/LogisticRegression.o -I$(INCLUDE_DIR)/ -c $(PWD)/LogisticRegression/LogisticRegression.cpp

clean:
	-rm -r obj