ScSPM: main.o scspm.o utility.o denseSift.o sparse_coding.o
	g++ -o ScSPM main.o scspm.o utility.o denseSift.o sparse_coding.o \
		     `pkg-config matio opencv --libs`
	rm *.o
main.o: main.cpp 
	g++ -o main.o -c main.cpp `pkg-config opencv --cflags`
scspm.o: scspm.h scspm.cpp
	g++ -o scspm.o -c scspm.cpp `pkg-config opencv --cflags`
utility.o: utility.h utility.cpp
	g++ -o utility.o -c utility.cpp `pkg-config opencv --cflags`
denseSift.o: denseSift.h denseSift.cpp
	g++ -o denseSift.o -c denseSift.cpp `pkg-config opencv --cflags`
sparse_coding.o: sparse_coding.h sparse_coding.cpp
	g++ -o sparse_coding.o -c sparse_coding.cpp `pkg-config opencv --cflags`
clean:
	rm *.o ScSPM
