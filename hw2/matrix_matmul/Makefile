all:matmul.o
	g++ -std=c++11 main.cpp matmul.o -o main

matmul.o:matmul.cpp
	g++ -c matmul.cpp -O3

clean:
	rm -rf *.o main
