all: compile link

compile:
	g++ -c main.cpp
	
# add -mwindows at end of link to hide console
link:
	g++ main.o -o main 	
