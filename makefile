qr : obj/qrdecomp.o obj/gridscheduler.o obj/gpucalc.cu.o makefile
	nvcc -arch=sm_20 obj/gpucalc.cu.o obj/qrdecomp.o obj/gridscheduler.o -O2 -lm -lpthread -g -G -o qr
obj/gpucalc.cu.o : include/gpucalc.h src/gpucalc.cu
	nvcc -arch=sm_20 -c src/gpucalc.cu -g -G -o obj/gpucalc.cu.o
obj/qrdecomp.o : qrdecomp.c qrdecomp.h
	gcc -c qrdecomp.c -g -o obj/qrdecomp.o
obj/gridscheduler.o : include/gridscheduler.h src/gridscheduler.c
	gcc -c src/gridscheduler.c -g -Wall -Wunused -o obj/gridscheduler.o
clean : 
	rm obj/* qr
