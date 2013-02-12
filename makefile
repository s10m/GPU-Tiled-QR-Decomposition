qr : obj/qrdecomp.o obj/gridscheduler.o obj/gpucalc.cu.o
	nvcc obj/gpucalc.cu.o obj/qrdecomp.o obj/gridscheduler.o -O2 -lm -lpthread -o qr
obj/gpucalc.cu.o : include/gpucalc.h src/gpucalc.cu
	nvcc -c src/gpucalc.cu -o obj/gpucalc.cu.o
obj/qrdecomp.o : qrdecomp.c qrdecomp.h
	gcc -c qrdecomp.c -o obj/qrdecomp.o
obj/gridscheduler.o : include/gridscheduler.h src/gridscheduler.c
	gcc -c src/gridscheduler.c -Wall -Wunused -o obj/gridscheduler.o
clean : 
	rm obj/* qr
