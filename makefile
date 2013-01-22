qr : obj/qrdecomp.o obj/gridscheduler.o
	gcc obj/qrdecomp.o obj/gridscheduler.o -Wall -Wunused -lm -o qr -O2
obj/qrdecomp.o : qrdecomp.c qrdecomp.h
	gcc -c qrdecomp.c -Wall -Wunused -o obj/qrdecomp.o
obj/gridscheduler.o : include/gridscheduler.h src/gridscheduler.c
	gcc -c src/gridscheduler.c -Wall -Wunused -o obj/gridscheduler.o
clean : 
	rm obj/* qr
