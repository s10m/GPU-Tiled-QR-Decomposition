qr : qrdecomp.o
	gcc obj/qrdecomp.o -Wall -lm -o qr -O2
qrdecomp.o : qrdecomp.c qrdecomp.h
	gcc -c qrdecomp.c -Wall -o obj/qrdecomp.o
clean : 
	rm obj/* qr
