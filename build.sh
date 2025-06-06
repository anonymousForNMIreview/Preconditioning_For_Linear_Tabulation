export CANTERA_INCLUDE_DIR=/mnt/c/complete/cantera/include
export CANTERA_LIB_DIR=/mnt/c/complete/cantera/build/lib
export ISATAB_SERIAL_DIR=/mnt/c/complete/ISAT/isatab_ser
export ISAT_LIB_DIR=/mnt/c/complete/ISAT/lib
rm *.o
rm *.mod
rm main.exe
icpx -O3 -L$CANTERA_LIB_DIR -I$CANTERA_INCLUDE_DIR -c reactor.cpp -lcantera
ifx -O3 -L$CANTERA_LIB_DIR -I$CANTERA_INCLUDE_DIR -o PaSR.exe PaSR.f90 reactor.o -lstdc++ -lcantera -Bstatic -L$ISAT_LIB_DIR -I$ISATAB_SERIAL_DIR -lisat7_ser -L -mkl -Bdynamic -mkl -lstdc++
