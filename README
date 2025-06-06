This is the source code used to generate the data reported in "Hessian Preconditioning for Piecewise Linear Tabulation of Chemical Kinetics".

INSTALLATION INSTRUCTIONS

1) Install ISAT (available at https://tcg.mae.cornell.edu/ISATCK7/)

   ISAT installation notes: As downloaded from the above source, ISAT's build system is written with Intel's Fortran compilers in mind,
   and requires a link to MKL. The user needs to set the environment variable TACC_MKL_LIB to the path where MKL is installed. Not every package
   in the ISAT-CK7 project needs to be installed (some of them require Chemkin), just ISATAB and its dependencies.
   
2) Install Cantera (version 3.1 or newer, Anaconda recommended for the installation process)

3) Download the Cantera example code "custom.cpp" from https://cantera.org/3.1/examples/cxx/custom.html (accessed 06/05/2025) and copy
   lines 21-147 into a new file named "custom.hpp" and located in this directory (the code in question is freely available online but copyrighted,
   hence why it is not included in this repository)

4) Change the directory paths on the first 4 lines of "build.sh" to the respective paths where ISAT and Cantera are built on your machine

5) Run "build.sh"

6) If training the preconditioning multilayer perceptron (MLP) network, install PyTorch

USING THE CODE

1) To generate MLP training data:
   - Set mode = 1 on line 49 of "PaSR.f90" and compile the code
   - Run the command:
      ./PaSR.exe > out
     to store the output of PaSR.exe in the file "out"
   - Run the MATLAB script "PaSRtoMLP.m", which will generate a file "data.csv" with 40,000 training samples
   
2) To train the MLP:
   - Set the desired number of hidden layer neurons on line 21 of "trainMLP.py"
   - On lines 22-23, set SEP-ISAT=1, IP-ISAT=0 if you want to train SEP-preconditioning, or set
     SEP-ISAT=0, IP-ISAT=1 if you want to train IP-preconditioning
   - run "trainMLP.py"
   - run the MATLAB script "convertMLPweights.m"
   
3) To run a preconditioned ISAT PaSR simulation:
   - Either perform steps 1) and 2), or copy the "A*.csv" and "B*.csv" from one of the directories "IP-I-10", "SEP-I-10", etc.
   into this directory (the number in the directory name indicates the number of hidden neurons in the network)
   - Change "nNeurons" on line 12 of "reactor.cpp" to the appropriate number of neurons in your MLP network
   - In "PaSR.f90", set "mode = 2" on line 49, and on line 56 set the error tolerance you would like ISAT to enforce
   - Compile the code and run "PaSR.exe" (performance statistics are output at the end of the code's execution)
   
4) To run a standalone ISAT simulation:
   - It is not necessary to perform the MLP training steps
   - In "PaSR.f90", set "mode = 3" on line 49, and on line 56 set the error tolerance you would like ISAT to enforce
   - Compile the code and run "PaSR.exe" (performance statistics are output at the end of the code's execution)
   
5) To run a standalone MLP simulation:
   - Either perform steps 1) and 2), or copy the "A*.csv" and "B*.csv" from one of the directories "IP-I-10", "SEP-I-10", etc.
   into this directory (the number in the directory name indicates the number of hidden neurons in the network)
   - Change "nNeurons" on line 12 of "reactor.cpp" to the appropriate number of neurons in your MLP network
   - In "PaSR.f90", set "mode = 4" on line 49
   - Compile the code and run "PaSR.exe" (performance statistics are output at the end of the code's execution)
   
   
Any inquiries regarding this code can be addressed to Pavel Popov at ppopov@sdsu.edu
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
      
