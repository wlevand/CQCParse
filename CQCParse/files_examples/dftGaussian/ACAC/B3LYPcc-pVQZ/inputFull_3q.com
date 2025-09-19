%chk=inputFull_3q
%mem=745MB
#P B3LYP/cc-pVQZ Opt=(maxcycle=150,cartesian,VeryTight)
        Int=UltraFine SCF=VeryTight

 Title

0 1
C                 -0.09668831   -0.10218975    0.00000000
C                  1.38206335    0.15859430    0.00000000
O                 -0.82042688    0.99931050    0.00000000
H                  1.64909085    0.74739900   -0.87680170
H                  1.64909085    0.74739900    0.87680170
H                  1.92630450   -0.77928644    0.00000000
O                 -0.57213563   -1.22871207    0.00000000
H                 -1.79623637    0.78127373    0.00000000

--Link1--
%chk=inputFull_3q
%mem=745MB
#P B3LYP/cc-pVQZ Freq=(Anharmonic,raman,hpmodes,ReadAnharm)
        Int=UltraFine SCF=VeryTight
        iop(7/33=1) IOp(10/96=2)
        Guess=Read Geom=AllCheck

Spectro=MaxQuanta=3

