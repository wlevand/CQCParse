%chk=inputFull_3q
%mem=3GB
#P HF/STO-3G Opt=VeryTight
        Int=UltraFine SCF=VeryTight

 Title

0 1
O 0.000000 0.000000 0.000000
C 0.000000 0.000000 1.128200

--Link1--
%chk=inputFull_3q
%mem=3GB
#P HF/STO-3G Freq=(Anharmonic,raman,hpmodes,ReadAnharm)
        Int=UltraFine SCF=VeryTight
        iop(7/33=1) IOp(10/96=2)
        Guess=Read Geom=AllCheck

Spectro=MaxQuanta=3

