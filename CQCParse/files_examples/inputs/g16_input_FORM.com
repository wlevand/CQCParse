%chk=g16_input_FORM
%mem=745MB
#P CAM-B3LYP/aug-cc-pVTZ Opt=VeryTight
        Int=UltraFine SCF=VeryTight

 Title

0 1
O  0.000000 -0.000000  0.006669
C -0.000000  0.000000  1.184582
H  0.926529  0.000000  1.764322
H -0.926529 -0.000000  1.764322

--Link1--
%chk=inputFull_3q
%mem=745MB
#P CAM-B3LYP/aug-cc-pVTZ Freq=(Anharmonic,raman,hpmodes,ReadAnharm)
        Int=UltraFine SCF=VeryTight
        iop(7/33=1) IOp(10/96=2)
        Guess=Read Geom=AllCheck

Spectro=MaxQuanta=3
