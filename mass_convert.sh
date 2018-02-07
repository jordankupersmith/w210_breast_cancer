#!/bin/bash

find . -type f -name \*.dcm > dcm_files.txt

#for x in *.dcm; do dcmj2pnm $x "${x%.dcm}".png; done;  original
##for x in *.dcm; do dcmj2pnm $x "${x%.dcm}".png; done;
#while read LINE; do dcmj2pnm $LINE "${LINE%.dcm}".png; done < dcm_files.txt
#while read LINE; do dcmj2pnm $LINE +oj "${LINE%.dcm}".jpg; done < dcm_files.txt
while read LINE; do dcmj2pnm $LINE +oj "${LINE%.dcm}".png; done < dcm_files.txt

##test it with cd command below


#cd /home/ubuntu/cancer_download/doi/calc-training_p_02584_left_cc/1.3.6.1.4.1.9590.100.1.2.140305054011667036029668430513140552918/1.3.6.1.4.1.9590.100.1.2.11403071312563907128675757372167498700