This is the Picklemaker used for making pickles for DRN validation on official HtoAA samples.
To create the pickles, change the filename/path and the folder path in "preparePickles_Off_HtoAA"
To make any changes in the inputs (or include or exclude some high level features), change the Extract_Off_HtoAA.py file.
Once the changes are made, run the picklemaker as follows
./preparePickles_Off_HtoAA 1 1 1

Note: The pickle maker always produces a 80-20 train-validation split but since the pickles are used only for validation, please ignore this and draw the inference accordingly.
The picklemaker creates directries labelled EBEB (or EEEE) or mixed/EB (or mixed/EE) and separates the cases when both As are in 1) ECAL barrel  or 2) ECAL endcaps or 3) one A in ECAL barrel and one in endcap
The models used for barrel and endcaps is different and hence the inference on these needs to be run separately.
