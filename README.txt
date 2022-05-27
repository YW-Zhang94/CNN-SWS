# CNN-SWS


CNN-based Auto-picking method for Shear-wave spitting (SWS) measurements

Versions:
        python 3.6.9
        numpy 1.19.5
        Tensorflow 1.14.0
        Keras 2.2.5

**Please report confusions/errors to yzcd4@umsystem.edu or sgao@mst.edu

The technique is decribed in the paper:


------------------------------------------------

Data structure:

All sac files must be stored at:
~/{data_path}/XKSOut/stname_NW/EQ123456789/stname_NW.ro

{data_path}:
        data_path can be changed at ~/train/2_train/parameter.list for training process and ~/load/2_load/parameter.list for testing process.

XKSOut:
        for PKS phase must be PKSOut.
        for SKS phase must be SKSOut.
        for SKKS phase must be SKKOut.

stname_NW:
        stname is station name. If station name is less than 6 character, shoud be filled by 'x'.
        NW is name of network.

EQ123456789:
        Event name
        12 is for year.
        345 is for Julday.
        67 is for hour.
        89 is for minute.

stname_NW.ro:
        stname_NW is same with previous one.
        ro represent component.
        Each event should have 4 components which are original radial component (ro), original transverse component (to), and corrected radial component (rl), corrected transverse component (tl) gived by minimum XKS energy method (Sliver and Chan, 1991).



The length of sac data should be 50 s with a 0.05 sampling rate and centered at XKS arrival time.



The list of event must be at ~/{data_path}/Out_Bin/XKS.out
        for PKS phase must be PKS.out
        for SKS phase must be SKS.out
        for SKKS phase must be SKK.out
        lists contain 3 columns.
                The First one is stname_NW.
                The second one is EQ123456789.
                The third one is quailty of measurements. A and B are accepted measurements, others are unaccepted measurements （For training process).


--------------------------------------------------

Training process:

Run Do_train.cmd to train CNN.
The parameters can be changed at ~/train/2_train/parameter.list
Link of training dataset in paper: https://figshare.com/articles/dataset/CNN_SWS_data/19904833
        Download under ~/train/ 
        Unzip to use it.

---------------------------------------------------


Testing process:

Run Do_load.cmd to run CNN.
The parameters can be changed at ~/load/2_load/parameter.list
CNN model of paper is at ~/model/CNN_XKS.h5
Output is at ~/load/2_load/Outp
        The output of each event contain 2 numbers.
        The first one represents the possibility of accepeted measurement.
        The seconded one represents the possibility of unaccepeted measurement.
Link of testing dataset in paper: https://figshare.com/articles/dataset/CNN_SWS_data/19904833
        Download under ~/load/
        Unzip to use it.

--------------------------------------------------
