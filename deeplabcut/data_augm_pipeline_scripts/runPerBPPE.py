from perBodypartPixelError import perBodypartPixelError
for snap in range(0,4):
    (
        testerror_per_bodypart,
        testerror_per_bodypart_pcutoff,
        trainerror_per_bodypart,
        trainerror_per_bodypart_pcutoff,
        testerror_per_bodypart_std,
        testerror_per_bodypart_std_pcutoff,
        trainerror_per_bodypart_std,
        trainerror_per_bodypart_std_pcutoff
    )  = perBodypartPixelError(
            '/home/jonas2/DLC_files/projects/geneva_protocol_paper_austin_2020_bat_data-DLC-2022-07-19/config.yaml',
            shuffle=1,
            snapindex=snap,
            trainFractionIndex = 0,
            modelprefix = "data_augm_00_baseline"
    )
    print("snapshot: " + str(snap+1))
    print(testerror_per_bodypart['ALL'])
    #testerror_per_bodypart_pcutoff
    #trainerror_per_bodypart
    #trainerror_per_bodypart_pcutoff
    #testerror_per_bodypart_std
    #testerror_per_bodypart_std_pcutoff
    #trainerror_per_bodypart_std
    #trainerror_per_bodypart_std_pcutoff