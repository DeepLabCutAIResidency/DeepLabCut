def perBodypartPixelError(config_path, shuffle=0, snapindex=-1, trainFractionIndex = 0, modelprefix = ""):
    """
    Uses result from evaluate_network to calculate per bodypart pixel error.

    ----------
    config : string
        Full path of the config.yaml file as a string.

    shuffle: integer
        integers specifying shuffle index of the training dataset. The default is 0.

    numdigits: number of digits to round for distances. #leaving in as I might use this...
    """
    import deeplabcut # need to figure out how to access without importing, like, I shouldn't have to import deeplabcut coz this is deeplabcut

    from deeplabcut.pose_estimation_tensorflow.config import load_config
    from deeplabcut.pose_estimation_tensorflow.core import evaluate
    from deeplabcut.utils import auxiliaryfunctions
    import os
    import pandas as pd
    import numpy as np
    from pathlib import Path

    cfg = deeplabcut.auxiliaryfunctions.read_config(config_path) # get the project configuration
    trainingsetfolder = deeplabcut.auxiliaryfunctions.GetTrainingSetFolder(cfg) 

    # this is the human labeled data
    Data = pd.read_hdf(
        os.path.join(
            cfg["project_path"],
            str(trainingsetfolder),
            "CollectedData_" + cfg["scorer"] + ".h5",
        )
    )

    trainFraction = cfg["TrainingFraction"][trainFractionIndex] # whic training fraction are we testing?

    # get the folder where the evaulation result is stored
    evaluationfolder = os.path.join(
        cfg["project_path"],
        str(
            deeplabcut.auxiliaryfunctions.get_evaluation_folder(
                trainFraction, shuffle, cfg, modelprefix=modelprefix
            )
        ),
    )

    # get the folder of the model we are testing
    modelfolder = os.path.join(
        cfg["project_path"],
        str(
            deeplabcut.auxiliaryfunctions.get_model_folder(
                trainFraction, shuffle, cfg, modelprefix=modelprefix
            )
        ),
    )

    Snapshots = np.array(
        [
            fn.split(".")[0]
            for fn in os.listdir(os.path.join(str(modelfolder), "train"))
            if "index" in fn
        ]
    )

    increasing_indices = np.argsort(
        [int(m.split("-")[1]) for m in Snapshots]
    )
    Snapshots = Snapshots[increasing_indices]

    path_test_config = Path(modelfolder) / "test" / "pose_cfg.yaml"
    dlc_cfg = load_config(str(path_test_config))

    dlc_cfg["init_weights"] = os.path.join(
        str(modelfolder), "train", Snapshots[snapindex]
    )  # setting weights to corresponding snapshot.

    trainingsiterations = (dlc_cfg["init_weights"].split(os.sep)[-1]).split("-")[
        -1
    ]  # read how many training siterations that corresponds to.

    DLCscorer, DLCscorerlegacy = auxiliaryfunctions.GetScorerName(
                cfg, shuffle, trainFraction, trainingsiterations, modelprefix=modelprefix
    )

    (
        notanalyzed,
        resultsfilename,
        DLCscorer,
    ) = auxiliaryfunctions.CheckifNotEvaluated(
            str(evaluationfolder), DLCscorer, DLCscorerlegacy, Snapshots[snapindex]
    )

    if notanalyzed:
        print("Model not trained/evaluated!")
        print("Make sure a model with the provided input is trained (train_network) and evaluated (evaluate_network)!")
        return 0, 0, 0, 0
        # what's the proper way to exit here?
    

    DataMachine = pd.read_hdf(resultsfilename)

    DataCombined = pd.concat([Data.T, DataMachine.T], axis=0).T

    RMSE, RMSEpcutoff = evaluate.pairwisedistances(
        DataCombined,
        cfg["scorer"],
        DLCscorer,
        cfg["pcutoff"]
    )

    # get the metadata filename
    _, metadatafn = deeplabcut.auxiliaryfunctions.GetDataandMetaDataFilenames(
        trainingsetfolder, trainFraction, shuffle, cfg
    )
    _, trainIndices, testIndices, _ = deeplabcut.auxiliaryfunctions.LoadMetadata(
        os.path.join(cfg["project_path"], metadatafn)
    )

    # return values

    testerror_all_mean = np.nanmean(RMSE.iloc[testIndices].values.flatten())
    testerror_all_mean_pcutoff = np.nanmean(RMSE.iloc[testIndices].values.flatten())

    trainerror_all_mean = np.nanmean(RMSE.iloc[trainIndices].values.flatten())
    trainerror_all_mean_pcutoff = np.nanmean(RMSE.iloc[trainIndices].values.flatten())

    testerror_per_bodypart_mean = pd.DataFrame([np.nanmean(RMSE.iloc[testIndices].values, axis=0)], columns=cfg["bodyparts"])
    testerror_per_bodypart_mean['ALL'] = testerror_all_mean

    testerror_per_bodypart_mean_pcutoff = pd.DataFrame([np.nanmean(RMSEpcutoff.iloc[testIndices].values, axis=0)], columns=cfg["bodyparts"])
    testerror_per_bodypart_mean_pcutoff['ALL'] = testerror_all_mean_pcutoff

    trainerror_per_bodypart_mean = pd.DataFrame([np.nanmean(RMSE.iloc[trainIndices].values, axis=0)], columns=cfg["bodyparts"])
    trainerror_per_bodypart_mean['ALL'] = trainerror_all_mean

    trainerror_per_bodypart_mean_pcutoff = pd.DataFrame([np.nanmean(RMSEpcutoff.iloc[trainIndices].values, axis=0)], columns=cfg["bodyparts"])
    trainerror_per_bodypart_mean_pcutoff['ALL'] = trainerror_all_mean_pcutoff

    testerror_all_std = np.nanstd(RMSE.iloc[testIndices].values.flatten())
    testerror_all_std_pcutoff = np.nanstd(RMSE.iloc[testIndices].values.flatten())

    trainerror_all_std = np.nanstd(RMSE.iloc[trainIndices].values.flatten())
    trainerror_all_std_pcutoff = np.nanstd(RMSE.iloc[trainIndices].values.flatten())

    testerror_per_bodypart_std = pd.DataFrame([np.nanstd(RMSE.iloc[testIndices].values, axis=0)], columns=cfg["bodyparts"])
    testerror_per_bodypart_std['ALL'] = testerror_all_std

    testerror_per_bodypart_std_pcutoff = pd.DataFrame([np.nanstd(RMSEpcutoff.iloc[testIndices].values, axis=0)], columns=cfg["bodyparts"])
    testerror_per_bodypart_std_pcutoff['ALL'] = testerror_all_std_pcutoff

    trainerror_per_bodypart_std = pd.DataFrame([np.nanstd(RMSE.iloc[trainIndices].values, axis=0)], columns=cfg["bodyparts"])
    trainerror_per_bodypart_std['ALL'] = trainerror_all_std

    trainerror_per_bodypart_std_pcutoff = pd.DataFrame([np.nanstd(RMSEpcutoff.iloc[trainIndices].values, axis=0)], columns=cfg["bodyparts"])
    trainerror_per_bodypart_std_pcutoff['ALL'] = trainerror_all_std_pcutoff

    return  testerror_per_bodypart_mean, testerror_per_bodypart_mean_pcutoff,\
            trainerror_per_bodypart_mean, trainerror_per_bodypart_mean_pcutoff,\
            testerror_per_bodypart_std, testerror_per_bodypart_std_pcutoff, \
            trainerror_per_bodypart_std, trainerror_per_bodypart_std_pcutoff

