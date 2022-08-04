from operator import concat


def getErrorDistribution(config_path, shuffle=0, snapindex=-1, trainFractionIndex = 0, modelprefix = ""):
    """
    This function returns the distributions of the euclidian distances for the validation data.
    Both the train error, the test error, and the total error. The return structure looks like this
   
    |inddex|bodypart1_error|bodypart2_error|...
    | 138  |     3.63      |     1.82      |...
    | 420  |     1.41      |     2.69      |...
    | 666  |     2.03      |     0.85      |...
    | ...  |     ...       |     ...       |...


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

    ErrorDistribution_all = RMSE.iloc[:]
    ErrorDistribution_test = RMSE.iloc[testIndices]
    ErrorDistribution_train = RMSE.iloc[trainIndices]
    
    ErrorDistributionPCutOff_all = RMSEpcutoff.iloc[:]
    ErrorDistributionPCutOff_test = RMSEpcutoff.iloc[testIndices]
    ErrorDistributionPCutOff_train = RMSEpcutoff.iloc[testIndices]

    return  ErrorDistribution_all, ErrorDistribution_test,\
            ErrorDistribution_train, ErrorDistributionPCutOff_all,\
            ErrorDistributionPCutOff_test, ErrorDistributionPCutOff_train