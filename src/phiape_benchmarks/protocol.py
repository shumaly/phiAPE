"""Locked protocol used by the published phiAPE TST experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass


METHODS = ("No-APE", "Learned APE", "Sin-APE", "tAPE", "phiAPE")
ARCHIVES = ("uea", "ucr")
DIMENSIONS = (32, 128)
UEA_DATASETS = (
    "ArticularyWordRecognition",
    "AtrialFibrillation",
    "BasicMotions",
    "CharacterTrajectories",
    "Cricket",
    "DuckDuckGeese",
    "EigenWorms",
    "Epilepsy",
    "ERing",
    "EthanolConcentration",
    "FaceDetection",
    "FingerMovements",
    "HandMovementDirection",
    "Handwriting",
    "Heartbeat",
    "InsectWingbeat",
    "JapaneseVowels",
    "Libras",
    "LSST",
    "NATOPS",
    "MotorImagery",
    "PEMS-SF",
    "PenDigits",
    "PhonemeSpectra",
    "RacketSports",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "SpokenArabicDigits",
    "StandWalkJump",
    "UWaveGestureLibrary",
)
UCR_DATASETS = (
    "ACSF1",
    "Adiac",
    "AllGestureWiimoteX",
    "AllGestureWiimoteY",
    "AllGestureWiimoteZ",
    "ArrowHead",
    "BME",
    "Beef",
    "BeetleFly",
    "BirdChicken",
    "CBF",
    "Car",
    "Chinatown",
    "ChlorineConcentration",
    "CinCECGTorso",
    "Coffee",
    "Computers",
    "CricketX",
    "CricketY",
    "CricketZ",
    "Crop",
    "DiatomSizeReduction",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxTW",
    "DodgerLoopDay",
    "DodgerLoopGame",
    "DodgerLoopWeekend",
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
    "EOGHorizontalSignal",
    "EOGVerticalSignal",
    "Earthquakes",
    "ElectricDevices",
    "EthanolLevel",
    "FaceAll",
    "FaceFour",
    "FacesUCR",
    "FiftyWords",
    "Fish",
    "FordA",
    "FordB",
    "FreezerRegularTrain",
    "FreezerSmallTrain",
    "Fungi",
    "GestureMidAirD1",
    "GestureMidAirD2",
    "GestureMidAirD3",
    "GesturePebbleZ1",
    "GesturePebbleZ2",
    "GunPoint",
    "GunPointAgeSpan",
    "GunPointMaleVersusFemale",
    "GunPointOldVersusYoung",
    "Ham",
    "HandOutlines",
    "Haptics",
    "Herring",
    "HouseTwenty",
    "InlineSkate",
    "InsectEPGRegularTrain",
    "InsectEPGSmallTrain",
    "InsectWingbeatSound",
    "ItalyPowerDemand",
    "LargeKitchenAppliances",
    "Lightning2",
    "Lightning7",
    "Mallat",
    "Meat",
    "MedicalImages",
    "MelbournePedestrian",
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxTW",
    "MixedShapesRegularTrain",
    "MixedShapesSmallTrain",
    "MoteStrain",
    "NonInvasiveFetalECGThorax1",
    "NonInvasiveFetalECGThorax2",
    "OSULeaf",
    "OliveOil",
    "PLAID",
    "PhalangesOutlinesCorrect",
    "Phoneme",
    "PickupGestureWiimoteZ",
    "PigAirwayPressure",
    "PigArtPressure",
    "PigCVP",
    "Plane",
    "PowerCons",
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxTW",
    "RefrigerationDevices",
    "Rock",
    "ScreenType",
    "SemgHandGenderCh2",
    "SemgHandMovementCh2",
    "SemgHandSubjectCh2",
    "ShakeGestureWiimoteZ",
    "ShapeletSim",
    "ShapesAll",
    "SmallKitchenAppliances",
    "SmoothSubspace",
    "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2",
    "StarLightCurves",
    "Strawberry",
    "SwedishLeaf",
    "Symbols",
    "SyntheticControl",
    "ToeSegmentation1",
    "ToeSegmentation2",
    "Trace",
    "TwoLeadECG",
    "TwoPatterns",
    "UMD",
    "UWaveGestureLibraryAll",
    "UWaveGestureLibraryX",
    "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ",
    "Wafer",
    "Wine",
    "WordSynonyms",
    "Worms",
    "WormsTwoClass",
    "Yoga",
)


@dataclass(frozen=True)
class Protocol:
    epochs: int = 200
    repeats: int = 16
    first_model_seed: int = 2026
    split_seed: int = 2026
    validation_fraction: float = 0.2
    batch_size: int = 64
    early_stopping_patience: int = 32
    maximum_length: int = 512
    encoder_layers: int = 3
    attention_heads: int = 8
    feedforward_dimension: int = 256
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 4.0
    validation_label_smoothing: float = 0.05
    scale_embedding: bool = True

    @property
    def model_seeds(self) -> tuple[int, ...]:
        return tuple(range(self.first_model_seed, self.first_model_seed + self.repeats))

    def repeat_for_seed(self, seed: int) -> int:
        if seed not in self.model_seeds:
            raise ValueError(f"seed must be one of {self.model_seeds}; got {seed}")
        return seed - self.first_model_seed + 1

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


PROTOCOL = Protocol()

RESULT_COLUMNS = (
    "dataset",
    "train_cases",
    "test_cases",
    "channels",
    "sequence_length",
    "original_length",
    "classes",
    "embedding_size",
    "positional_encoding",
    "repeat",
    "seed",
    "val_accuracy",
    "test_loss",
    "test_correct",
    "test_accuracy",
)
