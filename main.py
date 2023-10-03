import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import datetime
from timeit import default_timer as timer

import pandas as pd
import pathlib
import pytorch_lightning as pl

from loguru import logger
from typing import Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from core.model import compute_cat_emb_dimensions, SparseCNNEncoder
from util.torch import RANDOM_SEED, seed_everything
from util.loader import DataModuleClass


BATCH_SIZE = 2_000

# All feature columns in the data
LABEL_COLUMNS = ["cohort_month", "churned"]

CATEGORICAL_COLUMNS = [
    "country",
    "plan_name",
    "payment_gateway",
]

BOOL_COLUMNS = []

INT_COLUMNS = [
    "age",
    "num_product_uses",
    "num_searches",
    "avg_session_time",
]


def encode_categorical_feature(
    dataset: pd.DataFrame,
    categorical_features: list[str],
    encoders: Optional[list[LabelEncoder]] = None,
) -> tuple[pd.DataFrame, list[LabelEncoder]]:
    if encoders is None:
        encoders = []
        for feat in categorical_features:
            encoder = LabelEncoder()
            encoder.fit(dataset[feat])
            dataset[feat] = encoder.transform(dataset[feat])
            encoders.append(encoder)
    else:
        if len(categorical_features) != len(encoders):
            raise ValueError("length of features do not match number of encoders")

        for feat, encoder in zip(categorical_features, encoders):
            dataset[feat] = encoder.transform(dataset[feat])

    return dataset, encoders


def load_dataset(filename: str) -> tuple[pd.DataFrame, pd.Series]:
    base_data_dir = pathlib.Path("data")
    dataset = pd.read_csv(base_data_dir / filename)
    dataset = dataset.append(dataset)

    for feat in BOOL_COLUMNS:
        dataset[feat] = dataset[feat].apply(lambda x: 1 if x else 0)

    label = dataset[LABEL_COLUMNS]
    dataset = dataset.drop(LABEL_COLUMNS, axis=1)
    return dataset, label


def main() -> None:
    seed_everything()

    logger.info("loading training data")
    features, labels = load_dataset("dataset.csv")

    train_dataset, test_dataset, train_label, test_label = train_test_split(
        features,
        labels,
        test_size=0.1,
        random_state=RANDOM_SEED,
    )

    train_dataset, encoders = encode_categorical_feature(train_dataset, CATEGORICAL_COLUMNS)
    test_dataset, _ = encode_categorical_feature(test_dataset, CATEGORICAL_COLUMNS, encoders)

    logger.info("splitting data into train and test datasets")
    features = CATEGORICAL_COLUMNS + INT_COLUMNS + BOOL_COLUMNS
    num_cols = len(features)
    logger.info(f"number of features: { num_cols }")

    train_dataset = train_dataset[features]
    test_dataset = test_dataset[features]
    train_dataset, val_dataset, train_label, val_label = train_test_split(
        train_dataset,
        train_label,
        test_size=0.1,
        random_state=RANDOM_SEED,
    )

    logger.info("set up categorical embedding dimensions")
    cat_dims = list(train_dataset[CATEGORICAL_COLUMNS].nunique().values)
    cat_dims = [x + 1 for x in cat_dims]
    cat_emb_dims = compute_cat_emb_dimensions(cat_dims)

    logger.info("training model")
    model = SparseCNNEncoder(
        num_features=num_cols,
        num_targets=1,
        cat_idxs=[x for x in range(len(CATEGORICAL_COLUMNS))],
        cat_dims=cat_dims,
        cat_emb_dim=cat_emb_dims,
        hidden_sizes=[256, 64, 64, 64],
    )

    data_module = DataModuleClass(
        num_workers=1,
        batch_size=BATCH_SIZE,
        training_data=train_dataset,
        training_label=train_label,
        val_data=val_dataset,
        val_label=val_label,
        test_data=test_dataset,
        test_label=test_label,
    )
    trainer = pl.Trainer(
        gpus=1,
        deterministic=True,
        max_epochs=20,
        profiler="simple",
        precision=16,
        log_every_n_steps=10,
    )

    start = timer()
    trainer.fit(model=model, datamodule=data_module)
    end = timer()
    elapsed_time = end - start

    logger.info("testing model")
    trainer.test(model=model, datamodule=data_module)
    results = model.test_results

    test_loss = results["avg_test_loss"]
    logger.info(f"test loss: { test_loss }")
    logger.info(f"training time: " + str(datetime.timedelta(seconds=elapsed_time)))


if __name__ == "__main__":
    main()
