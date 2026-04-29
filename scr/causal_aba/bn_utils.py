from pathlib import Path
from ArgCausalDisco.utils.data_utils import load_bnlearn_data_dag


def get_dataset(dataset_name='cancer',
                seed=42,
                sample_size=5000,
                data_path=Path(__file__).resolve().parents[2] / 'ArgCausalDisco' / 'datasets'):
    X_s, B_true = load_bnlearn_data_dag(dataset_name,
                                        data_path,
                                        sample_size,
                                        seed=seed,
                                        print_info=True,
                                        standardise=True)
    return X_s, B_true
