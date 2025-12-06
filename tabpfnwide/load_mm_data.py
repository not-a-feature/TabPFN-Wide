import os
import pandas as pd
import joblib
from sklearn.decomposition import SparsePCA
from sklearn.preprocessing import scale
from functools import lru_cache

ALL_MULTIOMICS_DATASETS = set(['BRCA', 'COAD', 'GBM', 'LGG', 'OV'])
ALL_MULTIOMICS_DATASETS_SHAMIR = set(['aml', 'breast', 'colon', 'gbm', 'kidney', 'liver', 'lung', 'melanoma', 'ovarian', 'sarcoma'])

@lru_cache(maxsize=1)
def load_multiomics_benchmark_ds(dataset, preprocessing, dir_path=None):
    assert dataset in ALL_MULTIOMICS_DATASETS, f"Dataset {dataset} not in {ALL_MULTIOMICS_DATASETS}"
    assert preprocessing in ['Aligned', 'Original'], f"Preprocessing {preprocessing} not in ['Aligned', 'Original']"
    file_appendix = f"_{preprocessing.lower()}" if preprocessing != "Original" else ""
    
    if dir_path is None:
        dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'benchmark_data', 'multiomics_benchmark_data')
    path_to_benchmark_ds_pre = os.path.join(dir_path, "Cancer-Multi-Omics-Benchmark", "Main_Dataset", 
                                        "Classification_datasets", f"GS-{dataset}", preprocessing)

    cnv_data = pd.read_csv(os.path.join(path_to_benchmark_ds_pre, f"{dataset}_CNV{file_appendix}.csv"), index_col=0)
    # Remove rows with all zeros
    cnv_data = cnv_data.loc[~(cnv_data == 0).all(axis=1)]
    # Remove duplicated rows and keep the first occurrence
    cnv_data = cnv_data[~cnv_data.index.duplicated(keep='first')]
    cnv_data = cnv_data.T
    # Fill NaN values with the mean of each column
    cnv_data.fillna(cnv_data.mean(), inplace=True)
    cnv_data.columns = cnv_data.columns.astype(str)

    methylation_data = pd.read_csv(os.path.join(path_to_benchmark_ds_pre, f"{dataset}_Methy{file_appendix}.csv"), index_col=0)
    methylation_data = methylation_data.loc[~(methylation_data == 0).all(axis=1)]
    methylation_data = methylation_data[~methylation_data.index.duplicated(keep='first')]
    methylation_data = methylation_data.T
    methylation_data.fillna(methylation_data.mean(), inplace=True)
    methylation_data.columns = methylation_data.columns.astype(str)

    rna_data = pd.read_csv(os.path.join(path_to_benchmark_ds_pre, f"{dataset}_mRNA{file_appendix}.csv"), index_col=0)
    rna_data = rna_data.loc[~(rna_data == 0).all(axis=1)]
    rna_data = rna_data[~rna_data.index.duplicated(keep='first')]
    rna_data = rna_data.T
    rna_data.fillna(rna_data.mean(), inplace=True)
    rna_data.columns = rna_data.columns.astype(str)

    # For miRNA, we use the original data as there is no high-dimensional version
    mirna_data = pd.read_csv(os.path.join(path_to_benchmark_ds_pre, f"{dataset}_miRNA{file_appendix}.csv"), index_col=0)
    mirna_data = mirna_data.loc[~(mirna_data == 0).all(axis=1)]    
    mirna_data = mirna_data[~mirna_data.index.duplicated(keep='first')]
    mirna_data = mirna_data.T
    mirna_data.fillna(mirna_data.mean(), inplace=True)
    mirna_data.columns = mirna_data.columns.astype(str)

    labels = pd.read_csv(os.path.join(path_to_benchmark_ds_pre, f"{dataset}_label_num.csv"))
    
    return {
        'cnv': cnv_data,
        'methylation': methylation_data,
        'mrna': rna_data,
        'mirna': mirna_data,
        # Leave these for compatibility with existing code
        'original_cnv': cnv_data,
        'original_methylation': methylation_data,
        'original_mrna': rna_data,
        'original_mirna': mirna_data,
        'labels' : labels.values.flatten(),
    }
    

@lru_cache(maxsize=1)
def load_multiomics_benchmark_shamir(dataset, normalize, aligned=True, subtype_labels=False):
    assert dataset in ALL_MULTIOMICS_DATASETS_SHAMIR, f"Dataset {dataset} not in {ALL_MULTIOMICS_DATASETS_SHAMIR}"
    shamir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'benchmark_data', 'shamir_data', 'Multi_Omics_Cancer_Benchmark_TCGA_Shamir')
    dir_path = os.path.join(shamir_path, dataset)
    
    mrna_data = pd.read_table(os.path.join(dir_path, "exp"), header=0, index_col=0, sep=r"\s+")
    mrna_data = mrna_data.T
    mrna_data = mrna_data[~mrna_data.index.duplicated(keep='first')]
    assert pd.isna(mrna_data).sum().sum() == 0
    if normalize:
        # Normalize the data
        mrna_data[:] = scale(mrna_data)
        
    methylation_data = pd.read_table(os.path.join(dir_path, "methy"), header=0, index_col=0, sep=r"\s+")
    methylation_data = methylation_data.T
    methylation_data = methylation_data[~methylation_data.index.duplicated(keep='first')]
    illumina_mapping = pd.read_csv(os.path.join(shamir_path, "methylation_gene_mapping.csv"), skiprows=7, index_col=0, dtype=str)
    gene_names = illumina_mapping["UCSC_RefGene_Name"].dropna()
    gene_names = gene_names.str.split(";").apply(lambda x: x[0] if len(x) > 0 else None).to_dict()
    new_columns = [gene_names.get(col, col) for col in methylation_data.columns]
    methylation_data.columns = new_columns
    assert pd.isna(methylation_data).sum().sum() == 0
    if normalize:
        # Normalize the data
        methylation_data[:] = scale(methylation_data)
        
    mirna_data = pd.read_table(os.path.join(dir_path, "mirna"), header=0, index_col=0, sep=r"\s+")
    mirna_data = mirna_data.T
    mirna_data = mirna_data[~mirna_data.index.duplicated(keep='first')]
    mirna_data.columns = [col.replace("-", ".") for col in mirna_data.columns]
    assert pd.isna(mirna_data).sum().sum() == 0
    if normalize:
        # Normalize the data
        mirna_data[:] = scale(mirna_data)
    

    if subtype_labels and dataset in ["gbm", "breast", "colon", "kidney", "sarcoma"]:
        if dataset == "colon":
            st_labels = pd.read_csv(os.path.join(dir_path, "subtype_labels"), index_col=0, skiprows=1)
            st_labels.index = st_labels.index.str.replace("-", ".")
            labels = st_labels
        elif dataset == "kidney":
            raise ValueError("Kidney dataset only consists of clear cell subtype, no subtype labels available.")
        else:
            clinical_data = pd.read_table(os.path.join(dir_path, dataset))
            clinical_data["sampleID"] = clinical_data["sampleID"].str.replace("-", ".")
            clinical_data.index = clinical_data["sampleID"]
            label_column = {
                "gbm": "GeneExp_Subtype",
                "breast": "PAM50Call_RNAseq",
                "sarcoma": "histological_type"
            }[dataset]
            labels = clinical_data[label_column]
            labels.dropna(inplace=True)
    else:
        labels = pd.read_table(os.path.join(dir_path, "survival"), header=0, index_col=0, sep=r"\s+")
        labels = labels[~labels.index.duplicated(keep='first')]

    
    if aligned:
        intersection_omic = list(
            set(mrna_data.index)
            .intersection(set(methylation_data.index))
            .intersection(set(mirna_data.index))
        )

        sample_delimiter_label = "." if "." in labels.index[0] else "-"
        sample_len_label = len(labels.index[0].split(sample_delimiter_label))
        for sample in labels.index:
            assert len(sample.split(sample_delimiter_label)) == sample_len_label, f"Sample {sample} does not have the expected format"
            
        sample_delimiter_omic = "." if "." in intersection_omic[0] else "-"
        sample_len_omic = len(intersection_omic[0].split(sample_delimiter_omic))
        for sample in intersection_omic:    
            assert len(sample.split(sample_delimiter_omic)) == sample_len_omic, f"Sample {sample} does not have the expected format"

        match_length = min(sample_len_label, sample_len_omic)
        standardized_all_omic = { sample_delimiter_omic.join(sample.split(sample_delimiter_omic)[:match_length]).lower(): sample for sample in intersection_omic}
        standardized_labels = { sample_delimiter_omic.join(sample.split(sample_delimiter_label)[:match_length]).lower(): sample for sample in labels.index}

        intersection = set(standardized_all_omic.keys()).intersection(set(standardized_labels.keys()))
        intersection_all_omic = [standardized_all_omic[sample] for sample in intersection]
        intersection_all_labels = [standardized_labels[sample] for sample in intersection]

        assert len(intersection_all_omic) == len(intersection_all_labels), f"Number of samples in omics {len(intersection_all_omic)} and labels {len(intersection_all_labels)} do not match" 

        return {
            'methylation': methylation_data.loc[intersection_all_omic],
            'mrna': mrna_data.loc[intersection_all_omic],
            'mirna': mirna_data.loc[intersection_all_omic],
            'labels' : labels.loc[intersection_all_labels]["Death"].values if not subtype_labels else labels.loc[intersection_all_labels].values,
            # Leave these for compatibility with existing code
            'aligned_methylation': methylation_data.loc[intersection_all_omic],
            'aligned_mrna': mrna_data.loc[intersection_all_omic],
            'aligned_mirna': mirna_data.loc[intersection_all_omic],
            'aligned_labels' : labels.loc[intersection_all_labels]["Death"].values if not subtype_labels else labels.loc[intersection_all_labels].values,
        }
    
    result = {}
    
    for omic, data in zip(
        ['methylation', 'mrna', 'mirna'],
        [methylation_data, mrna_data, mirna_data]
    ):
        intersection_omic = [sample for sample in data.index if ".".join(sample.split(".")[:3]).lower() in labels.index.str.lower()]
        intersection_labels = [".".join(sample.split(".")[:3]).lower() for sample in intersection_omic]
        
        assert len(intersection_omic) == len(intersection_labels), f"Number of samples in omics {len(intersection_omic)} and labels {len(intersection_labels)} do not match"
        result[f'original_{omic}'] = data.loc[intersection_omic]
        result[f'labels_{omic}'] = labels.loc[intersection_labels]
        
    return result


def load_multiomics(dataset, **kwargs):
    if dataset in ALL_MULTIOMICS_DATASETS:
        dict_ds = load_multiomics_benchmark_ds(dataset, preprocessing=kwargs.get('preprocessing', 'Original'), **kwargs)
        labels = dict_ds['labels']
        return dict_ds, labels
    if dataset in ALL_MULTIOMICS_DATASETS_SHAMIR:
        dict_ds = load_multiomics_benchmark_shamir(
            dataset,
            normalize=kwargs.get('normalize', True), 
            subtype_labels=kwargs.get('subtype_labels', True),
            **kwargs)
        labels = dict_ds['labels']
        return dict_ds, labels
    raise ValueError(f"Dataset {dataset} not recognized. Available datasets: {ALL_MULTIOMICS_DATASETS.union(ALL_MULTIOMICS_DATASETS_SHAMIR)}")
        