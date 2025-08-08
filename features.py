# features.py

# Reference
# Notwell, J. H., & Wood, M. W. (2023). ADMET property prediction through combinations of molecular fingerprints. 
# *arXiv*. https://doi.org/10.48550/arXiv.2310.00174
# https://github.com/maplightrx/MapLight-TDC

import numpy as np
import pandas as pd
import logging
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Avalon.pyAvalonTools import GetAvalonCountFP
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from molfeat.trans.pretrained import PretrainedDGLTransformer
from tqdm import tqdm

# disable RDKit warning messages
RDLogger.DisableLog('rdApp.*')

# Reference
# from https://www.blopig.com/blog/2022/06/how-to-turn-a-molecule-into-a-vector-of-physicochemical-descriptors-using-rdkit/
def get_chosen_descriptors():
    return ['BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1',
    'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 
    'Chi4v', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 
    'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 
    'EState_VSA8', 'EState_VSA9', 'ExactMolWt', 'FpDensityMorgan1', 
    'FpDensityMorgan2', 'FpDensityMorgan3', 'FractionCSP3', 'HallKierAlpha', 
    'HeavyAtomCount', 'HeavyAtomMolWt', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 
    'LabuteASA', 'MaxAbsEStateIndex', 'MaxAbsPartialCharge', 'MaxEStateIndex', 
    'MaxPartialCharge', 'MinAbsEStateIndex', 'MinAbsPartialCharge', 
    'MinEStateIndex', 'MinPartialCharge', 'MolLogP', 'MolMR', 'MolWt', 
    'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 
    'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 
    'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 
    'NumRadicalElectrons', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 
    'NumSaturatedHeterocycles', 'NumSaturatedRings', 'NumValenceElectrons', 
    'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 
    'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 
    'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'RingCount', 'SMR_VSA1', 'SMR_VSA10', 
    'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 
    'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 
    'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 
    'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'VSA_EState1', 
    'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 
    'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'fr_Al_COO', 
    'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 
    'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 
    'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 
    'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 
    'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 
    'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 
    'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 
    'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 
    'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan', 
    'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 
    'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 
    'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime', 
    'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid', 
    'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 
    'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 
    'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 
    'fr_unbrch_alkane', 'fr_urea', 'qed']

def get_all_features(smiles_series: pd.Series, use_gin: bool):
    # SMILES -> RDKit molecule object to calculate features
    logging.info("SMILES -> RDKit molecule object conversion...")
    molecules = [Chem.MolFromSmiles(s) for s in tqdm(smiles_series, desc="Mol Conversion")]

    # list to store features and names
    all_features_list = []
    
    # Morgan fingerprint
    logging.info("Morgan fingerprint calculation...")
    morgan_fp = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) if m else np.zeros(1024) for m in molecules]
    all_features_list.append(np.array(morgan_fp, dtype=np.float32))
    
    # Avalon fingerprint
    logging.info("Avalon fingerprint calculation...")
    avalon_fp = [GetAvalonCountFP(m, nBits=1024) if m else np.zeros(1024) for m in molecules]
    all_features_list.append(np.array([list(fp) for fp in avalon_fp], dtype=np.float32))

    # RDKit descriptor
    logging.info("RDKit descriptor calculation...")
    desc_names = get_chosen_descriptors()
    calculator = MolecularDescriptorCalculator(desc_names)
    rdkit_desc = [calculator.CalcDescriptors(m) if m else [np.nan]*len(desc_names) for m in molecules]
    all_features_list.append(np.array(rdkit_desc, dtype=np.float32))
    
    # GIN feature generation (optional)
    # Reference
    # Fabian, B., et al. (2020). Molecular representation learning with language models and domain-relevant auxiliary tasks. 
    # *arXiv*. https://doi.org/10.48550/arXiv.2011.13230
    # https://github.com/datamol-io/molfeat

    if use_gin:
        logging.info("GIN feature extraction... (may take a while)")
        try:
            transformer = PretrainedDGLTransformer(kind='gin_supervised_infomax', dtype=float)
            gin_features = transformer(smiles_series.tolist())
            all_features_list.append(gin_features.astype(np.float32))
        except Exception as e:
            logging.error(f"GIN feature extraction failed: {e}. Proceeding without GIN features.")

    logging.info("All features combined and preprocessed...")
    X_features = np.concatenate(all_features_list, axis=1).astype(np.float32)
    
    # NaN/Inf value handling (for safety)
    X_features[np.isinf(X_features)] = np.nan
    if np.isnan(X_features).any():
        logging.warning("NaN values found, replacing with mean values.")
        col_mean = np.nanmean(X_features, axis=0)
        inds = np.where(np.isnan(X_features))
        X_features[inds] = np.take(col_mean, inds[1])

    return X_features