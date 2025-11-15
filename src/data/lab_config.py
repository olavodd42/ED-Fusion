"""
Configuração dos 68 testes laboratoriais relevantes para ED-Copilot
Baseado na Tabela 7 do paper original
"""

# Mapeia itemid do MIMIC-IV para os 68 lab tests
# Nota: Estes são itemids de EXEMPLO - você precisa verificar d_labitems.csv
# para mapear corretamente os nomes dos testes aos itemids do seu MIMIC-IV

RELEVANT_LAB_ITEMIDS = {
    # Complete Blood Count (CBC) - 30 min
    'CBC': [
        51221,  # Hematocrit
        51300,  # White Blood Cells
        51222,  # Hemoglobin
        51279,  # Red Blood Cells
        51250,  # MCV (Mean Corpuscular Volume)
        51248,  # MCH (Mean Corpuscular Hemoglobin)
        51249,  # MCHC (Mean Corpuscular Hemoglobin Concentration)
        51277,  # RDW (Red Cell Distribution Width)
        51265,  # Platelet Count
        51146,  # Basophils
        51200,  # Eosinophils
        51244,  # Lymphocytes
        51254,  # Neutrophils
        51256,  # Neutrophils Absolute
        51143,  # Bands
        51252,  # Monocytes
    ],
    
    # Chemistry (CHEM) - 60 min
    'CHEM': [
        51006,  # Urea Nitrogen
        50912,  # Creatinine
        50983,  # Sodium
        50902,  # Chloride
        50882,  # Bicarbonate
        50931,  # Glucose
        50971,  # Potassium
        50868,  # Anion Gap
        50893,  # Calcium, Total
    ],
    
    # Coagulation (COAG) - 48 min
    'COAG': [
        51274,  # PT (Prothrombin Time)
        51237,  # INR (International Normalized Ratio)
        51275,  # PTT (Partial Thromboplastin Time)
    ],
    
    # Urinalysis (UA) - 40 min
    'UA': [
        51094,  # pH (Urine)
        51109,  # Specific Gravity
        51464,  # RBC (Urine)
        51508,  # WBC (Urine)
        51487,  # Epithelial Cells
        51466,  # Hyaline Casts
        51478,  # Ketone
        51506,  # Urobilinogen
        51491,  # Glucose (Urine)
        51486,  # Protein
    ],
    
    # Lactate - 4 min
    'LACTATE': [
        50813,  # Lactate
    ],
    
    # Liver Function Tests (LFTs) - 104 min
    'LFTS': [
        50861,  # Alkaline Phosphatase
        50878,  # AST (Aspartate Aminotransferase)
        50861,  # ALT (Alanine Aminotransferase)
        50885,  # Bilirubin, Total
        50862,  # Albumin
    ],
    
    # Lipase - 100 min
    'LIPASE': [
        50956,  # Lipase
    ],
    
    # Electrolytes (LYTES) - 89 min
    'LYTES': [
        50960,  # Magnesium
        50970,  # Phosphate
    ],
    
    # Cardiovascular (CARDIO) - 122 min
    'CARDIO': [
        50963,  # NT-proBNP
        51002,  # Troponin T
    ],
    
    # Blood Gas - 12 min
    'BLOOD_GAS': [
        50821,  # PO2
        50818,  # PCO2
        50820,  # pH (Blood Gas)
        50804,  # Calculated Total CO2
        50802,  # Base Excess
        50931,  # Glucose (Blood Gas)
        50983,  # Sodium (Whole Blood)
        50971,  # Potassium (Whole Blood)
    ],
    
    # Toxicology (TOX) - 70 min
    'TOX': [
        50386,  # Ethanol
    ],
    
    # Inflammation (INFLAM) - 178 min
    'INFLAM': [
        50910,  # Creatine Kinase (CK)
        50889,  # C-Reactive Protein
    ],
}

# Lista única de todos os itemids relevantes
ALL_RELEVANT_ITEMIDS = []
for group_name, itemids in RELEVANT_LAB_ITEMIDS.items():
    ALL_RELEVANT_ITEMIDS.extend(itemids)

# Remove duplicatas
ALL_RELEVANT_ITEMIDS = list(set(ALL_RELEVANT_ITEMIDS))

# Mapa reverso: itemid -> grupo
ITEMID_TO_GROUP = {}
for group_name, itemids in RELEVANT_LAB_ITEMIDS.items():
    for itemid in itemids:
        ITEMID_TO_GROUP[itemid] = group_name

# Time-costs médios por grupo (em minutos)
GROUP_TIME_COSTS = {
    'CBC': 30,
    'CHEM': 60,
    'COAG': 48,
    'UA': 40,
    'LACTATE': 4,
    'LFTS': 104,
    'LIPASE': 100,
    'LYTES': 89,
    'CARDIO': 122,
    'BLOOD_GAS': 12,
    'TOX': 70,
    'INFLAM': 178,
}

def get_lab_group(itemid: int) -> str:
    """Retorna o grupo de um itemid"""
    return ITEMID_TO_GROUP.get(itemid, 'UNKNOWN')

def get_group_time_cost(group_name: str) -> int:
    """Retorna o time-cost de um grupo (em minutos)"""
    return GROUP_TIME_COSTS.get(group_name, 0)

def verify_itemids_exist(d_labitems_df):
    """
    Verifica quais itemids existem no d_labitems.csv
    Útil para debugging
    """
    available_itemids = set(d_labitems_df['itemid'].unique())
    
    print("🔍 Verificando itemids...")
    missing_by_group = {}
    
    for group_name, itemids in RELEVANT_LAB_ITEMIDS.items():
        missing = [iid for iid in itemids if iid not in available_itemids]
        if missing:
            missing_by_group[group_name] = missing
            print(f"⚠️  {group_name}: {len(missing)} itemids não encontrados")
    
    if not missing_by_group:
        print("✅ Todos os itemids foram encontrados!")
    
    return missing_by_group